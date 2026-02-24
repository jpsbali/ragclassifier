from typing import Any, TypedDict
import time

from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import END, START, StateGraph

from src.agents import (
    build_chat_model,
    build_reconciliation_guidance,
    classify_with_agent,
    finalize_with_supervisor,
)
from src.config import AppConfig
from src.models import AgentVote, RoundHistory, SupervisorDecision, TokenUsage


class GraphState(TypedDict, total=False):
    document_id: str
    document_name: str
    document_text: str
    round_num: int
    retry_context: str
    agent_a_vote: dict[str, Any]
    agent_b_vote: dict[str, Any]
    consensus_reached: bool
    consensus_score: float
    should_finalize: bool
    token_usages: list[dict[str, Any]]
    supervisor_decision: dict[str, Any]
    history: list[dict[str, Any]]
    agent_a_duration: float
    agent_b_duration: float
    supervisor_duration: float


def _timed_classify(llm, document_name, document_text, round_num, retry_context):
    start = time.time()
    vote, usage = classify_with_agent(
        llm, document_name, document_text, round_num, retry_context
    )
    duration = time.time() - start
    return vote, usage, duration


class DocumentClassifier:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.supervisor_llm = build_chat_model(config.supervisor, config.use_openrouter)
        self.agent_a_llm = build_chat_model(config.agent_a, config.use_openrouter)
        self.agent_b_llm = build_chat_model(config.agent_b, config.use_openrouter)
        self.graph = self._build_graph()

    def _run_agents(self, state: GraphState) -> GraphState:
        round_num = state.get("round_num", 1)
        retry_context = state.get("retry_context")
        document_name = state["document_name"]
        document_text = state["document_text"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            agent_a_future = executor.submit(
                _timed_classify,
                llm=self.agent_a_llm,
                document_name=document_name,
                document_text=document_text,
                round_num=round_num,
                retry_context=retry_context,
            )
            agent_b_future = executor.submit(
                _timed_classify,
                llm=self.agent_b_llm,
                document_name=document_name,
                document_text=document_text,
                round_num=round_num,
                retry_context=retry_context,
            )
            agent_a_vote, agent_a_tokens, duration_a = agent_a_future.result()
            agent_b_vote, agent_b_tokens, duration_b = agent_b_future.result()

        # Update cumulative durations
        total_a = state.get("agent_a_duration", 0.0) + duration_a
        total_b = state.get("agent_b_duration", 0.0) + duration_b

        # Calculate costs for history
        def _calc_cost(usage, cfg):
            return (usage.prompt_tokens / 1_000_000 * cfg.input_cost_per_m) + \
                   (usage.completion_tokens / 1_000_000 * cfg.output_cost_per_m)
        
        cost_a = _calc_cost(agent_a_tokens, self.config.agent_a)
        cost_b = _calc_cost(agent_b_tokens, self.config.agent_b)

        current_usages = state.get("token_usages", [])
        
        usage_a = agent_a_tokens.model_dump()
        usage_a["source"] = "agent_a"
        usage_b = agent_b_tokens.model_dump()
        usage_b["source"] = "agent_b"
        current_usages.extend([usage_a, usage_b])

        # Record history for this round
        history_entry = {
            "round": round_num,
            "agent_a": agent_a_vote.model_dump(),
            "agent_b": agent_b_vote.model_dump(),
            "agent_a_duration_s": duration_a,
            "agent_b_duration_s": duration_b,
            "agent_a_token_usage": agent_a_tokens.model_dump(),
            "agent_b_token_usage": agent_b_tokens.model_dump(),
            "agent_a_cost": cost_a,
            "agent_b_cost": cost_b,
        }
        updated_history = state.get("history", []) + [history_entry]

        return {
            "agent_a_vote": agent_a_vote.model_dump(),
            "agent_b_vote": agent_b_vote.model_dump(),
            "token_usages": current_usages,
            "history": updated_history,
            "agent_a_duration": total_a,
            "agent_b_duration": total_b,
        }

    def _evaluate_votes(self, state: GraphState) -> GraphState:
        agent_a = AgentVote.model_validate(state["agent_a_vote"])
        agent_b = AgentVote.model_validate(state["agent_b_vote"])

        labels_match = agent_a.classification == agent_b.classification
        consensus_score = (agent_a.confidence + agent_b.confidence) / 2.0
        consensus_reached = (
            labels_match and consensus_score >= self.config.consensus.min_confidence
        )

        round_num = state.get("round_num", 1)
        should_finalize = consensus_reached or round_num >= self.config.consensus.max_rounds

        return {
            "consensus_reached": consensus_reached,
            "consensus_score": consensus_score,
            "should_finalize": should_finalize,
        }

    def _route_after_evaluation(self, state: GraphState) -> str:
        return "finalize" if state.get("should_finalize") else "reconcile"

    def _reconcile(self, state: GraphState) -> GraphState:
        round_num = state.get("round_num", 1)
        agent_a = AgentVote.model_validate(state["agent_a_vote"])
        agent_b = AgentVote.model_validate(state["agent_b_vote"])

        start = time.time()
        guidance, recon_tokens = build_reconciliation_guidance(
            supervisor_llm=self.supervisor_llm,
            document_name=state["document_name"],
            document_text=state["document_text"],
            round_num=round_num,
            agent_a_vote=agent_a,
            agent_b_vote=agent_b,
        )
        duration = time.time() - start
        total_sup = state.get("supervisor_duration", 0.0) + duration

        current_usages = state.get("token_usages", [])
        usage = recon_tokens.model_dump()
        usage["source"] = "supervisor"
        current_usages.append(usage)

        return {
            "retry_context": guidance.instructions_for_retry,
            "round_num": round_num + 1,
            "token_usages": current_usages,
            "supervisor_duration": total_sup,
        }

    def _finalize(self, state: GraphState) -> GraphState:
        agent_a = AgentVote.model_validate(state["agent_a_vote"])
        agent_b = AgentVote.model_validate(state["agent_b_vote"])

        start = time.time()
        final_decision, finalizer_tokens = finalize_with_supervisor(
            supervisor_llm=self.supervisor_llm,
            document_id=state["document_id"],
            document_name=state["document_name"],
            rounds_used=state.get("round_num", 1),
            consensus_reached=state.get("consensus_reached", False),
            consensus_score=state.get("consensus_score", 0.0),
            agent_a_vote=agent_a,
            agent_b_vote=agent_b,
        )
        duration = time.time() - start
        total_sup = state.get("supervisor_duration", 0.0) + duration

        # Ensure rounds_used is accurately reflected from state
        final_decision.rounds_used = state.get("round_num", 1)

        # Populate cumulative durations
        final_decision.supervisor_duration_s = total_sup
        final_decision.agent_a_duration_s = state.get("agent_a_duration", 0.0)
        final_decision.agent_b_duration_s = state.get("agent_b_duration", 0.0)

        # Populate history
        raw_history = state.get("history", [])
        final_decision.history = [
            RoundHistory(
                round=h["round"],
                agent_a=AgentVote.model_validate(h["agent_a"]),
                agent_b=AgentVote.model_validate(h["agent_b"]),
                agent_a_duration_s=h.get("agent_a_duration_s", 0.0),
                agent_b_duration_s=h.get("agent_b_duration_s", 0.0),
                agent_a_token_usage=TokenUsage(**h.get("agent_a_token_usage", {})) if h.get("agent_a_token_usage") else None,
                agent_b_token_usage=TokenUsage(**h.get("agent_b_token_usage", {})) if h.get("agent_b_token_usage") else None,
                agent_a_cost=h.get("agent_a_cost", 0.0),
                agent_b_cost=h.get("agent_b_cost", 0.0),
            )
            for h in raw_history
        ]

        current_usages = state.get("token_usages", [])
        usage = finalizer_tokens.model_dump()
        usage["source"] = "supervisor"
        current_usages.append(usage)

        # Calculate estimated cost
        total_cost = 0.0
        sup_cost = 0.0
        a_cost = 0.0
        b_cost = 0.0
        for u in current_usages:
            source = u.get("source")
            prompt = u.get("prompt_tokens", 0)
            completion = u.get("completion_tokens", 0)
            
            agent_cfg = None
            if source == "supervisor":
                agent_cfg = self.config.supervisor
            elif source == "agent_a":
                agent_cfg = self.config.agent_a
            elif source == "agent_b":
                agent_cfg = self.config.agent_b
            
            if agent_cfg:
                cost = (prompt / 1_000_000 * agent_cfg.input_cost_per_m) + \
                       (completion / 1_000_000 * agent_cfg.output_cost_per_m)
                total_cost += cost
                if source == "supervisor":
                    sup_cost += cost
                elif source == "agent_a":
                    a_cost += cost
                elif source == "agent_b":
                    b_cost += cost
        
        final_decision.estimated_cost = total_cost
        final_decision.supervisor_cost = sup_cost
        final_decision.agent_a_cost = a_cost
        final_decision.agent_b_cost = b_cost

        total_prompt = sum(u.get("prompt_tokens", 0) for u in current_usages)
        total_completion = sum(u.get("completion_tokens", 0) for u in current_usages)
        total = sum(u.get("total_tokens", 0) for u in current_usages)

        def _sum_usage(source: str) -> TokenUsage:
            prompt = sum(u.get("prompt_tokens", 0) for u in current_usages if u.get("source") == source)
            completion = sum(u.get("completion_tokens", 0) for u in current_usages if u.get("source") == source)
            total_ = sum(u.get("total_tokens", 0) for u in current_usages if u.get("source") == source)
            return TokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total_
            )

        total_usage = TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total,
        )
        
        # Populate individual agent usages
        final_decision.total_token_usage = total_usage
        final_decision.supervisor_token_usage = _sum_usage("supervisor")
        final_decision.agent_a_token_usage = _sum_usage("agent_a")
        final_decision.agent_b_token_usage = _sum_usage("agent_b")

        return {"supervisor_decision": final_decision.model_dump()}

    def _build_graph(self):
        graph_builder = StateGraph(GraphState)
        graph_builder.add_node("run_agents", self._run_agents)
        graph_builder.add_node("evaluate", self._evaluate_votes)
        graph_builder.add_node("reconcile", self._reconcile)
        graph_builder.add_node("finalize", self._finalize)

        graph_builder.add_edge(START, "run_agents")
        graph_builder.add_edge("run_agents", "evaluate")
        graph_builder.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "reconcile": "reconcile",
                "finalize": "finalize",
            },
        )
        graph_builder.add_edge("reconcile", "run_agents")
        graph_builder.add_edge("finalize", END)

        return graph_builder.compile()

    def classify_document(
        self, document_id: str, document_name: str, document_text: str
    ) -> SupervisorDecision:
        start_time = time.time()
        initial_state: GraphState = {
            "document_id": document_id,
            "document_name": document_name,
            "document_text": document_text,
            "round_num": 1,
            "token_usages": [],
        }
        final_state = self.graph.invoke(initial_state)
        decision = SupervisorDecision.model_validate(final_state["supervisor_decision"])
        decision.total_duration_s = time.time() - start_time
        return decision
