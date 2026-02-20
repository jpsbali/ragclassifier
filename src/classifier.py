from typing import Any, TypedDict

from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import END, START, StateGraph

from src.agents import (
    build_chat_model,
    build_reconciliation_guidance,
    classify_with_agent,
    finalize_with_supervisor,
)
from src.config import AppConfig
from src.models import AgentVote, SupervisorDecision


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
    supervisor_decision: dict[str, Any]


class DocumentClassifier:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.supervisor_llm = build_chat_model(config.supervisor)
        self.agent_a_llm = build_chat_model(config.agent_a)
        self.agent_b_llm = build_chat_model(config.agent_b)
        self.graph = self._build_graph()

    def _run_agents(self, state: GraphState) -> GraphState:
        round_num = state.get("round_num", 1)
        retry_context = state.get("retry_context")
        document_name = state["document_name"]
        document_text = state["document_text"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            agent_a_future = executor.submit(
                classify_with_agent,
                llm=self.agent_a_llm,
                document_name=document_name,
                document_text=document_text,
                round_num=round_num,
                retry_context=retry_context,
            )
            agent_b_future = executor.submit(
                classify_with_agent,
                llm=self.agent_b_llm,
                document_name=document_name,
                document_text=document_text,
                round_num=round_num,
                retry_context=retry_context,
            )
            agent_a_vote = agent_a_future.result()
            agent_b_vote = agent_b_future.result()

        return {
            "agent_a_vote": agent_a_vote.model_dump(),
            "agent_b_vote": agent_b_vote.model_dump(),
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

        guidance = build_reconciliation_guidance(
            supervisor_llm=self.supervisor_llm,
            document_name=state["document_name"],
            document_text=state["document_text"],
            round_num=round_num,
            agent_a_vote=agent_a,
            agent_b_vote=agent_b,
        )

        return {
            "retry_context": guidance.instructions_for_retry,
            "round_num": round_num + 1,
        }

    def _finalize(self, state: GraphState) -> GraphState:
        agent_a = AgentVote.model_validate(state["agent_a_vote"])
        agent_b = AgentVote.model_validate(state["agent_b_vote"])

        final_decision: SupervisorDecision = finalize_with_supervisor(
            supervisor_llm=self.supervisor_llm,
            document_id=state["document_id"],
            document_name=state["document_name"],
            rounds_used=state.get("round_num", 1),
            consensus_reached=state.get("consensus_reached", False),
            consensus_score=state.get("consensus_score", 0.0),
            agent_a_vote=agent_a,
            agent_b_vote=agent_b,
        )

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
        initial_state: GraphState = {
            "document_id": document_id,
            "document_name": document_name,
            "document_text": document_text,
            "round_num": 1,
        }
        final_state = self.graph.invoke(initial_state)
        return SupervisorDecision.model_validate(final_state["supervisor_decision"])
