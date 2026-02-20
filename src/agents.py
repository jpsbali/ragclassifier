from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import AgentModelConfig
from src.models import AgentVote, ReconciliationGuidance, SupervisorDecision
from src.rubric import AGENT_SYSTEM_PROMPT, RUBRIC_TEXT


def build_chat_model(cfg: AgentModelConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        timeout=cfg.timeout_s,
    )


def classify_with_agent(
    llm: ChatOpenAI,
    document_name: str,
    document_text: str,
    round_num: int,
    retry_context: str | None = None,
) -> AgentVote:
    structured = llm.with_structured_output(AgentVote)

    user_prompt = f"""
Document name: {document_name}
Round: {round_num}

Document content:
\"\"\"
{document_text}
\"\"\"
"""
    if retry_context:
        user_prompt += f"""

Retry context:
{retry_context}
"""

    return structured.invoke(
        [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )


def build_reconciliation_guidance(
    supervisor_llm: ChatOpenAI,
    document_name: str,
    document_text: str,
    round_num: int,
    agent_a_vote: AgentVote,
    agent_b_vote: AgentVote,
) -> ReconciliationGuidance:
    structured = supervisor_llm.with_structured_output(ReconciliationGuidance)
    prompt = f"""
You are the supervisor coordinating two classification agents.
Draft neutral retry guidance to help both agents converge to a high-confidence decision
without forcing a label.

Rubric:
{RUBRIC_TEXT}

Document name: {document_name}
Current round: {round_num}
Document content:
\"\"\"
{document_text}
\"\"\"

Agent A vote:
{agent_a_vote.model_dump_json(indent=2)}

Agent B vote:
{agent_b_vote.model_dump_json(indent=2)}
"""

    return structured.invoke(
        [
            SystemMessage(
                content="Write concise, evidence-focused retry instructions for both agents."
            ),
            HumanMessage(content=prompt),
        ]
    )


def finalize_with_supervisor(
    supervisor_llm: ChatOpenAI,
    document_id: str,
    document_name: str,
    rounds_used: int,
    consensus_reached: bool,
    consensus_score: float,
    agent_a_vote: AgentVote,
    agent_b_vote: AgentVote,
) -> SupervisorDecision:
    structured = supervisor_llm.with_structured_output(SupervisorDecision)
    prompt = f"""
You are the final decision-maker.
Generate final structured decision from the two agent votes.

Rubric:
{RUBRIC_TEXT}

Decision requirements:
- If the votes agree and confidence is high, keep the agreed class.
- If votes disagree, pick the class best supported by rubric evidence.
- Prefer CONFIDENTIAL as default for sensitive internal data not clearly RESTRICTED or PUBLIC.
- Keep confidence calibrated (0 to 1).

Decision context:
- document_id: {document_id}
- document_name: {document_name}
- rounds_used: {rounds_used}
- consensus_reached: {consensus_reached}
- consensus_score: {consensus_score}

Agent A vote:
{agent_a_vote.model_dump_json(indent=2)}

Agent B vote:
{agent_b_vote.model_dump_json(indent=2)}
"""
    return structured.invoke(
        [
            SystemMessage(
                content=(
                    "Return a complete SupervisorDecision object with clear rationale tied to rubric."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

