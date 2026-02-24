from typing import Any, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import AgentModelConfig
from src.models import (
    AgentVote,
    ReconciliationGuidance,
    SupervisorDecision,
    TokenUsage,
)
from src.rubric import AGENT_SYSTEM_PROMPT, RUBRIC_TEXT


def build_chat_model(cfg: AgentModelConfig) -> ChatOpenAI:
    # If api_key or base_url are empty/None, pass None to ChatOpenAI.
    # This makes it fall back to environment variables like OPENAI_API_KEY
    # and the default OpenAI base URL.
    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key or None,
        base_url=cfg.base_url or None,
        temperature=cfg.temperature,
        timeout=cfg.timeout_s,
    )


def _extract_token_usage(response: dict[str, Any]) -> TokenUsage:
    """Extracts token usage from the raw response of a LangChain LLM call."""
    raw_message = response.get("raw")
    if not raw_message or not hasattr(raw_message, "response_metadata"):
        return TokenUsage()

    token_usage_dict = raw_message.response_metadata.get("token_usage", {})
    return TokenUsage(
        prompt_tokens=token_usage_dict.get("prompt_tokens", 0),
        completion_tokens=token_usage_dict.get("completion_tokens", 0),
        total_tokens=token_usage_dict.get("total_tokens", 0),
    )


def classify_with_agent(
    llm: ChatOpenAI,
    document_name: str,
    document_text: str,
    round_num: int,
    retry_context: str | None = None,
) -> Tuple[AgentVote, TokenUsage]:
    """Classifies a document and returns the vote and token usage."""
    structured = llm.with_structured_output(AgentVote, include_raw=True)

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

    response = structured.invoke(
        [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    vote = response["parsed"]
    token_usage = _extract_token_usage(response)
    return vote, token_usage


def build_reconciliation_guidance(
    supervisor_llm: ChatOpenAI,
    document_name: str,
    document_text: str,
    round_num: int,
    agent_a_vote: AgentVote,
    agent_b_vote: AgentVote,
) -> Tuple[ReconciliationGuidance, TokenUsage]:
    """Builds reconciliation guidance and returns it with token usage."""
    structured = supervisor_llm.with_structured_output(
        ReconciliationGuidance, include_raw=True
    )
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
    response = structured.invoke(
        [
            SystemMessage(
                content="Write concise, evidence-focused retry instructions for both agents."
            ),
            HumanMessage(content=prompt),
        ]
    )
    guidance = response["parsed"]
    token_usage = _extract_token_usage(response)
    return guidance, token_usage


def finalize_with_supervisor(
    supervisor_llm: ChatOpenAI,
    document_id: str,
    document_name: str,
    rounds_used: int,
    consensus_reached: bool,
    consensus_score: float,
    agent_a_vote: AgentVote,
    agent_b_vote: AgentVote,
) -> Tuple[SupervisorDecision, TokenUsage]:
    """Generates the final decision and returns it with token usage."""
    structured = supervisor_llm.with_structured_output(SupervisorDecision, include_raw=True)
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
    response = structured.invoke(
        [
            SystemMessage(
                content=(
                    "Return a complete SupervisorDecision object with clear rationale tied to rubric."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    decision = response["parsed"]
    token_usage = _extract_token_usage(response)
    return decision, token_usage
