from typing import Any, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import AgentModelConfig
from src.models import (
    AgentVote,
    ReconciliationGuidance,
    SupervisorDecision,
    TokenUsage,
)
from src.rubric import AGENT_SYSTEM_PROMPT, RUBRIC_TEXT


def build_chat_model(cfg: AgentModelConfig, use_openrouter: bool) -> BaseChatModel:
    if use_openrouter:
        return ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key or None,
            base_url="https://openrouter.ai/api/v1",
            temperature=cfg.temperature,
            timeout=cfg.timeout_s,
        )
    else:
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
    """
    Generates the final decision using the supervisor LLM.

    The supervisor is prompted with the final votes and a set of rules to make a
    conclusive decision. These rules include enforcing `HUMAN_REVIEW` for
    persistent disagreements or two-tier gaps, with the latter being marked
    as high-priority.

    Args:
        supervisor_llm: The chat model for the supervisor.
        document_id: The ID of the document.
        document_name: The name of the document.
        rounds_used: The number of debate rounds that occurred.
        consensus_reached: Whether the agents reached consensus.
        consensus_score: The final consensus score.
        agent_a_vote: The final vote from Agent A.
        agent_b_vote: The final vote from Agent B.

    Returns:
        A tuple containing the final `SupervisorDecision` and the token usage.
    """
    structured = supervisor_llm.with_structured_output(SupervisorDecision, include_raw=True)
    prompt = f"""
You are the final decision-maker.
Generate final structured decision from the two agent votes.

Rubric:
{RUBRIC_TEXT}

Decision requirements:
- If `rounds_used` is 2 or more and the agent votes still disagree, the classification MUST be HUMAN_REVIEW.
- If `rounds_used` is 2 or more and there is a two-tier gap between votes (one RESTRICTED, one PUBLIC), the classification MUST be HUMAN_REVIEW and you MUST set `review_priority` to "HIGH".
- If the above rules for HUMAN_REVIEW do not apply:
    - If the votes agree and confidence is high, keep the agreed class.
    - If votes disagree in the first round, pick the class best supported by rubric evidence.
    - Prefer CONFIDENTIAL as default for sensitive internal data not clearly RESTRICTED or PUBLIC.
- Keep confidence calibrated (0 to 1). When classifying as HUMAN_REVIEW, set confidence to 0.0 and provide a reason explaining which rule was triggered.

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
