from enum import Enum

from pydantic import BaseModel, Field


class ClassificationLabel(str, Enum):
    RESTRICTED = "RESTRICTED"
    CONFIDENTIAL = "CONFIDENTIAL"
    PUBLIC = "PUBLIC"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentVote(BaseModel):
    classification: ClassificationLabel
    confidence: float
    reason: str = Field(min_length=1)
    matched_rubric_points: list[str] = Field(default_factory=list)


class ReconciliationGuidance(BaseModel):
    instructions_for_retry: str = Field(min_length=1)


class RoundHistory(BaseModel):
    round: int
    agent_a: AgentVote
    agent_b: AgentVote
    agent_a_duration_s: float = 0.0
    agent_b_duration_s: float = 0.0
    agent_a_token_usage: TokenUsage | None = None
    agent_b_token_usage: TokenUsage | None = None
    agent_a_cost: float = 0.0
    agent_b_cost: float = 0.0


class SupervisorDecision(BaseModel):
    document_id: str
    document_name: str
    classification: ClassificationLabel
    confidence: float
    reason: str = Field(min_length=1)
    matched_rubric_points: list[str] = Field(default_factory=list)
    agent_a_vote: AgentVote
    agent_b_vote: AgentVote
    consensus_reached: bool
    consensus_score: float
    rounds_used: int
    total_token_usage: TokenUsage | None = None
    supervisor_token_usage: TokenUsage | None = None
    agent_a_token_usage: TokenUsage | None = None
    agent_b_token_usage: TokenUsage | None = None
    history: list[RoundHistory] = Field(default_factory=list)
    total_duration_s: float = 0.0
    supervisor_duration_s: float = 0.0
    agent_a_duration_s: float = 0.0
    agent_b_duration_s: float = 0.0
    estimated_cost: float = 0.0
    supervisor_cost: float = 0.0
    agent_a_cost: float = 0.0
    agent_b_cost: float = 0.0
