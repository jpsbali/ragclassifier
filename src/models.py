from enum import Enum

from pydantic import BaseModel, Field


class ClassificationLabel(str, Enum):
    RESTRICTED = "RESTRICTED"
    CONFIDENTIAL = "CONFIDENTIAL"
    PUBLIC = "PUBLIC"


class AgentVote(BaseModel):
    classification: ClassificationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    matched_rubric_points: list[str] = Field(default_factory=list)


class ReconciliationGuidance(BaseModel):
    instructions_for_retry: str = Field(min_length=1)


class SupervisorDecision(BaseModel):
    document_id: str
    document_name: str
    classification: ClassificationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    matched_rubric_points: list[str] = Field(default_factory=list)
    agent_a_vote: AgentVote
    agent_b_vote: AgentVote
    consensus_reached: bool
    consensus_score: float = Field(ge=0.0, le=1.0)
    rounds_used: int = Field(ge=1)

