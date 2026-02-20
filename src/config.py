import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass
class AgentModelConfig:
    name: str
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    timeout_s: float = 60.0


@dataclass
class ConsensusConfig:
    min_confidence: float = 0.90
    max_rounds: int = 3


@dataclass
class AppConfig:
    supervisor: AgentModelConfig
    agent_a: AgentModelConfig
    agent_b: AgentModelConfig
    consensus: ConsensusConfig


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _fallback_key(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def load_default_config() -> AppConfig:
    load_dotenv()

    supervisor = AgentModelConfig(
        name="supervisor",
        model=os.getenv("SUPERVISOR_MODEL", "gpt-4.1"),
        base_url=os.getenv("SUPERVISOR_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        api_key=_fallback_key("SUPERVISOR_API_KEY", "OPENAI_API_KEY"),
        temperature=_env_float("SUPERVISOR_TEMPERATURE", 0.1),
        timeout_s=_env_float("SUPERVISOR_TIMEOUT_S", 60.0),
    )
    agent_a = AgentModelConfig(
        name="agent_a",
        model=os.getenv("AGENT_A_MODEL", "gpt-4.1-mini"),
        base_url=os.getenv("AGENT_A_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        api_key=_fallback_key("AGENT_A_API_KEY", "OPENAI_API_KEY"),
        temperature=_env_float("AGENT_A_TEMPERATURE", 0.0),
        timeout_s=_env_float("AGENT_A_TIMEOUT_S", 60.0),
    )
    agent_b = AgentModelConfig(
        name="agent_b",
        model=os.getenv("AGENT_B_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("AGENT_B_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        api_key=_fallback_key("AGENT_B_API_KEY", "OPENAI_API_KEY"),
        temperature=_env_float("AGENT_B_TEMPERATURE", 0.0),
        timeout_s=_env_float("AGENT_B_TIMEOUT_S", 60.0),
    )

    consensus = ConsensusConfig(
        min_confidence=_env_float("CONSENSUS_MIN_CONFIDENCE", 0.90),
        max_rounds=max(1, _env_int("CONSENSUS_MAX_ROUNDS", 3)),
    )

    return AppConfig(
        supervisor=supervisor,
        agent_a=agent_a,
        agent_b=agent_b,
        consensus=consensus,
    )
