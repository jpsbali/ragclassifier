import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class AgentModelConfig:
    name: str
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    timeout_s: float = 60.0
    input_cost_per_m: float = 0.0
    output_cost_per_m: float = 0.0


@dataclass
class ConsensusConfig:
    min_confidence: float = 0.90
    max_rounds: int = 3


@dataclass
class AppConfig:
    supervisor: AgentModelConfig
    agent_a: AgentModelConfig
    agent_b: AgentModelConfig
    evaluator: AgentModelConfig
    consensus: ConsensusConfig
    use_openrouter: bool = False
    max_file_size_mb: int = 10
    enable_risk_evaluation: bool = True  # If true, enables the cost-based risk evaluation step.


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).lower()
    return value in ("true", "1", "t", "y", "yes")


def _fallback_key(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def load_default_config() -> AppConfig:
    load_dotenv()
    use_openrouter = _env_bool("USE_OPENROUTER", False)
    api_key_fallbacks = ("OPENROUTER_API_KEY", "OPENAI_API_KEY") if use_openrouter else ("OPENAI_API_KEY",)

    if use_openrouter:
        supervisor_model_env = "OPENROUTER_SUPERVISOR_MODEL"
        agent_a_model_env = "OPENROUTER_AGENT_A_MODEL"
        agent_b_model_env = "OPENROUTER_AGENT_B_MODEL"
        evaluator_model_env = "OPENROUTER_EVALUATOR_MODEL"
        base_url_env = "OPENROUTER_BASE_URL"

        sup_in_cost = "OPENROUTER_SUPERVISOR_MODEL_COST_INPUT_TOKENS"
        sup_out_cost = "OPENROUTER_SUPERVISOR_MODEL_COST_OUTPUT_TOKENS"
        a_in_cost = "OPENROUTER_AGENT_A_MODEL_COST_INPUT_TOKENS"
        a_out_cost = "OPENROUTER_AGENT_A_MODEL_COST_OUTPUT_TOKENS"
        b_in_cost = "OPENROUTER_AGENT_B_MODEL_COST_INPUT_TOKENS"
        b_out_cost = "OPENROUTER_AGENT_B_MODEL_COST_OUTPUT_TOKENS"
        eval_in_cost = "OPENROUTER_EVALUATOR_MODEL_COST_INPUT_TOKENS"
        eval_out_cost = "OPENROUTER_EVALUATOR_MODEL_COST_OUTPUT_TOKENS"
    else:
        supervisor_model_env = "OPENAI_SUPERVISOR_MODEL"
        agent_a_model_env = "OPENAI_AGENT_A_MODEL"
        agent_b_model_env = "OPENAI_AGENT_B_MODEL"
        evaluator_model_env = "OPENAI_EVALUATOR_MODEL"
        base_url_env = "OPENAI_BASE_URL"

        sup_in_cost = "OPENAI_SUPERVISOR_MODEL_COST_INPUT_TOKENS"
        sup_out_cost = "OPENAI_SUPERVISOR_MODEL_COST_OUTPUT_TOKENS"
        a_in_cost = "OPENAI_AGENT_A_MODEL_COST_INPUT_TOKENS"
        a_out_cost = "OPENAI_AGENT_A_MODEL_COST_OUTPUT_TOKENS"
        b_in_cost = "OPENAI_AGENT_B_MODEL_COST_INPUT_TOKENS"
        b_out_cost = "OPENAI_AGENT_B_MODEL_COST_OUTPUT_TOKENS"
        eval_in_cost = "OPENAI_EVALUATOR_MODEL_COST_INPUT_TOKENS"
        eval_out_cost = "OPENAI_EVALUATOR_MODEL_COST_OUTPUT_TOKENS"

    supervisor = AgentModelConfig(
        name="supervisor",
        model=os.getenv(supervisor_model_env, ""),
        base_url=os.getenv(base_url_env, ""),
        api_key=_fallback_key("SUPERVISOR_API_KEY", *api_key_fallbacks),
        temperature=_env_float("SUPERVISOR_TEMPERATURE", 0.1),
        timeout_s=_env_float("SUPERVISOR_TIMEOUT_S", 60.0),
        input_cost_per_m=_env_float(sup_in_cost, 0.0),
        output_cost_per_m=_env_float(sup_out_cost, 0.0),
    )
    agent_a = AgentModelConfig(
        name="agent_a",
        model=os.getenv(agent_a_model_env, ""),
        base_url=os.getenv(base_url_env, ""),
        api_key=_fallback_key("AGENT_A_API_KEY", *api_key_fallbacks),
        temperature=_env_float("AGENT_A_TEMPERATURE", 0.0),
        timeout_s=_env_float("AGENT_A_TIMEOUT_S", 60.0),
        input_cost_per_m=_env_float(a_in_cost, 0.0),
        output_cost_per_m=_env_float(a_out_cost, 0.0),
    )
    agent_b = AgentModelConfig(
        name="agent_b",
        model=os.getenv(agent_b_model_env, ""),
        base_url=os.getenv(base_url_env, ""),
        api_key=_fallback_key("AGENT_B_API_KEY", *api_key_fallbacks),
        temperature=_env_float("AGENT_B_TEMPERATURE", 0.0),
        timeout_s=_env_float("AGENT_B_TIMEOUT_S", 60.0),
        input_cost_per_m=_env_float(b_in_cost, 0.0),
        output_cost_per_m=_env_float(b_out_cost, 0.0),
    )
    evaluator = AgentModelConfig(
        name="evaluator",
        model=os.getenv(evaluator_model_env, ""),
        base_url=os.getenv(base_url_env, ""),
        api_key=_fallback_key("EVALUATOR_API_KEY", *api_key_fallbacks),
        temperature=_env_float("EVALUATOR_TEMPERATURE", 0.0),
        timeout_s=_env_float("EVALUATOR_TIMEOUT_S", 60.0),
        input_cost_per_m=_env_float(eval_in_cost, 0.0),
        output_cost_per_m=_env_float(eval_out_cost, 0.0),
    )

    consensus = ConsensusConfig(
        min_confidence=_env_float("CONSENSUS_MIN_CONFIDENCE", 0.90),
        max_rounds=max(1, _env_int("CONSENSUS_MAX_ROUNDS", 3)),
    )

    return AppConfig(
        supervisor=supervisor,
        agent_a=agent_a,
        agent_b=agent_b,
        evaluator=evaluator,
        consensus=consensus,
        use_openrouter=use_openrouter,
        max_file_size_mb=_env_int("MAX_FILE_SIZE_MB", 10),
        enable_risk_evaluation=_env_bool("ENABLE_RISK_EVALUATION", True),
    )
