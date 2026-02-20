import json

import streamlit as st

from src.classifier import DocumentClassifier
from src.config import AgentModelConfig, AppConfig, ConsensusConfig, load_default_config
from src.document_loader import extract_text_from_upload


def _render_agent_settings(title: str, cfg: AgentModelConfig) -> AgentModelConfig:
    st.markdown(f"### {title}")
    model = st.text_input(f"{title} Model", value=cfg.model, key=f"{cfg.name}_model")
    base_url = st.text_input(
        f"{title} Base URL", value=cfg.base_url, key=f"{cfg.name}_base_url"
    )
    api_key = st.text_input(
        f"{title} API Key",
        value=cfg.api_key,
        type="password",
        key=f"{cfg.name}_api_key",
        help="Defaults from .env if provided, but can be overridden here.",
    )
    temperature = st.slider(
        f"{title} Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg.temperature),
        step=0.05,
        key=f"{cfg.name}_temperature",
    )
    timeout_s = st.number_input(
        f"{title} Timeout (seconds)",
        min_value=10.0,
        max_value=300.0,
        value=float(cfg.timeout_s),
        step=5.0,
        key=f"{cfg.name}_timeout",
    )

    return AgentModelConfig(
        name=cfg.name,
        model=model.strip(),
        base_url=base_url.strip(),
        api_key=api_key.strip(),
        temperature=float(temperature),
        timeout_s=float(timeout_s),
    )


def build_config_from_sidebar(default_cfg: AppConfig) -> AppConfig:
    with st.sidebar:
        st.header("LLM Configuration")
        supervisor = _render_agent_settings("Supervisor", default_cfg.supervisor)
        agent_a = _render_agent_settings("Agent A", default_cfg.agent_a)
        agent_b = _render_agent_settings("Agent B", default_cfg.agent_b)

        st.markdown("### Consensus")
        min_confidence = st.slider(
            "Min Confidence",
            min_value=0.50,
            max_value=1.00,
            value=float(default_cfg.consensus.min_confidence),
            step=0.01,
        )
        max_rounds = st.number_input(
            "Max Rounds", min_value=1, max_value=10, value=default_cfg.consensus.max_rounds
        )

    return AppConfig(
        supervisor=supervisor,
        agent_a=agent_a,
        agent_b=agent_b,
        consensus=ConsensusConfig(
            min_confidence=float(min_confidence),
            max_rounds=int(max_rounds),
        ),
    )


def _validate_keys(cfg: AppConfig) -> list[str]:
    missing = []
    if not cfg.supervisor.api_key:
        missing.append("Supervisor API key")
    if not cfg.agent_a.api_key:
        missing.append("Agent A API key")
    if not cfg.agent_b.api_key:
        missing.append("Agent B API key")
    return missing


def main() -> None:
    st.set_page_config(page_title="LangGraph Document Classifier", layout="wide")
    st.title("LangGraph Multi-Agent Document Classifier")
    st.caption(
        "Supervisor agent coordinates two sub-agents with configurable model endpoints and API keys."
    )

    default_cfg = load_default_config()
    runtime_cfg = build_config_from_sidebar(default_cfg)

    files = st.file_uploader(
        "Upload documents",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )

    run_clicked = st.button("Classify Documents", type="primary", disabled=not files)
    if not run_clicked:
        return

    missing = _validate_keys(runtime_cfg)
    if missing:
        st.error(f"Missing required keys: {', '.join(missing)}")
        return

    classifier = DocumentClassifier(runtime_cfg)
    results = []
    progress = st.progress(0.0)

    for i, uploaded in enumerate(files, start=1):
        content_bytes = uploaded.getvalue()
        text = extract_text_from_upload(uploaded.name, content_bytes)
        if not text:
            st.warning(f"Skipping empty document: {uploaded.name}")
            progress.progress(i / len(files))
            continue

        with st.spinner(f"Classifying: {uploaded.name}"):
            decision = classifier.classify_document(
                document_id=f"doc-{i}",
                document_name=uploaded.name,
                document_text=text,
            )
            results.append(decision)

        progress.progress(i / len(files))

    if not results:
        st.warning("No results to display.")
        return

    st.subheader("Classification Results")
    st.dataframe(
        [
            {
                "document_id": d.document_id,
                "document_name": d.document_name,
                "classification": d.classification.value,
                "confidence": round(d.confidence, 3),
                "consensus_reached": d.consensus_reached,
                "consensus_score": round(d.consensus_score, 3),
                "rounds_used": d.rounds_used,
            }
            for d in results
        ],
        use_container_width=True,
    )

    for decision in results:
        with st.expander(f"{decision.document_name} - {decision.classification.value}"):
            st.write("Reason:", decision.reason)
            st.write("Matched rubric points:", decision.matched_rubric_points)
            st.write("Agent A vote:", decision.agent_a_vote.model_dump())
            st.write("Agent B vote:", decision.agent_b_vote.model_dump())

    json_payload = [d.model_dump() for d in results]
    st.download_button(
        label="Download JSON",
        data=json.dumps(json_payload, indent=2),
        file_name="classification_results.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()

