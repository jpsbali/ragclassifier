import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
import io
import csv
import uuid

import altair as alt
import pandas as pd
import streamlit as st

from src.classifier import DocumentClassifier
from zipfile import ZipFile
from src.config import AgentModelConfig, AppConfig, ConsensusConfig, load_default_config
from src.document_loader import extract_text_from_upload
from src.risk_evaluator import RiskEvaluator, TCHClassification
from src.report_generator import generate_pdf_report, HAS_REPORTLAB
from src.excel_logger import log_run_to_excel


def _initialize_configs_in_state():
    """Loads default config and populates session state. Does not rerun."""
    default_cfg = load_default_config()
    for agent_cfg in [default_cfg.supervisor, default_cfg.agent_a, default_cfg.agent_b, default_cfg.evaluator]:
        st.session_state[f"{agent_cfg.name}_model"] = agent_cfg.model
        st.session_state[f"{agent_cfg.name}_base_url"] = agent_cfg.base_url
        st.session_state[f"{agent_cfg.name}_api_key"] = agent_cfg.api_key
        st.session_state[f"{agent_cfg.name}_temperature"] = agent_cfg.temperature
        st.session_state[f"{agent_cfg.name}_timeout_s"] = agent_cfg.timeout_s
        st.session_state[f"{agent_cfg.name}_input_cost_per_m"] = agent_cfg.input_cost_per_m
        st.session_state[f"{agent_cfg.name}_output_cost_per_m"] = agent_cfg.output_cost_per_m

    st.session_state["enable_risk_evaluation"] = default_cfg.enable_risk_evaluation
    st.session_state["max_file_size_mb"] = default_cfg.max_file_size_mb
    st.session_state["consensus_min_confidence"] = default_cfg.consensus.min_confidence
    st.session_state["consensus_max_rounds"] = default_cfg.consensus.max_rounds


def _reset_all_configs_callback():
    """Callback to reset configs and rerun."""
    _initialize_configs_in_state()

def _render_agent_settings(
    title: str, cfg: AgentModelConfig, use_openrouter: bool
) -> AgentModelConfig:
    model = st.text_input("Model", key=f"{cfg.name}_model")

    base_url = st.text_input(
        "Base URL", key=f"{cfg.name}_base_url", help="Optionally override the default Base URL"
    )

    api_key_help = "Optionally override the default API Key"
    api_key = st.text_input(
        "API Key",
        type="password",
        key=f"{cfg.name}_api_key",
        help=api_key_help,
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key=f"{cfg.name}_temperature",
    )
    timeout_s = st.number_input(
        "Timeout (seconds)",
        min_value=10.0,
        max_value=300.0,
        step=5.0,
        key=f"{cfg.name}_timeout_s",
    )
    input_cost = st.number_input(
        "Input Cost ($/1M tokens)",
        min_value=0.0,
        step=0.01,
        format="%.2f",
        key=f"{cfg.name}_input_cost_per_m",
    )
    output_cost = st.number_input(
        "Output Cost ($/1M tokens)",
        min_value=0.0,
        step=0.01,
        format="%.2f",
        key=f"{cfg.name}_output_cost_per_m",
    )

    return AgentModelConfig(
        name=cfg.name,
        model=model.strip(),
        base_url=base_url.strip(),
        api_key=api_key.strip(),
        temperature=float(temperature),
        timeout_s=float(timeout_s),
        input_cost_per_m=float(input_cost),
        output_cost_per_m=float(output_cost),
    )
def build_config_from_sidebar(default_cfg: AppConfig) -> AppConfig:
    with st.sidebar:
        with st.expander("General Settings", expanded=False):
            enable_risk_evaluation = st.toggle(
                "Enable Risk Evaluation",
                help="If enabled, calculates asymmetric cost and flags high-risk items for review.",
                key="enable_risk_evaluation",
            )
            max_file_size_mb = st.slider(
                "Max File Size (MB)",
                min_value=1,
                max_value=50,
                help="Set the maximum size for individual files to be processed.",
                key="max_file_size_mb",
            )
            st.markdown("### Consensus")
            min_confidence = st.slider(
                "Min Confidence",
                min_value=0.50,
                max_value=1.00,
                step=0.01,
                key="consensus_min_confidence",
            )
            max_rounds = st.number_input(
                "Max Rounds",
                min_value=1,
                max_value=10,
                key="consensus_max_rounds",
            )

        with st.expander("LLM Configuration", expanded=False):
            use_openrouter = default_cfg.use_openrouter

            with st.expander("Supervisor"):
                supervisor = _render_agent_settings("Supervisor", default_cfg.supervisor, use_openrouter)
            with st.expander("Agent A"):
                agent_a = _render_agent_settings("Agent A", default_cfg.agent_a, use_openrouter)
            with st.expander("Agent B"):
                agent_b = _render_agent_settings("Agent B", default_cfg.agent_b, use_openrouter)
            with st.expander("Evaluator"):
                evaluator = _render_agent_settings("Evaluator", default_cfg.evaluator, use_openrouter)

            st.markdown("---")
            st.button(
                "Reset All to Defaults",
                on_click=_reset_all_configs_callback,
                use_container_width=True,
                type="secondary",
                help="Resets all LLM and consensus settings to the values from the .env file.",
            )

    return AppConfig(
        supervisor=supervisor,
        agent_a=agent_a,
        agent_b=agent_b,
        evaluator=evaluator,
        consensus=ConsensusConfig(
            min_confidence=float(min_confidence),
            max_rounds=int(max_rounds),
        ),
        use_openrouter=default_cfg.use_openrouter,
        max_file_size_mb=int(max_file_size_mb),
        enable_risk_evaluation=enable_risk_evaluation,
    )



def _render_history_sidebar(min_confidence: float) -> None:
    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []

    if "history_page" not in st.session_state:
        st.session_state["history_page"] = 0

    with st.sidebar:
        with st.expander("Historical Runs", expanded=False):
            # Visualization of confidence scores
            if st.session_state["run_history"]:
                chart_data = []
                for item in st.session_state["run_history"]:
                    # Calculate average confidence for the run
                    if item["results"]:
                        avg_conf = sum(r.confidence for r in item["results"]) / len(
                            item["results"]
                        )
                        chart_data.append(
                            {
                                "timestamp": item["timestamp"],
                                "avg_confidence": avg_conf,
                                "run_id": item["run_id"],
                            }
                        )
                if chart_data:
                    st.caption(f"Average Confidence Trend (Threshold: {min_confidence})")

                    df = pd.DataFrame(chart_data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    base = alt.Chart(df).encode(
                        x=alt.X(
                            "timestamp",
                            axis=alt.Axis(labels=False, title=None),
                            scale=alt.Scale(nice=True),
                        )
                    )

                    # Define selection explicitly to avoid Streamlit injection errors with configured charts
                    run_selection = alt.selection_point(name="run_selection", on="click", fields=["run_id"])

                    # Determine current run ID for conditional coloring
                    current_run_id = st.session_state.get("run_id")

                    chart = base.mark_circle(size=60).encode(
                        color=alt.condition(
                            alt.datum.run_id == current_run_id,
                            alt.value("#FF8C00"),  # Highlight color (Dark Orange)
                            alt.value("#00B4D8"),  # Default color
                        ),
                        y=alt.Y("avg_confidence", scale=alt.Scale(domain=[0, 1]), title="Conf"),
                        tooltip=[
                            alt.Tooltip("timestamp", title="Run Time", format="%Y-%m-%d %H:%M:%S"),
                            alt.Tooltip("avg_confidence", title="Avg. Confidence", format=".3f"),
                        ],
                    ).add_params(run_selection)

                    chart = (
                        chart
                        .properties(height=150, background="transparent")
                        .configure_axis(gridColor="#444444")
                        .configure_axisY(titleColor="#FAFAFA", labelColor="#FAFAFA")
                        .configure_axisX(titleColor="#FAFAFA", labelColor="#FAFAFA")
                        .configure_title(color="#FAFAFA")
                        .configure_legend(titleColor="#FAFAFA", labelColor="#FAFAFA")
                    )
                    event = st.altair_chart(
                        chart,
                        use_container_width=True,
                        theme=None,
                        on_select="rerun",
                    )

                    if event.selection and "run_selection" in event.selection:
                        selection_data = event.selection["run_selection"]
                        if selection_data:
                            # selection_data is a list of dicts, e.g. [{'run_id': '...'}]
                            run_id = selection_data[0]["run_id"]
                            if st.session_state.get("run_id") != run_id:
                                for item in st.session_state["run_history"]:
                                    if item["run_id"] == run_id:
                                        st.session_state["results"] = item["results"]
                                        st.session_state["run_id"] = item["run_id"]
                                        st.session_state["results_dir"] = item["results_dir"]
                                        st.rerun()

            search_date = st.date_input(
                "Filter by date", value=None, max_value=datetime.now()
            )

            # Sort by timestamp descending to ensure newest are always on top
            sorted_history = sorted(
                st.session_state["run_history"], key=lambda x: x["timestamp"], reverse=True
            )

            # Apply filter
            filtered_history = [
                item
                for item in sorted_history
                if not search_date or str(search_date) in item["timestamp"]
            ]

            # Pagination logic
            items_per_page = 5
            total_pages = max(1, (len(filtered_history) + items_per_page - 1) // items_per_page)

            # Ensure current page is valid
            if st.session_state["history_page"] >= total_pages:
                st.session_state["history_page"] = total_pages - 1
            if st.session_state["history_page"] < 0:
                st.session_state["history_page"] = 0

            current_page = st.session_state["history_page"]
            start_idx = current_page * items_per_page
            end_idx = start_idx + items_per_page

            # Display page items
            for item in filtered_history[start_idx:end_idx]:
                label = f"`{item['run_id']}` ({len(item['results'])} docs)"
                if st.button(label, key=f"hist_{item['run_id']}", use_container_width=True):
                    st.session_state["results"] = item["results"]
                    st.session_state["run_id"] = item["run_id"]
                    st.session_state["results_dir"] = item["results_dir"]
                    st.rerun()

                # Export to ZIP button
                results_dir = Path(item["results_dir"])
                run_id = item["run_id"]
                csv_path = results_dir / f"classify_{run_id}.csv"
                json_path = results_dir / f"classify_{run_id}.json"

                if csv_path.exists() and json_path.exists():
                    zip_buffer = BytesIO()
                    with ZipFile(zip_buffer, "w") as zip_file:
                        zip_file.write(csv_path, arcname=csv_path.name)
                        zip_file.write(json_path, arcname=json_path.name)
                    
                    st.download_button(
                        label="📥",
                        data=zip_buffer.getvalue(),
                        file_name=f"run_{run_id}.zip",
                        mime="application/zip",
                        key=f"zip_{run_id}",
                        help="Download run results as ZIP",
                    )

            # Pagination controls
            if total_pages > 1:
                col_prev, col_info, col_next = st.columns([0.3, 0.4, 0.3])
                with col_prev:
                    if st.button("Prev", key="hist_prev", disabled=current_page == 0):
                        st.session_state["history_page"] -= 1
                        st.rerun()
                with col_info:
                    st.caption(f"Page {current_page + 1}/{total_pages}")
                with col_next:
                    if st.button(
                        "Next", key="hist_next", disabled=current_page == total_pages - 1
                    ):
                        st.session_state["history_page"] += 1
                        st.rerun()

            # Then show the clear button or empty message
            if st.session_state["run_history"]:
                def _clear_history():
                    st.session_state["run_history"] = []
                    if "results" in st.session_state:
                        del st.session_state["results"]

                st.button(
                    "Clear History",
                    on_click=_clear_history,
                    use_container_width=True,
                    type="secondary",
                    help="Deletes all past run history from this session.",
                )
            else:
                st.caption("No past runs in this session.")


def _render_results(runtime_cfg: AppConfig) -> None:
    """Renders the classification results, including the sidebar queue and main table."""
    if "results" not in st.session_state:
        return

    results = st.session_state["results"]
    run_id = st.session_state.get("run_id")
    results_dir = st.session_state.get("results_dir")

    def set_expanded_doc(doc_id: str):
        """Callback to set the currently expanded document."""
        # If the same button is clicked again, collapse it.
        if st.session_state.expanded_doc_id == doc_id:
            st.session_state.expanded_doc_id = None
        else:
            st.session_state.expanded_doc_id = doc_id

    col_header, col_clear = st.columns([0.85, 0.15], vertical_alignment="bottom")
    with col_header:
        st.subheader("Classification Results")
    with col_clear:
        if st.button("Clear Results", use_container_width=True):
            del st.session_state["results"]
            st.rerun()

    if run_id and results_dir:
        st.success(f"Results exported to `{results_dir}` with run ID `{run_id}`.")

        if HAS_REPORTLAB:
            pdf_data = generate_pdf_report(run_id, results, risk_evaluator)
            if pdf_data:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name=f"report_{run_id}.pdf",
                    mime="application/pdf",
                    key="download_pdf_report"
                )

    risk_evaluator = RiskEvaluator() if runtime_cfg.enable_risk_evaluation else None

    # Pre-calculate risk for all results to avoid redundant computations
    evaluated_results = []
    for d in results:
        risk_eval = None
        if risk_evaluator:
            risk_eval = risk_evaluator.calculate_risk(
                TCHClassification(d.classification.value), d.confidence
            )
        evaluated_results.append({"decision": d, "risk": risk_eval})  # risk can be None

    # Add Human Review Queue to sidebar if there are high-risk items
    if risk_evaluator:
        high_risk_docs = [
            item for item in evaluated_results if item["risk"] and item["risk"].is_high_risk
        ]
        if high_risk_docs:
            with st.sidebar:
                with st.expander("⚠️ Human Review Queue", expanded=True):
                    st.caption(f"{len(high_risk_docs)} document(s) require review.")
                    for item in high_risk_docs:
                        doc_id = item["decision"].document_id
                        doc_name = item["decision"].document_name
                        st.button(
                            doc_name,
                            key=f"hr_queue_{doc_id}",
                            on_click=set_expanded_doc,
                            args=(doc_id,),
                            use_container_width=True,
                            type="secondary",
                            help="Click to view this document's details",
                        )

                    st.markdown("---")  # Visual separator

                    # Prepare data for CSV export
                    csv_headers = [
                        "document_id",
                        "document_name",
                        "classification",
                        "confidence",
                        "consensus_reached",
                        "consensus_score",
                        "rounds_used",
                        "estimated_cost",
                        "risk_expected_cost",
                        "risk_reasoning",
                    ]
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=csv_headers)
                    writer.writeheader()
                    for item in high_risk_docs:
                        d = item["decision"]
                        risk_eval = item["risk"]
                        writer.writerow(
                            {
                                "document_id": d.document_id,
                                "document_name": d.document_name,
                                "classification": TCHClassification.HUMAN_REVIEW.value,
                                "confidence": d.confidence,
                                "consensus_reached": d.consensus_reached,
                                "consensus_score": d.consensus_score,
                                "rounds_used": d.rounds_used,
                                "estimated_cost": d.estimated_cost,
                                "risk_expected_cost": risk_eval.expected_cost,
                                "risk_reasoning": risk_eval.reasoning,
                            }
                        )

                    csv_data = output.getvalue()

                    st.download_button(
                        label="Export Queue for Review",
                        data=csv_data,
                        file_name=f"human_review_queue_{run_id}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_hr_queue",
                    )

    # Prepare data for display with Risk Evaluation
    display_data = []
    for item in evaluated_results:
        d = item["decision"]
        risk_eval = item["risk"]
        final_classification = d.classification.value
        risk_cost_str = "N/A"
        risk_flag_str = "⚪"
        if risk_eval:
            # Determine the final classification and UI icon based on a hierarchy of rules:
            # 1. Was it flagged for review by the classifier agent directly (e.g., disagreement)?
            # 2. Was it flagged for review by the risk evaluator (cost > threshold)?
            # 3. If flagged, is it high-priority?
            is_hr_by_classifier = d.classification.value == TCHClassification.HUMAN_REVIEW.value
            is_hr_by_evaluator = risk_eval.is_high_risk
            is_high_priority = d.review_priority == "HIGH"

            if is_hr_by_classifier or is_hr_by_evaluator:
                final_classification = TCHClassification.HUMAN_REVIEW.value
                # Use a siren for high-priority, otherwise a warning sign.
                risk_flag_str = "🚨" if is_high_priority else "⚠️"
            else:
                # If no review is needed, the classification is the original one.
                final_classification = d.classification.value
                risk_flag_str = "✅"

            risk_cost_str = f"${risk_eval.expected_cost:.2f}"

        display_data.append(
            {
                "document_id": d.document_id,
                "document_name": d.document_name,
                "classification": final_classification,
                "confidence": round(d.confidence, 3),
                "consensus_reached": d.consensus_reached,
                "rounds_used": d.rounds_used,
                "estimated_cost": f"${d.estimated_cost:.4f}",
                "risk_cost": risk_cost_str,
                "risk_flag": risk_flag_str,
            }
        )

    if display_data:
        st.markdown("### Classification Distribution")
        df_dist = pd.DataFrame(display_data)
        dist_counts = df_dist["classification"].value_counts().reset_index()
        dist_counts.columns = ["Classification", "Count"]

        chart = (
            alt.Chart(dist_counts)
            .mark_bar()
            .encode(
                x=alt.X("Count", title="Count", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Classification", sort="-x", title=None),
                color=alt.Color("Classification", legend=None),
                tooltip=["Classification", "Count"],
            )
            .properties(height=max(100, len(dist_counts) * 40), background="transparent")
            .configure_axis(gridColor="#444444", labelColor="#FAFAFA", titleColor="#FAFAFA")
        )
        st.altair_chart(chart, use_container_width=True)

    st.dataframe(display_data, use_container_width=True)

    for item in evaluated_results:
        decision = item["decision"]
        risk_eval = item["risk"]

        # Determine final classification and if it's a human review case
        is_hr_by_classifier = decision.classification.value == TCHClassification.HUMAN_REVIEW.value
        is_hr_by_evaluator = risk_eval and risk_eval.is_high_risk
        is_human_review = is_hr_by_classifier or is_hr_by_evaluator
        final_classification = TCHClassification.HUMAN_REVIEW.value if is_human_review else decision.classification.value

        is_expanded = decision.document_id == st.session_state.expanded_doc_id
        with st.expander(
            f"{decision.document_name} - {final_classification}", expanded=is_expanded
        ):
            if risk_evaluator and risk_eval:
                col1, col2, col3, col4, col5 = st.columns(5)
            else:
                col1, col2, col3, col4 = st.columns(4)

            col1.metric("Confidence", f"{decision.confidence:.2f}")
            col2.metric("Rounds Used", decision.rounds_used)
            col3.metric("Total Time", f"{decision.total_duration_s:.2f}s")
            col4.metric("Est. Cost", f"${decision.estimated_cost:.4f}")
            if risk_evaluator and risk_eval and not is_hr_by_classifier:
                col5.metric(
                    "Risk Cost",
                    f"${risk_eval.expected_cost:.2f}",
                    delta="-High Risk" if risk_eval.is_high_risk else "OK",
                    delta_color="inverse",
                )
            st.write("Reason:", decision.reason)

            # Display the appropriate warning/error box in the expander.
            if is_human_review:
                # Consolidate the reason for the review.
                reason_text = ""
                if is_hr_by_classifier:
                    # Reason comes from the supervisor agent's decision.
                    reason_text = decision.reason
                elif is_hr_by_evaluator:
                    # Reason comes from the risk evaluator's calculation.
                    reason_text = risk_eval.reasoning

                # Set the title based on priority.
                title = "FLAGGED FOR HUMAN REVIEW"
                if decision.review_priority == "HIGH":
                    title = "FLAGGED FOR HIGH-PRIORITY HUMAN REVIEW"

                st.error(f"🚨 **{title}**\n\n{reason_text}", icon="🚨")

            elif risk_eval and risk_eval.adjusted_prediction.value != decision.classification.value:
                # If not a full human review, show a warning if the risk evaluator
                # adjusted the classification due to low confidence.
                st.warning(f"**Risk Adjustment:** {risk_eval.reasoning}")

            if len(decision.history) > 1:
                st.markdown("---")
                st.subheader("Debate History")
                for h in decision.history:
                    st.markdown(f"**Round {h.round}**")
                    c1, c2 = st.columns(2)
                    c1.info(
                        f"Agent A: {h.agent_a.classification.value} ({h.agent_a.confidence:.2f}) - {h.agent_a_duration_s:.2f}s"
                    )
                    c1.caption(h.agent_a.reason)
                    c2.info(
                        f"Agent B: {h.agent_b.classification.value} ({h.agent_b.confidence:.2f}) - {h.agent_b_duration_s:.2f}s"
                    )
                    c2.caption(h.agent_b.reason)

            st.write("Matched rubric points:", decision.matched_rubric_points)

            st.markdown("---")
            st.subheader("Execution & Cost Breakdown")

            def _get_counts(usage):
                if not usage:
                    return 0, 0, 0
                return usage.prompt_tokens, usage.completion_tokens, usage.total_tokens

            s_p, s_c, s_t = _get_counts(decision.supervisor_token_usage)
            a_p, a_c, a_t = _get_counts(decision.agent_a_token_usage)
            b_p, b_c, b_t = _get_counts(decision.agent_b_token_usage)

            # Calculate totals
            t_p = s_p + a_p + b_p
            t_c = s_c + a_c + b_c
            t_t = s_t + a_t + b_t

            s_cost = decision.supervisor_cost
            a_cost = decision.agent_a_cost
            b_cost = decision.agent_b_cost
            t_cost = s_cost + a_cost + b_cost

            s_time = decision.supervisor_duration_s
            a_time = decision.agent_a_duration_s
            b_time = decision.agent_b_duration_s
            t_time = s_time + a_time + b_time

            breakdown_rows = []

            # Supervisor
            breakdown_rows.append(
                {
                    "Component": "Supervisor (Total)",
                    "Execution Time": f"{s_time:.2f}s",
                    "Cost": f"${s_cost:.4f}",
                    "Prompt Tokens": str(s_p),
                    "Completion Tokens": str(s_c),
                    "Total Tokens": str(s_t),
                }
            )

            # Agent A Total
            breakdown_rows.append(
                {
                    "Component": "Agent A (Total)",
                    "Execution Time": f"{a_time:.2f}s",
                    "Cost": f"${a_cost:.4f}",
                    "Prompt Tokens": str(a_p),
                    "Completion Tokens": str(a_c),
                    "Total Tokens": str(a_t),
                }
            )

            # Agent B Total
            breakdown_rows.append(
                {
                    "Component": "Agent B (Total)",
                    "Execution Time": f"{b_time:.2f}s",
                    "Cost": f"${b_cost:.4f}",
                    "Prompt Tokens": str(b_p),
                    "Completion Tokens": str(b_c),
                    "Total Tokens": str(b_t),
                }
            )

            # Per Run Data
            for h in decision.history:
                for agent_name, usage, duration, cost in [
                    (
                        f"Agent A Run {h.round}",
                        h.agent_a_token_usage,
                        h.agent_a_duration_s,
                        h.agent_a_cost,
                    ),
                    (
                        f"Agent B Run {h.round}",
                        h.agent_b_token_usage,
                        h.agent_b_duration_s,
                        h.agent_b_cost,
                    ),
                ]:
                    p, c, t = _get_counts(usage)
                    breakdown_rows.append(
                        {
                            "Component": agent_name,
                            "Execution Time": f"{duration:.2f}s",
                            "Cost": f"${cost:.4f}",
                            "Prompt Tokens": str(p),
                            "Completion Tokens": str(c),
                            "Total Tokens": str(t),
                        }
                    )

            # Total
            breakdown_rows.append(
                {
                    "Component": "Total",
                    "Execution Time": f"{t_time:.2f}s",
                    "Cost": f"${t_cost:.4f}",
                    "Prompt Tokens": str(t_p),
                    "Completion Tokens": str(t_c),
                    "Total Tokens": str(t_t),
                }
            )

            breakdown_df = pd.DataFrame(breakdown_rows)

            def highlight_total(row):
                return (
                    ["background-color: #262730"] * len(row)
                    if row["Component"] == "Total"
                    else [""] * len(row)
                )

            st.dataframe(
                breakdown_df.style.apply(highlight_total, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            for h in decision.history:
                st.write(f"**Agent A Run {h.round}:**", h.agent_a.model_dump())
                st.write(f"**Agent B Run {h.round}:**", h.agent_b.model_dump())


def main() -> None:
    st.set_page_config(page_title="LangChain Multi-Agent Document Classifier", layout="wide")

    # Initialize session state for UI interactions
    if "expanded_doc_id" not in st.session_state:
        st.session_state.expanded_doc_id = None

    # Initialize config state on first run, before any widgets are created
    if "supervisor_model" not in st.session_state:
        _initialize_configs_in_state()

    # Add custom logo to the sidebar
    st.logo("https://www.theclearinghouse.org/assets/tch/images/logo/tch-logo-white.png")

    st.title("LangChain Multi-Agent Document Classifier")

    # Force dark mode theme
    css = """ 
    <style>
    body {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    [data-testid="stSidebar"] {
        background-color: #262730;
        color: #FAFAFA;
    }
    /* Force text colors for inputs and text to match theme */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        color: #FAFAFA !important;
        -webkit-text-fill-color: #FAFAFA !important;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .stDataFrame {
        color: #FAFAFA !important;
    }
    /* Reduce font size for main body text */
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li {
        font-size: 0.9rem;
    }
    /* Reduce font size for expander headers in sidebar */
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary p {
        font-size: 14px !important;
    }
    /* Reduce padding in sidebar */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0;
    }
    /* Add space around history chart */
    [data-testid="stSidebar"] [data-testid="stAltairChart"] {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    /* Ensure main content isn't hidden behind fixed footer */
    section.main .block-container {
        padding-bottom: 4rem;
    }
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        z-index: 100;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    st.caption(
        "Supervisor agent coordinates two sub-agents with configurable model endpoints and API keys."
    )

    # Add footer
    st.markdown(
        '<div class="footer">© The Clearing House Payments Company L.L.C. | <a href="https://www.theclearinghouse.org/privacy-policy" target="_blank" style="color: #FAFAFA; text-decoration: underline;">Privacy Policy</a></div>',
        unsafe_allow_html=True,
    )

    default_cfg = load_default_config()
    runtime_cfg = build_config_from_sidebar(default_cfg)

    _render_history_sidebar(runtime_cfg.consensus.min_confidence)

    input_mode = st.radio("Input Source", ["Upload Files", "Local Directory"], horizontal=True)
    files_to_process = []

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    if input_mode == "Upload Files":
        uploaded_files = st.file_uploader(
            f"Upload documents (max {runtime_cfg.max_file_size_mb}MB each)",
            type=["txt", "md", "pdf", "docx", "pptx", "xlsx"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}",
            max_upload_size=runtime_cfg.max_file_size_mb,
        )
        if uploaded_files:
            files_to_process = uploaded_files
    else:
        if "dir_path" not in st.session_state:
            st.session_state["dir_path"] = ""

        def _browse_callback():
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.wm_attributes("-topmost", 1)
                folder_path = filedialog.askdirectory(master=root)
                root.destroy()
                if folder_path:
                    st.session_state["dir_path"] = folder_path
            except Exception as e:
                st.error(f"Could not open folder dialog: {e}")

        def _clear_callback():
            st.session_state["dir_path"] = ""

        col1, col2, col3 = st.columns([0.75, 0.15, 0.10], vertical_alignment="bottom")
        with col1:
            dir_path_str = st.text_input(
                "Directory Path", placeholder="e.g., /path/to/documents", key="dir_path"
            )
        with col2:
            st.button(
                "Browse",
                on_click=_browse_callback,
                use_container_width=True,
                help="Open a dialog to select a folder.",
            )
        with col3:
            st.button(
                "Clear",
                on_click=_clear_callback,
                use_container_width=True,
                help="Clear the directory path.",
            )

        selected_exts = st.multiselect(
            "File types to process",
            options=[".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx"],
            default=[".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx"],
        )

        if dir_path_str:
            dir_path = Path(dir_path_str)
            if dir_path.is_dir():
                files_to_process = [
                    p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in selected_exts
                ]
                st.info(f"Found {len(files_to_process)} supported files in directory.")
            else:
                st.error("Invalid directory path.")

    run_clicked = st.button("Classify Documents", type="primary", disabled=not files_to_process)
    if run_clicked:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp_str}_{uuid.uuid4().hex[:8]}"
        classifier = DocumentClassifier(runtime_cfg)
        risk_evaluator = RiskEvaluator() if runtime_cfg.enable_risk_evaluation else None
        results = []
        total_files = len(files_to_process)
        progress_bar = st.progress(0, text="Starting classification...")
        max_size_bytes = runtime_cfg.max_file_size_mb * 1024 * 1024

        for i, file_obj in enumerate(files_to_process, start=1):
            file_name = file_obj.name
            file_size = 0
            if hasattr(file_obj, "getvalue"):
                # Streamlit UploadedFile
                file_size = file_obj.size
            else:
                # pathlib.Path
                file_size = file_obj.stat().st_size

            if file_size > max_size_bytes:
                st.warning(
                    f"Skipping '{file_name}': file size ({file_size / (1024*1024):.2f} MB) "
                    f"exceeds the {runtime_cfg.max_file_size_mb} MB limit."
                )
                progress_bar.progress(i / total_files, text=f"Processed {i}/{total_files}")
                continue

            if hasattr(file_obj, "getvalue"):
                content_bytes = file_obj.getvalue()
            else:
                content_bytes = file_obj.read_bytes()

            text = extract_text_from_upload(file_name, content_bytes)
            if not text:
                st.warning(f"Skipping empty document: {file_name}")
                progress_bar.progress(i / total_files, text=f"Processed {i}/{total_files}")
                continue

            with st.spinner(f"Classifying: {file_name} ({i}/{total_files})"):
                decision = classifier.classify_document(
                    document_id=f"doc-{run_id}-{i}",
                    document_name=file_name,
                    document_text=text,
                )
                results.append(decision)

                # --- Log run to Excel ---
                try:
                    # Calculate risk metrics for logging purposes
                    log_risk_cost = 0.0
                    log_risk_flag = "Low"
                    if risk_evaluator:
                        r_eval = risk_evaluator.calculate_risk(
                            TCHClassification(decision.classification.value), decision.confidence
                        )
                        log_risk_cost = r_eval.expected_cost
                        if r_eval.is_high_risk:
                            log_risk_flag = "High"

                    log_data = {
                        "timestamp": datetime.now().isoformat(),
                        "run_id": run_id,
                        "document_id": decision.document_id,
                        "document_name": decision.document_name,
                        "supervisor_model": runtime_cfg.supervisor.model,
                        "agent_a_model": runtime_cfg.agent_a.model,
                        "agent_b_model": runtime_cfg.agent_b.model,
                        "model_parameters": {
                            "supervisor_temp": runtime_cfg.supervisor.temperature,
                            "agent_a_temp": runtime_cfg.agent_a.temperature,
                            "agent_b_temp": runtime_cfg.agent_b.temperature,
                        },
                        "prompt_version": "1",
                        "classification": decision.classification.value,
                        "confidence": decision.confidence,
                        "consensus_reached": decision.consensus_reached,
                        "rounds_used": decision.rounds_used,
                        "estimated_cost": decision.estimated_cost,
                        "risk_cost": log_risk_cost,
                        "risk_flag": log_risk_flag,
                        "input_tokens": decision.total_token_usage.prompt_tokens if decision.total_token_usage else 0,
                        "output_tokens": decision.total_token_usage.completion_tokens if decision.total_token_usage else 0,
                        "ground_truth": None,
                    }
                    log_run_to_excel(log_data)
                except Exception as e:
                    print(f"Failed to log to Excel: {e}")

            progress_bar.progress(i / total_files, text=f"Processed {i}/{total_files}")

        if results:
            # --- New file export logic ---
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            # Export summary to CSV
            csv_filename = results_dir / f"classify_{run_id}.csv"
            csv_headers = [
                "document_id",
                "document_name",
                "classification",
                "confidence",
                "consensus_reached",
                "consensus_score",
                "rounds_used",
                "estimated_cost",
                "supervisor_cost",
                "agent_a_cost",
                "agent_b_cost",
                "total_duration_s",
                "supervisor_duration_s",
                "agent_a_duration_s",
                "agent_b_duration_s",
                "risk_adjusted_classification",
                "risk_expected_cost",
                "risk_is_high",
                "risk_reasoning",
            ]
            try:
                with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_headers)
                    writer.writeheader()
                    for d in results:
                        # Calculate risk and determine final classification
                        if risk_evaluator:
                            risk_eval = risk_evaluator.calculate_risk(
                                TCHClassification(d.classification.value), d.confidence
                            )
                            final_classification = TCHClassification.HUMAN_REVIEW.value if risk_eval.is_high_risk else d.classification.value
                        else:
                            risk_eval, final_classification = None, d.classification.value

                        writer.writerow({
                                "document_id": d.document_id,
                                "document_name": d.document_name,
                                "classification": final_classification,
                                "confidence": d.confidence,
                                "consensus_reached": d.consensus_reached,
                                "consensus_score": d.consensus_score,
                                "rounds_used": d.rounds_used,
                                "estimated_cost": d.estimated_cost,
                                "supervisor_cost": d.supervisor_cost,
                                "agent_a_cost": d.agent_a_cost,
                                "agent_b_cost": d.agent_b_cost,
                                "total_duration_s": d.total_duration_s,
                                "supervisor_duration_s": d.supervisor_duration_s,
                                "agent_a_duration_s": d.agent_a_duration_s,
                                "agent_b_duration_s": d.agent_b_duration_s,
                                "risk_adjusted_classification": risk_eval.adjusted_prediction.value if risk_eval else "N/A",
                                "risk_expected_cost": risk_eval.expected_cost if risk_eval else 0.0,
                                "risk_is_high": risk_eval.is_high_risk if risk_eval else False,
                                "risk_reasoning": risk_eval.reasoning if risk_eval else "Risk evaluation disabled.",
                        })
            except IOError as e:
                st.error(f"Failed to write CSV file: {e}")

            # Export detailed results to JSON
            json_filename = results_dir / f"classify_{run_id}.json"
            json_payload = []
            for d in results:
                item = d.model_dump(mode="json")
                if risk_evaluator:
                    risk_eval = risk_evaluator.calculate_risk(
                        TCHClassification(d.classification.value), d.confidence
                    )
                    final_classification = TCHClassification.HUMAN_REVIEW.value if risk_eval.is_high_risk else d.classification.value
                
                    item["classification"] = final_classification  # Override with workflow decision
                    item["risk_evaluation"] = {
                        "adjusted_prediction": risk_eval.adjusted_prediction.value,
                        "expected_cost": risk_eval.expected_cost,
                        "is_high_risk": risk_eval.is_high_risk,
                        "reasoning": risk_eval.reasoning,
                    }
                json_payload.append(item)

            try:
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(json_payload, f, indent=2)
            except IOError as e:
                st.error(f"Failed to write JSON file: {e}")

            st.session_state["results"] = results
            st.session_state["run_id"] = run_id
            st.session_state["results_dir"] = str(results_dir)

            # Add to history
            if "run_history" not in st.session_state:
                st.session_state["run_history"] = []
            st.session_state["run_history"].append({
                "run_id": run_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
                "results_dir": str(results_dir),
            })

            # Clear the uploader by incrementing the key
            st.session_state["uploader_key"] += 1
            st.rerun()
        else:
            st.warning("No results to display.")

    _render_results(runtime_cfg)


if __name__ == "__main__":
    main()
