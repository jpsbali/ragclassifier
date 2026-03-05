# TCH Multi-Agent Document Classifier

This application classifies documents into `RESTRICTED`, `CONFIDENTIAL`, or `PUBLIC` categories using a multi-agent LLM architecture. It features a Supervisor agent that coordinates two worker agents to reach a consensus based on a strict rubric, and incorporates an asymmetric risk evaluation engine to flag high-cost errors for human review.

## Features

-   **Multi-Agent Debate**: A Supervisor agent orchestrates two worker agents (Agent A and Agent B) to classify documents. They engage in debate and reconciliation loops to reach consensus.
-   **Asymmetric Risk Evaluation**: A deterministic `RiskEvaluator` calculates the "Expected Cost" of potential misclassifications. It applies an "Adjustment Rule" for low-confidence predictions and flags documents for human review if the expected cost exceeds a configurable threshold.
-   **Human Review Workflow**:
    -   Documents are automatically flagged for `HUMAN_REVIEW` if agents persistently disagree or if there's a significant "two-tier" classification gap (e.g., one agent says `PUBLIC`, another `RESTRICTED`).
    -   High-priority human review items are explicitly marked.
-   **Comprehensive Reporting**:
    -   Generates PDF summary reports with executive summaries, classification distribution, and detailed results, highlighting high-priority items.
    -   Exports detailed results to CSV and JSON.
-   **Rich User Interface (Streamlit)**:
    -   Interactive dashboard with Dark Mode.
    -   **Historical Runs**: Tracks and allows re-loading of past classification runs.
    -   **Human Review Queue**: A clickable sidebar queue lists documents requiring manual review.
    -   **Configurable LLMs**: Granular settings for each agent (model, API key, base URL, costs, temperature) are editable in the sidebar.
    -   **Reset to Defaults**: Button to revert all configurations to `.env` values.
    -   **Visual Indicators**: Uses distinct icons (🚨, ⚠️, ✅) for different risk levels in the results table.
    -   **Classification Distribution**: A bar chart shows the breakdown of classifications for the current run.
-   **Flexible Input**: Supports file uploads (drag-and-drop) or scanning local directories for `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, and `.xlsx` files.
-   **Observability**: Integrated with LangSmith for detailed tracing of agent interactions.

## Setup

1.  **Prerequisites**:
    -   Python 3.10+
    -   `uv` (recommended for fast dependency management) or `pip`

2.  **Install Dependencies**:
    ```bash
    uv sync
    ```

2.  **Configuration**:
    Copy `.env.example` to `.env` and configure your keys.
    ```bash
    cp .env.example .env
    ```

    **For OpenAI:**
    -   Set `USE_OPENROUTER=false` in `.env`.
    -   Provide your `OPENAI_API_KEY`.

    **For OpenRouter:**
    -   Set `USE_OPENROUTER=true` in `.env`.
    -   Provide your `OPENROUTER_API_KEY`.
    -   Configure `OPENROUTER_SUPERVISOR_MODEL`, `OPENROUTER_AGENT_A_MODEL`, etc., as desired.

    **Risk Evaluation**:
    -   Adjust `COST_TRUE_...` variables in `.env` to define the asymmetric cost matrix.
    -   Set `COST_RISK_THRESHOLD` to define the acceptable error cost.
    -   Toggle `ENABLE_RISK_EVALUATION` (default `true`).

3.  **LangSmith (Optional)**:
    Set `LANGCHAIN_TRACING_V2=true` and provide `LANGCHAIN_API_KEY` in `.env`.

## Usage

1.  **Run the Streamlit application**:
```bash
uv run streamlit run app.py
```

### Offline Testing

Run the test suite without API calls:
```bash
uv run pytest
```

Run the dummy CLI for end-to-end logic testing:
```bash
uv run dummy-test --document ClassifyingRules.docx
```

## Project Structure

-   `src/classifier.py`: Main LangGraph workflow.
-   `src/agents.py`: LLM interaction logic.
-   `src/config.py`: Configuration loading.
-   `app.py`: Streamlit UI.