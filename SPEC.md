# Project Specification: Multi-Agent Document Classifier

**Project Root**: `c:\projects\TCH\ragclassifier`

## 1. Executive Summary

This project implements a robust document classification system using a multi-agent LLM approach. It classifies documents as `RESTRICTED`, `CONFIDENTIAL`, or `PUBLIC` based on a defined rubric. It incorporates an asymmetric Risk Evaluation engine to flag high-cost errors for `HUMAN_REVIEW`.
 
## 2. Technical Architecture

### Core Frameworks
-   **LangGraph**: Manages the state machine and workflow (`src/classifier.py`).
-   **LangChain**: Handles LLM interactions (`src/agents.py`).
-   **Streamlit**: Provides the web UI (`app.py`).
-   **Pydantic**: Enforces structured data (`src/models.py`).
-   **ReportLab**: Generates PDF summary reports (`report_generator.py`). (Optional dependency)

### Agent Roles
1.  **Supervisor**: Orchestrates the process, generates reconciliation guidance, and makes the final decision.
2.  **Worker Agents (A & B)**: Independently classify documents.
3.  **Risk Evaluator**: A deterministic engine that calculates the expected cost of error based on an asymmetric cost matrix and flags high-risk items for human review. It does not use an LLM.

### Configuration & Providers
-   **OpenAI**: Native support via `ChatOpenAI`.
-   **OpenRouter**: Supported via `ChatOpenAI` client pointing to OpenRouter base URL.
-   **Switching**: Controlled by `USE_OPENROUTER` env var and `src/config.py`.
-   **Risk Config**: Cost matrix and thresholds defined in `.env`.
    -   `COST_TRUE_...`: Defines the asymmetric penalties for misclassification.
    -   `COST_RISK_THRESHOLD`: The threshold above which an item is flagged as high-risk.

## 3. Data Flow

1.  **Input**: File upload or directory scan (`.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.xlsx`).
2.  **Graph Execution**:
    -   Parallel execution of Agent A and Agent B.
    -   Consensus check (Label match + Confidence threshold).
    -   Reconciliation loop (Supervisor guidance) if consensus fails.
    -   Finalization by Supervisor (Enforces `HUMAN_REVIEW` on persistent disagreement).
3.  **Risk Evaluation**:
    -   Calculates expected cost of error using the Cost Matrix.
    -   Applies adjustment rules for low confidence.
    -   Flags `HUMAN_REVIEW` (High Priority) if cost > threshold.
4.  **Output**:
    -   CSV/JSON export with risk metadata.
    -   PDF Report with executive summary.
    -   Interactive UI with High-Priority flags (🚨).

## 4. Key Implementation Details

### `src/config.py`
-   Loads env vars.
-   Handles `USE_OPENROUTER` logic to select correct model names and base URLs.
-   Manages Agent configurations (Supervisor, A, B, Evaluator).

### `src/agents.py`
-   `build_chat_model`: Instantiates `ChatOpenAI` with appropriate API key and Base URL based on config.
-   `classify_with_agent`: Structured output generation for worker agents.
-   `finalize_with_supervisor`: Implements logic to trigger `HUMAN_REVIEW` on disagreement.

### `app.py`
-   **History**: Persists run history in `st.session_state`.
-   **Visualization**: Altair scatter plot for confidence trends with interactive selection.
-   **Risk UI**: Displays siren/warning icons for high-risk items and Human Review queue.
-   **Theme**: Forced Dark Mode via CSS injection.

## 5. Development Tasks

1.  **Environment**:
    -   Install with `uv sync`.
    -   Configure `.env`.

2.  **Testing**:
    -   Unit tests: `uv run pytest`.
    -   Risk tests: `python -m unittest test_risk_evaluator.py`.
    -   Offline simulation: `uv run dummy-test`.

3.  **Observability**:
    -   Configure LangSmith keys in `.env` to trace agent execution.