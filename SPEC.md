# Project Specification: Multi-Agent Document Classifier

**Project Root**: `c:\projects\TCH\ragclassifier`

## 1. Executive Summary

This project implements a robust document classification system using a multi-agent LLM approach. It classifies documents as `RESTRICTED`, `CONFIDENTIAL`, or `PUBLIC` based on a defined rubric.

## 2. Technical Architecture

### Core Frameworks
-   **LangGraph**: Manages the state machine and workflow (`src/classifier.py`).
-   **LangChain**: Handles LLM interactions (`src/agents.py`).
-   **Streamlit**: Provides the web UI (`app.py`).
-   **Pydantic**: Enforces structured data (`src/models.py`).

### Agent Roles
1.  **Supervisor**: Orchestrates the process, generates reconciliation guidance, and makes the final decision.
2.  **Worker Agents (A & B)**: Independently classify documents.

### Configuration & Providers
-   **OpenAI**: Native support via `ChatOpenAI`.
-   **OpenRouter**: Supported via `ChatOpenAI` client pointing to OpenRouter base URL.
-   **Switching**: Controlled by `USE_OPENROUTER` env var and `src/config.py`.

## 3. Data Flow

1.  **Input**: File upload or directory scan (`.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.xlsx`).
2.  **Graph Execution**:
    -   Parallel execution of Agent A and Agent B.
    -   Consensus check (Label match + Confidence threshold).
    -   Reconciliation loop (Supervisor guidance) if consensus fails.
    -   Finalization by Supervisor.
3.  **Output**: `SupervisorDecision` object containing classification, reasoning, and token usage stats.

## 4. Key Implementation Details

### `src/config.py`
-   Loads env vars.
-   Handles `USE_OPENROUTER` logic to select correct model names and base URLs.

### `src/agents.py`
-   `build_chat_model`: Instantiates `ChatOpenAI` with appropriate API key and Base URL based on config.
-   `classify_with_agent`: Structured output generation for worker agents.

### `app.py`
-   **History**: Persists run history in `st.session_state`.
-   **Visualization**: Altair scatter plot for confidence trends with interactive selection.
-   **Theme**: Forced Dark Mode via CSS injection.

## 5. Development Tasks

1.  **Environment**:
    -   Install with `uv sync`.
    -   Configure `.env`.

2.  **Testing**:
    -   Unit tests: `uv run pytest`.
    -   Offline simulation: `uv run dummy-test`.

3.  **Observability**:
    -   Configure LangSmith keys in `.env` to trace agent execution.