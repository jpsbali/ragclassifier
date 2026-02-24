# LangGraph Multi-Agent Document Classifier

This application classifies documents into `RESTRICTED`, `CONFIDENTIAL`, or `PUBLIC` categories using a multi-agent LLM architecture. It features a Supervisor agent that coordinates two worker agents to reach a consensus based on a strict rubric.

## Features

-   **Multi-Agent Architecture**: Supervisor + 2 Worker Agents with debate/reconciliation loops.
-   **Model Flexibility**: Supports OpenAI and OpenRouter (Anthropic, Google, etc.).
-   **Broad File Support**: Processes `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, and `.xlsx`.
-   **Observability**: Integrated with LangSmith for tracing.
-   **Rich UI**:
    -   Streamlit-based interface with Dark Mode.
    -   Historical run tracking with persistence.
    -   Confidence trend visualization (Altair charts).
    -   Export results to CSV/JSON/ZIP.
    -   Directory browsing and drag-and-drop uploads.

## Setup

1.  **Install Dependencies**:
    Ensure you have `uv` installed.
    ```bash
    uv sync
    ```

2.  **Configuration**:
    Copy `.env.example` to `.env` and configure your keys.
    ```bash
    cp .env.example .env
    ```

    **For OpenAI:**
    -   Set `USE_OPENROUTER=false`
    -   Set `OPENAI_API_KEY`

    **For OpenRouter:**
    -   Set `USE_OPENROUTER=true`
    -   Set `OPENROUTER_API_KEY`
    -   Update `OPENROUTER_SUPERVISOR_MODEL`, etc.

3.  **LangSmith (Optional)**:
    Set `LANGCHAIN_TRACING_V2=true` and provide `LANGCHAIN_API_KEY` in `.env`.

## Usage

Run the Streamlit application:
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