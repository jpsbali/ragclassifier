# Project Layout

This document describes the structure of the project, detailing the purpose of key directories and files.

```
.
├── .env.example         # Example environment variables file
├── .gitignore           # Standard git ignore file
├── ARCHITECTURE.md      # High-level system architecture
├── ClassifyingRules.docx # The source document for the classification rubric
├── PROJECT.md           # (This file) Project file and directory layout
├── README.md            # Main project README with setup and usage instructions
├── SPEC.md              # Project specification for developers
├── app.py               # Main Streamlit application entrypoint
├── pyproject.toml       # Project metadata and dependencies (for `uv`)
├── scripts
│   └── dummy_test_cli.py # Offline, end-to-end test script with no API calls
└── src
    ├── __init__.py
    ├── agents.py        # Functions for building and interacting with LLM agents
    ├── classifier.py    # Core classification logic orchestrating the agents
    ├── config.py        # Pydantic models for application and agent configuration
    ├── document_loader.py # Utilities for extracting text from uploaded files
    ├── models.py        # Pydantic models for structured data (votes, decisions)
    └── rubric.py        # Contains the classification rubric and system prompts
└── tests
    └── test_document_loader.py # Unit tests for the document loader utility
```

## Directory and File Descriptions

### Root Directory

-   **`app.py`**: The entrypoint for the Streamlit web application. It handles the UI, configuration sidebar, and invokes the `DocumentClassifier`.
-   **`pyproject.toml`**: Defines project dependencies for both runtime and development, managed by the `uv` package manager. It also configures the `dummy-test` script entrypoint.
-   **`ClassifyingRules.docx`**: The human-readable source document containing the classification rules. The text from this document is used to generate `src/rubric.py`.
-   **`.env.example`**: A template for the `.env` file, where users should store their `OPENAI_API_KEY` and other secrets or configurations.

### `src/`

The main source code for the application logic.

-   **`classifier.py`**: The heart of the application. The `DocumentClassifier` class manages the state and flow of the multi-agent classification process, including the reconciliation loop.
-   **`agents.py`**: Contains the functions that define the behavior of the agents. This includes building the `ChatOpenAI` clients and formatting the prompts sent to the LLMs for classification, reconciliation, and finalization.
-   **`config.py`**: Defines the Pydantic models used for configuring the application (e.g., `AppConfig`, `AgentModelConfig`). It handles loading configuration from the environment.
-   **`models.py`**: Defines the Pydantic models for the structured data passed between agents and as final output (e.g., `AgentVote`, `SupervisorDecision`). This is crucial for reliable data exchange with the LLMs.
-   **`rubric.py`**: A critical file that stores the classification rubric text and the main system prompts as Python constants. Centralizing the prompts here makes them easy to manage and reuse.
-   **`document_loader.py`**: A utility module with functions to extract plain text from various file formats (`.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.xlsx`).

### `scripts/`

-   **`dummy_test_cli.py`**: An essential tool for development and testing. It runs the entire classification workflow using mocked (heuristic-based) agent responses instead of making live LLM calls. This allows for fast, free, and deterministic testing of the application's logic and control flow.

### `tests/`

-   **`test_document_loader.py`**: Contains unit tests for the text extraction functions, ensuring they work correctly for different file types.