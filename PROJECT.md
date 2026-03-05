# Project Status & Roadmap

## Current Status
**Active Development** - The core multi-agent classification engine is stable. Recent efforts have focused on "Day 2" operations: risk management, reporting, and safety guardrails.

## Recent Updates
-   **Risk Engine**: Implemented `RiskEvaluator` with asymmetric cost matrix, adjustment rules, and a configurable risk threshold.
-   **Human Review Workflow**: Integrated `HUMAN_REVIEW` and `HIGH_PRIORITY` states directly into the supervisor's decision-making process for persistent disagreements or two-tier classification gaps.
-   **Reporting**: Added PDF report generation with executive summaries, including counts for human review and high-priority items.
-   **UI Enhancements**:
    -   Streamlit sidebar now includes a clickable "Human Review Queue" and a "Reset All to Defaults" button.
    -   Main results table uses distinct icons (🚨, ⚠️, ✅) for different risk levels.
    -   Classification distribution chart added.
-   **Configuration**: Granular agent configuration (model, API key, costs) is now collapsible and editable in the sidebar.
-   **Run Management**: Timestamped run IDs and document IDs for better traceability.

## Roadmap

### Phase 1: Core Stability (Completed)
-   [x] Multi-agent graph
-   [x] Supervisor orchestration
-   [x] Basic UI

### Phase 2: Risk & Operations (Current)
-   [x] Cost-sensitive risk evaluation
-   [x] PDF Reporting
-   [x] Run History & Export
-   [x] Configurable LLM endpoints

### Phase 3: Advanced Features (Planned)
-   [ ] **Feedback Loop**: Allow human review decisions to be fed back into the system as few-shot examples.
-   [ ] **Batch Processing**: Headless mode for processing large directories without UI.
-   [ ] **Custom Rubrics**: Allow uploading a custom rubric file via the UI.

## Known Issues
-   PDF generation requires `reportlab` installed.