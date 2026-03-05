# System Architecture

## Overview
The TCH RAG Classifier is designed as a pipeline of specialized components that progressively refine the classification of a document. The architecture emphasizes separation of concerns between the "Creative" classification (LLMs) and the "Deterministic" risk assessment (Math).

## Component Diagram

```mermaid
graph TD
    User[User / Streamlit UI] -->|Uploads| Loader[Document Loader]
    Loader -->|Text| Graph[LangGraph Classifier]
    
    subgraph "Multi-Agent Classification"
        Graph -->|Parallel| AgentA[Agent A]
        Graph -->|Parallel| AgentB[Agent B]
        AgentA --> Evaluator[Consensus Check]
        AgentB --> Evaluator
        Evaluator -->|Disagreement| Supervisor[Supervisor Agent]
        Supervisor -->|Guidance| AgentA
        Supervisor -->|Guidance| AgentB
        Evaluator -->|Consensus/Max Rounds| Finalizer[Supervisor Finalize]
        Finalizer -->|Decision| RiskEngine[Risk Evaluator]
    end
    
    subgraph "Risk Evaluation (Deterministic)"
        RiskEngine -->|Config| CostMatrix[Cost Matrix (.env)]
        RiskEngine -->|Input: Prediction, Confidence| ExpectedCost[Expected Cost Calculation]
        ExpectedCost -->|Output: Adjusted Pred, Expected Cost| RiskFlag[Risk Flag (High/Low)]
        RiskFlag -->|Threshold Check| HumanReviewTrigger[Human Review Trigger]
    end
    
    HumanReviewTrigger -->|Result| UI[Results Display]
    UI -->|Export| PDF[PDF Report]
    UI -->|Export| CSV[CSV/JSON]
```

## Key Components

### 1. The Classifier Graph (`src/classifier.py`)
A state machine that manages the lifecycle of a document classification. It handles the "Debate" loop between agents and enforces the consensus rules.

### 2. The Supervisor (`src/agents.py`)
Acts as the arbiter. In the final step, it is explicitly instructed to flag documents for `HUMAN_REVIEW` if the sub-agents cannot agree after $N$ rounds, ensuring the system fails safely rather than hallucinating a compromise.

### 3. The Risk Evaluator (`risk_evaluator.py`)
A pure Python component that applies business logic. It does not use an LLM.
-   **Input**: Classification Label, Confidence Score.
-   **Logic**: Asymmetric Cost Calculation (False Negatives are expensive).
-   **Output**: Adjusted Label, Expected Cost, High Risk Flag.

### 4. The Report Generator (`report_generator.py`)
Uses `ReportLab` to compile the session's results into a professional PDF summary, highlighting high-priority items for auditors.