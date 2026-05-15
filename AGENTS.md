# How the Agents Work

## Agent A & Agent B (Worker Agents)

- Run **in parallel** on the same document
- Each independently classifies it as `RESTRICTED`, `CONFIDENTIAL`, or `PUBLIC` using the rubric defined in `src/rubric.py`
- Return a structured vote containing:
  - Classification label
  - Confidence score (0–1)
  - Evidence-based reasoning
  - Matched rubric points
- They're stateless — on retries they receive the Supervisor's guidance as additional context but make their own independent judgment

## Supervisor (Orchestrator)

Has two roles depending on the workflow stage:

### 1. Reconciliation (Mid-workflow)

When agents disagree, the Supervisor reviews both votes and the document, then writes neutral "retry instructions" to help agents converge — without forcing a label.

### 2. Finalization (End of workflow)

After consensus is reached (or max rounds hit), the Supervisor makes the final call. It enforces safety rules:

- If agents **still disagree after 2+ rounds** → `HUMAN_REVIEW`
- If there's a **two-tier gap** (one says `PUBLIC`, other says `RESTRICTED`) → `HUMAN_REVIEW` with `HIGH` priority
- If votes agree with high confidence → keeps the agreed class

## The Flow

```
Round 1: Agent A + Agent B classify in parallel
    ↓
Consensus check: labels match AND avg confidence ≥ 0.90?
    ↓ No
Supervisor writes reconciliation guidance
    ↓
Round 2: Agents retry with guidance context
    ↓
... repeat up to max_rounds (default 3) ...
    ↓
Supervisor finalizes (picks best-supported label or flags HUMAN_REVIEW)
```

## Risk Evaluator (Deterministic — No LLM)

Runs **after** the Supervisor's final decision. It is not an LLM agent — it's pure math.

- **Adjustment Rule**: If confidence is below threshold (default 0.90), bumps the label up one sensitivity level (e.g., `PUBLIC` → `CONFIDENTIAL`)
- **Expected Cost Calculation**: Uses an asymmetric cost matrix to calculate the expected cost of being wrong (under-classification is penalized heavily)
- **Risk Flagging**: If expected cost exceeds the threshold (default $5.00) → flags for human review regardless of what the agents decided

## Design Principle

The LLM agents handle the "creative" classification work, while the Risk Evaluator applies deterministic business rules on top to catch high-cost errors. This separation ensures the system fails safely — erring on the side of over-classification rather than under-classification.
