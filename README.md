# LangGraph Multi-Agent Document Classifier

This app classifies documents into:
- `RESTRICTED`
- `CONFIDENTIAL`
- `PUBLIC`

It uses:
- 1 Supervisor agent
- 2 Sub-agents
- Configurable model endpoint + API key per agent
- Retry loop when sub-agents disagree or confidence is below threshold
- Pydantic structured output from supervisor

## Quickstart

1. Sync dependencies with `uv`:

```bash
uv sync
```

2. Create `.env` from `.env.example` and set keys/models/endpoints.
3. Run:

```bash
uv run streamlit run app.py
```

## Notes

- API keys are loaded from `.env` by default and are editable in the Streamlit sidebar.
- Each agent can use a different `base_url` and `model`.
