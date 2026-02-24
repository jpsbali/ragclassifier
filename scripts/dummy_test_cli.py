#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from unittest.mock import patch

import src.classifier as classifier_module
from src.classifier import DocumentClassifier
from src.config import AgentModelConfig, AppConfig, ConsensusConfig
from src.document_loader import extract_text_from_upload
from src.models import (
    AgentVote,
    ClassificationLabel,
    ReconciliationGuidance,
    SupervisorDecision,
    TokenUsage,
)


RESTRICTED_TERMS = [
    "social security number",
    "ssn",
    "credit card",
    "debit card",
    "account username and password",
    "network architecture",
    "source code",
    "vulnerability scan",
    "red team",
    "attorney-client",
]

PUBLIC_TERMS = [
    "press release",
    "public website",
    "marketing brochure",
    "conference materials",
]


def _build_offline_config(max_rounds: int, min_confidence: float) -> AppConfig:
    return AppConfig(
        supervisor=AgentModelConfig(
            name="supervisor",
            model="offline-supervisor",
            base_url="http://offline",
            api_key="offline",
        ),
        agent_a=AgentModelConfig(
            name="agent_a",
            model="offline-agent-a",
            base_url="http://offline",
            api_key="offline",
        ),
        agent_b=AgentModelConfig(
            name="agent_b",
            model="offline-agent-b",
            base_url="http://offline",
            api_key="offline",
        ),
        consensus=ConsensusConfig(
            min_confidence=min_confidence,
            max_rounds=max_rounds,
        ),
        use_openrouter=False,
        max_file_size_mb=50,  # Set a high limit for offline tests
    )


def _heuristic_label(text: str) -> tuple[ClassificationLabel, float, str, list[str]]:
    lower = text.lower()
    if any(term in lower for term in RESTRICTED_TERMS):
        return (
            ClassificationLabel.RESTRICTED,
            0.96,
            "Matched restricted-risk terms in the rubric.",
            ["restricted term match"],
        )
    if any(term in lower for term in PUBLIC_TERMS):
        return (
            ClassificationLabel.PUBLIC,
            0.94,
            "Matched clear public-facing terms.",
            ["public term match"],
        )
    return (
        ClassificationLabel.CONFIDENTIAL,
        0.92,
        "Defaulted to confidential for internal-sensitive style content.",
        ["default confidential rule"],
    )


def run_dummy_classification(
    documents: list[Path],
    force_disagreement: bool,
    max_rounds: int,
    min_confidence: float,
) -> list[dict]:
    def fake_classify_with_agent(
        llm,
        document_name: str,
        document_text: str,
        round_num: int,
        retry_context: str | None = None,
    ) -> tuple[AgentVote, TokenUsage]:
        label, confidence, reason, matched_points = _heuristic_label(document_text)

        if force_disagreement and round_num == 1:
            if llm == "agent_a":
                label = ClassificationLabel.CONFIDENTIAL
                confidence = 0.80
                reason = "Agent A interpreted it as internal-sensitive."
                matched_points = ["project plans"]
            elif llm == "agent_b":
                label = ClassificationLabel.PUBLIC
                confidence = 0.81
                reason = "Agent B interpreted it as public-facing."
                matched_points = ["conference materials"]
        elif retry_context:
            confidence = min(0.99, confidence + 0.03)

        vote = AgentVote(
            classification=label,
            confidence=confidence,
            reason=reason,
            matched_rubric_points=matched_points,
        )
        tokens = TokenUsage(prompt_tokens=500, completion_tokens=100, total_tokens=600)
        return vote, tokens

    def fake_build_reconciliation_guidance(
        **kwargs,
    ) -> tuple[ReconciliationGuidance, TokenUsage]:
        guidance = ReconciliationGuidance(
            instructions_for_retry="Re-evaluate intended audience and sensitivity signals."
        )
        tokens = TokenUsage(prompt_tokens=800, completion_tokens=50, total_tokens=850)
        return guidance, tokens

    def fake_finalize_with_supervisor(
        supervisor_llm,
        document_id: str,
        document_name: str,
        rounds_used: int,
        consensus_reached: bool,
        consensus_score: float,
        agent_a_vote: AgentVote,
        agent_b_vote: AgentVote,
    ) -> tuple[SupervisorDecision, TokenUsage]:
        if agent_a_vote.classification == agent_b_vote.classification:
            chosen = agent_a_vote
        elif agent_a_vote.confidence >= agent_b_vote.confidence:
            chosen = agent_a_vote
        else:
            chosen = agent_b_vote

        decision = SupervisorDecision(
            document_id=document_id,
            document_name=document_name,
            classification=chosen.classification,
            confidence=chosen.confidence,
            reason="Offline dummy supervisor decision.",
            matched_rubric_points=chosen.matched_rubric_points,
            agent_a_vote=agent_a_vote,
            agent_b_vote=agent_b_vote,
            consensus_reached=consensus_reached,
            consensus_score=consensus_score,
            rounds_used=rounds_used,
        )
        tokens = TokenUsage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)
        return decision, tokens

    # Use context managers to safely mock the functions that would make API calls
    with patch.object(
        classifier_module, "build_chat_model", new=lambda cfg: cfg.name
    ), patch.object(
        classifier_module, "classify_with_agent", new=fake_classify_with_agent
    ), patch.object(
        classifier_module,
        "build_reconciliation_guidance",
        new=fake_build_reconciliation_guidance,
    ), patch.object(
        classifier_module, "finalize_with_supervisor", new=fake_finalize_with_supervisor
    ):
        classifier = DocumentClassifier(_build_offline_config(max_rounds, min_confidence))
        results = []
        for idx, path in enumerate(documents, start=1):
            text = extract_text_from_upload(path.name, path.read_bytes())
            if not text.strip():
                continue
            decision = classifier.classify_document(f"doc-{idx}", path.name, text)
            results.append(decision.model_dump())
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline dummy tests for document classification without API calls."
    )
    parser.add_argument(
        "--document",
        action="append",
        default=[],
        help="Path to a document file or directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--force-disagreement",
        action="store_true",
        help="Force round-1 disagreement between agent_a and agent_b.",
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    args = parser.parse_args()

    input_paths = [Path(p).resolve() for p in args.document]
    documents = []
    supported_exts = {".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx"}

    if not input_paths:
        default_doc = Path("ClassifyingRules.docx").resolve()
        if default_doc.exists():
            input_paths = [default_doc]

    for path in input_paths:
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in supported_exts:
                    documents.append(child)
        elif path.exists():
            documents.append(path)
        else:
            print(f"Warning: Path {path} does not exist.", file=sys.stderr)

    if not documents:
        raise SystemExit(
            "No documents provided. Use --document <path> (can be repeated)."
        )

    results = run_dummy_classification(
        documents=documents,
        force_disagreement=args.force_disagreement,
        max_rounds=max(1, args.max_rounds),
        min_confidence=max(0.0, min(1.0, args.min_confidence)),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
