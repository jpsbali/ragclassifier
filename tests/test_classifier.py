import src.classifier as classifier_module
from src.classifier import DocumentClassifier
from src.config import AgentModelConfig, AppConfig, ConsensusConfig
from src.models import (
    AgentVote,
    ClassificationLabel,
    ReconciliationGuidance,
    SupervisorDecision,
)


def _test_config(max_rounds: int = 3) -> AppConfig:
    return AppConfig(
        supervisor=AgentModelConfig(
            name="supervisor",
            model="dummy-supervisor",
            base_url="http://dummy",
            api_key="dummy",
        ),
        agent_a=AgentModelConfig(
            name="agent_a",
            model="dummy-a",
            base_url="http://dummy",
            api_key="dummy",
        ),
        agent_b=AgentModelConfig(
            name="agent_b",
            model="dummy-b",
            base_url="http://dummy",
            api_key="dummy",
        ),
        consensus=ConsensusConfig(min_confidence=0.9, max_rounds=max_rounds),
    )


def _make_decision(
    supervisor_llm,
    document_id: str,
    document_name: str,
    rounds_used: int,
    consensus_reached: bool,
    consensus_score: float,
    agent_a_vote: AgentVote,
    agent_b_vote: AgentVote,
) -> SupervisorDecision:
    if agent_a_vote.classification == agent_b_vote.classification:
        final_label = agent_a_vote.classification
        final_conf = max(agent_a_vote.confidence, agent_b_vote.confidence)
    elif agent_a_vote.confidence >= agent_b_vote.confidence:
        final_label = agent_a_vote.classification
        final_conf = agent_a_vote.confidence
    else:
        final_label = agent_b_vote.classification
        final_conf = agent_b_vote.confidence

    return SupervisorDecision(
        document_id=document_id,
        document_name=document_name,
        classification=final_label,
        confidence=final_conf,
        reason="Offline test finalization.",
        matched_rubric_points=["offline test"],
        agent_a_vote=agent_a_vote,
        agent_b_vote=agent_b_vote,
        consensus_reached=consensus_reached,
        consensus_score=consensus_score,
        rounds_used=rounds_used,
    )


def test_classifier_reaches_consensus_in_first_round(monkeypatch) -> None:
    monkeypatch.setattr(classifier_module, "build_chat_model", lambda cfg: cfg.name)

    def fake_classify(**kwargs) -> AgentVote:
        return AgentVote(
            classification=ClassificationLabel.CONFIDENTIAL,
            confidence=0.95,
            reason="Sensitive internal content.",
            matched_rubric_points=["default confidential"],
        )

    monkeypatch.setattr(classifier_module, "classify_with_agent", fake_classify)
    monkeypatch.setattr(
        classifier_module,
        "build_reconciliation_guidance",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Reconciliation should not be called")
        ),
    )
    monkeypatch.setattr(
        classifier_module,
        "finalize_with_supervisor",
        lambda **kwargs: _make_decision(**kwargs),
    )

    classifier = DocumentClassifier(_test_config())
    decision = classifier.classify_document("doc-1", "doc.txt", "Internal project plan")

    assert decision.consensus_reached is True
    assert decision.rounds_used == 1
    assert decision.classification == ClassificationLabel.CONFIDENTIAL


def test_classifier_retries_when_agents_disagree(monkeypatch) -> None:
    reconcile_calls = {"count": 0}
    monkeypatch.setattr(classifier_module, "build_chat_model", lambda cfg: cfg.name)

    def fake_classify(**kwargs) -> AgentVote:
        llm = kwargs["llm"]
        round_num = kwargs["round_num"]
        if round_num == 1:
            if llm == "agent_a":
                return AgentVote(
                    classification=ClassificationLabel.CONFIDENTIAL,
                    confidence=0.82,
                    reason="Looks internal.",
                    matched_rubric_points=["project plans"],
                )
            return AgentVote(
                classification=ClassificationLabel.PUBLIC,
                confidence=0.83,
                reason="Could be external conference content.",
                matched_rubric_points=["conference materials"],
            )
        return AgentVote(
            classification=ClassificationLabel.CONFIDENTIAL,
            confidence=0.96,
            reason="After retry, internal and non-public.",
            matched_rubric_points=["default confidential"],
        )

    def fake_reconcile(**kwargs) -> ReconciliationGuidance:
        reconcile_calls["count"] += 1
        return ReconciliationGuidance(
            instructions_for_retry="Focus on internal-only signals and intended audience."
        )

    monkeypatch.setattr(classifier_module, "classify_with_agent", fake_classify)
    monkeypatch.setattr(
        classifier_module, "build_reconciliation_guidance", fake_reconcile
    )
    monkeypatch.setattr(
        classifier_module,
        "finalize_with_supervisor",
        lambda **kwargs: _make_decision(**kwargs),
    )

    classifier = DocumentClassifier(_test_config(max_rounds=3))
    decision = classifier.classify_document("doc-2", "doc.txt", "Product roadmap draft")

    assert reconcile_calls["count"] == 1
    assert decision.consensus_reached is True
    assert decision.rounds_used == 2
    assert decision.classification == ClassificationLabel.CONFIDENTIAL


def test_classifier_stops_at_max_rounds(monkeypatch) -> None:
    reconcile_calls = {"count": 0}
    monkeypatch.setattr(classifier_module, "build_chat_model", lambda cfg: cfg.name)

    def fake_classify(**kwargs) -> AgentVote:
        llm = kwargs["llm"]
        if llm == "agent_a":
            return AgentVote(
                classification=ClassificationLabel.RESTRICTED,
                confidence=0.60,
                reason="Contains potential account artifacts.",
                matched_rubric_points=["account data"],
            )
        return AgentVote(
            classification=ClassificationLabel.PUBLIC,
            confidence=0.61,
            reason="Appears broad and generic.",
            matched_rubric_points=["public website content"],
        )

    def fake_reconcile(**kwargs) -> ReconciliationGuidance:
        reconcile_calls["count"] += 1
        return ReconciliationGuidance(instructions_for_retry="Re-check data sensitivity.")

    monkeypatch.setattr(classifier_module, "classify_with_agent", fake_classify)
    monkeypatch.setattr(
        classifier_module, "build_reconciliation_guidance", fake_reconcile
    )
    monkeypatch.setattr(
        classifier_module,
        "finalize_with_supervisor",
        lambda **kwargs: _make_decision(**kwargs),
    )

    classifier = DocumentClassifier(_test_config(max_rounds=2))
    decision = classifier.classify_document("doc-3", "doc.txt", "Mixed content")

    assert reconcile_calls["count"] == 1
    assert decision.consensus_reached is False
    assert decision.rounds_used == 2
