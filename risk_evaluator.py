import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class TCHClassification(str, Enum):
    """Enumeration for TCH data classification levels."""
    PUBLIC = "PUBLIC"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass
class Evaluation:
    """Data structure for the final risk evaluation result."""
    adjusted_prediction: TCHClassification
    expected_cost: float
    is_high_risk: bool
    reasoning: str


class RiskEvaluator:
    """
    Implements the TCH cost-sensitive risk evaluation and adjustment logic.
    
    This class encapsulates several core principles from the TCH design:
    1.  **Adjustment Rule**: Low-confidence predictions (below `CONSENSUS_MIN_CONFIDENCE`)
        are escalated to the next-highest sensitivity level (e.g., PUBLIC -> CONFIDENTIAL)
        to ensure a conservative stance against under-classification.
    2.  **Asymmetric Cost Calculation**: It calculates the "Expected Cost" of an
        error by weighing the probability of each possible mistake against its
        business-defined penalty from the cost matrix.
    3.  **Risk-Based Flagging**: If the calculated `expected_cost` exceeds the
        `COST_RISK_THRESHOLD`, the document is flagged for human review.
    4.  **Workflow State Handling**: It gracefully handles inputs that have already
        been flagged for `HUMAN_REVIEW` by the upstream classification workflow,
        ensuring they are passed through as high-risk.
    """

    # Defines the sensitivity hierarchy: Lower index is less sensitive.
    SENSITIVITY_ORDER = [TCHClassification.PUBLIC, TCHClassification.CONFIDENTIAL, TCHClassification.RESTRICTED]

    def __init__(self):
        """
        Initializes the evaluator by loading the cost matrix and thresholds
        from environment variables.
        """
        self.cost_matrix: Dict[Tuple[TCHClassification, TCHClassification], float] = self._load_cost_matrix()
        self.adjustment_threshold: float = float(os.getenv("CONSENSUS_MIN_CONFIDENCE", 0.90))
        self.risk_threshold: float = float(os.getenv("COST_RISK_THRESHOLD", 5.0))

    def _load_cost_matrix(self) -> Dict[Tuple[TCHClassification, TCHClassification], float]:
        """
        Parses the cost matrix from environment variables into a structured dictionary.
        The key is a tuple (true_label, predicted_label), representing the penalty
        for a specific misclassification.
        """
        return {
            # Errors when True is RESTRICTED
            (TCHClassification.RESTRICTED, TCHClassification.CONFIDENTIAL): float(os.getenv("COST_TRUE_RESTRICTED_PRED_CONFIDENTIAL", 40.0)),
            (TCHClassification.RESTRICTED, TCHClassification.PUBLIC): float(os.getenv("COST_TRUE_RESTRICTED_PRED_PUBLIC", 100.0)),
            # Errors when True is CONFIDENTIAL
            (TCHClassification.CONFIDENTIAL, TCHClassification.RESTRICTED): float(os.getenv("COST_TRUE_CONFIDENTIAL_PRED_RESTRICTED", 3.0)),
            (TCHClassification.CONFIDENTIAL, TCHClassification.PUBLIC): float(os.getenv("COST_TRUE_CONFIDENTIAL_PRED_PUBLIC", 15.0)),
            # Errors when True is PUBLIC
            (TCHClassification.PUBLIC, TCHClassification.RESTRICTED): float(os.getenv("COST_TRUE_PUBLIC_PRED_RESTRICTED", 0.5)),
            (TCHClassification.PUBLIC, TCHClassification.CONFIDENTIAL): float(os.getenv("COST_TRUE_PUBLIC_PRED_CONFIDENTIAL", 2.0)),
        }

    def _apply_adjustment_rule(self, prediction: TCHClassification, confidence: float) -> Tuple[TCHClassification, str]:
        """
        If confidence is below the threshold, bumps the prediction to the next sensitivity level.
        """
        if confidence >= self.adjustment_threshold:
            return prediction, "Confidence is high, no adjustment needed."

        try:
            current_index = self.SENSITIVITY_ORDER.index(prediction)
            if current_index < len(self.SENSITIVITY_ORDER) - 1:
                adjusted_prediction = self.SENSITIVITY_ORDER[current_index + 1]
                reason = f"Low confidence ({confidence:.2f} < {self.adjustment_threshold}). Adjusted '{prediction.value}' to '{adjusted_prediction.value}'."
                return adjusted_prediction, reason
            else:
                # Already at the highest level (RESTRICTED)
                return prediction, f"Low confidence ({confidence:.2f}) but already at highest sensitivity '{prediction.value}'."
        except ValueError:
            return prediction, "Invalid prediction label provided."

    def calculate_risk(self, prediction: TCHClassification, confidence: float) -> Evaluation:
        """
        Calculates the expected cost of a misclassification and determines if it's high risk.

        This method performs the core risk evaluation:
        - If the prediction is already `HUMAN_REVIEW`, it's passed through as high-risk.
        - It applies the 'Adjustment Rule' for low-confidence predictions.
        - It calculates the total expected cost based on the cost matrix.
        - It compares the cost against the risk threshold to set the `is_high_risk` flag.

        Args:
            prediction: The classification label predicted by the upstream process.
            confidence: The confidence score (0.0 to 1.0) for the prediction.

        Returns:
            An `Evaluation` object containing the adjusted prediction, cost, and risk flag.
        """
        if prediction == TCHClassification.HUMAN_REVIEW:
            return Evaluation(
                adjusted_prediction=TCHClassification.HUMAN_REVIEW,
                expected_cost=self.risk_threshold + 1,  # Ensure it's always high risk
                is_high_risk=True,
                reasoning=(
                    "Flagged for human review by the classification workflow due to "
                    "persistent disagreement or a two-tier gap."
                ),
            )

        adjusted_prediction, adjustment_reason = self._apply_adjustment_rule(prediction, confidence)

        prob_error = 1.0 - confidence
        other_labels = [
            label for label in TCHClassification 
            if label != adjusted_prediction and label != TCHClassification.HUMAN_REVIEW
        ]
        prob_per_error_label = prob_error / len(other_labels) if other_labels else 0

        total_expected_cost = 0.0
        for true_label_candidate in other_labels:
            cost_key = (true_label_candidate, adjusted_prediction)
            misclassification_cost = self.cost_matrix.get(cost_key, 0.0)
            total_expected_cost += prob_per_error_label * misclassification_cost

        is_high_risk = total_expected_cost > self.risk_threshold
        reasoning = f"{adjustment_reason} Expected Cost: {total_expected_cost:.2f}. Risk Threshold: {self.risk_threshold}."

        return Evaluation(
            adjusted_prediction=adjusted_prediction,
            expected_cost=total_expected_cost,
            is_high_risk=is_high_risk,
            reasoning=reasoning.strip(),
        )