import unittest
from unittest.mock import patch
import os
import sys

from src.risk_evaluator import RiskEvaluator, TCHClassification

class TestRiskEvaluator(unittest.TestCase):
    def setUp(self):
        # Mock environment variables to ensure consistent test data
        self.env_patcher = patch.dict(os.environ, {
            "CONSENSUS_MIN_CONFIDENCE": "0.90",
            "COST_RISK_THRESHOLD": "5.0",
            # Standard TCH Cost Matrix values
            "COST_TRUE_RESTRICTED_PRED_CONFIDENTIAL": "40.0",
            "COST_TRUE_RESTRICTED_PRED_PUBLIC": "100.0",
            "COST_TRUE_CONFIDENTIAL_PRED_RESTRICTED": "3.0",
            "COST_TRUE_CONFIDENTIAL_PRED_PUBLIC": "15.0",
            "COST_TRUE_PUBLIC_PRED_RESTRICTED": "0.5",
            "COST_TRUE_PUBLIC_PRED_CONFIDENTIAL": "2.0",
        })
        self.env_patcher.start()
        self.evaluator = RiskEvaluator()

    def tearDown(self):
        self.env_patcher.stop()

    def test_high_confidence_no_adjustment(self):
        """Test that high confidence predictions are not adjusted."""
        eval_result = self.evaluator.calculate_risk(TCHClassification.PUBLIC, 0.95)
        self.assertEqual(eval_result.adjusted_prediction, TCHClassification.PUBLIC)
        self.assertIn("Confidence is high", eval_result.reasoning)

    def test_low_confidence_adjustment_public_to_confidential(self):
        """Test that low confidence PUBLIC is bumped to CONFIDENTIAL."""
        # Confidence 0.80 is < 0.90 threshold
        eval_result = self.evaluator.calculate_risk(TCHClassification.PUBLIC, 0.80)
        self.assertEqual(eval_result.adjusted_prediction, TCHClassification.CONFIDENTIAL)
        self.assertIn("Adjusted 'PUBLIC' to 'CONFIDENTIAL'", eval_result.reasoning)

    def test_low_confidence_adjustment_confidential_to_restricted(self):
        """Test that low confidence CONFIDENTIAL is bumped to RESTRICTED."""
        eval_result = self.evaluator.calculate_risk(TCHClassification.CONFIDENTIAL, 0.85)
        self.assertEqual(eval_result.adjusted_prediction, TCHClassification.RESTRICTED)

    def test_low_confidence_already_restricted(self):
        """Test that RESTRICTED stays RESTRICTED even with low confidence."""
        eval_result = self.evaluator.calculate_risk(TCHClassification.RESTRICTED, 0.50)
        self.assertEqual(eval_result.adjusted_prediction, TCHClassification.RESTRICTED)
        self.assertIn("already at highest sensitivity", eval_result.reasoning)

    def test_cost_calculation_high_risk_scenario(self):
        """
        Scenario: Model predicts PUBLIC with 0.92 confidence.
        Risk: It might be RESTRICTED (Cost 100) or CONFIDENTIAL (Cost 15).
        Prob Error = 0.08.
        Prob per label = 0.04.
        Expected Cost = (0.04 * 100) + (0.04 * 2) = 4.0 + 0.08 = 4.08.
        Threshold is 5.0. Should be Low Risk.
        """
        # Note: The matrix keys are (True, Pred).
        # If Pred=PUBLIC:
        #  - True=RESTRICTED -> Cost 100.0
        #  - True=CONFIDENTIAL -> Cost 2.0 (Wait, check matrix)
        # Matrix: COST_TRUE_PUBLIC_PRED_CONFIDENTIAL=2.0 is (True=Public, Pred=Conf)
        # We need (True=Restricted, Pred=Public) = 100.0
        # We need (True=Confidential, Pred=Public) = 15.0
        
        # Calculation:
        # Pred = PUBLIC. Conf = 0.92. Error = 0.08.
        # Other labels: CONFIDENTIAL, RESTRICTED. (2 labels)
        # Prob per label = 0.04.
        # Cost 1: True=CONFIDENTIAL, Pred=PUBLIC -> 15.0 * 0.04 = 0.6
        # Cost 2: True=RESTRICTED, Pred=PUBLIC -> 100.0 * 0.04 = 4.0
        # Total = 4.6.
        # Threshold = 5.0.
        # Result: 4.6 < 5.0 -> Not High Risk.
        
        eval_result = self.evaluator.calculate_risk(TCHClassification.PUBLIC, 0.92)
        self.assertAlmostEqual(eval_result.expected_cost, 4.6, places=2)
        self.assertFalse(eval_result.is_high_risk)

    def test_cost_calculation_very_high_risk_scenario(self):
        """
        Scenario: Model predicts PUBLIC with 0.85 confidence.
        Adjustment: Bumps to CONFIDENTIAL.
        New Pred: CONFIDENTIAL.
        Error: 0.15.
        Other labels: PUBLIC, RESTRICTED.
        Cost 1: True=PUBLIC, Pred=CONFIDENTIAL -> 2.0 * 0.075 = 0.15
        Cost 2: True=RESTRICTED, Pred=CONFIDENTIAL -> 40.0 * 0.075 = 3.0
        Total = 3.15.
        """
        eval_result = self.evaluator.calculate_risk(TCHClassification.PUBLIC, 0.85)
        self.assertEqual(eval_result.adjusted_prediction, TCHClassification.CONFIDENTIAL)
        self.assertAlmostEqual(eval_result.expected_cost, 3.15, places=2)

    def test_high_risk_human_review_trigger(self):
        """
        Test a scenario that triggers the HUMAN_REVIEW flag.
        Prediction: PUBLIC
        Confidence: 0.90 (Just enough to avoid adjustment, assuming threshold 0.90)
        Risk Calculation:
          Prob Error = 0.10
          Other labels are CONFIDENTIAL and RESTRICTED. Prob per label = 0.05.
          Risk of being CONFIDENTIAL: 0.05 * cost(True=CONFIDENTIAL, Pred=PUBLIC) = 0.05 * 15.0 = 0.75
          Risk of being RESTRICTED: 0.05 * cost(True=RESTRICTED, Pred=PUBLIC) = 0.05 * 100.0 = 5.00
          Total Expected Cost = 0.75 + 5.00 = 5.75
          Threshold = 5.0
          Result: 5.75 > 5.0 -> High Risk
        """
        # Ensure threshold is 0.90 as set in setUp
        eval_result = self.evaluator.calculate_risk(TCHClassification.PUBLIC, 0.90)
        
        self.assertEqual(eval_result.adjusted_prediction, TCHClassification.PUBLIC)
        self.assertAlmostEqual(eval_result.expected_cost, 5.75, places=2)
        self.assertTrue(eval_result.is_high_risk)

if __name__ == "__main__":
    unittest.main()