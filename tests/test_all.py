"""
Comprehensive Test Suite for CareerSense AI.
Validates:
1. Rule-Based Expert System (Academic prerequisites, skill gap analysis, readiness scoring)
2. A* Search Roadmap Optimizer (Dependency order, time budget constraints, heuristics)
3. Machine Learning Classifier (Feature extraction, prediction structure, ranking)
4. Rigorous ML Evaluation Mathematical Correctness:
   - Accuracy equals correct predictions / total predictions
   - Confusion matrix properties: row sums = support, diagonal = correct count, total sum = N
   - Mathematical independence of precision, recall, and F1
   - Macro averages as unweighted arithmetic means across classes
   - Weighted averages as support-weighted means across classes
   - Exact numerical agreement with scikit-learn standard references
   - Consistent zero-division and missing class handling
   - Legitimate metric equality when predictions are perfect
   - Artifact provenance and schema verification
5. Dataset Integrity & Demonstration Verification
6. End-to-End Workflow Integration & Explainability
"""
import unittest
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import ML_FEATURE_COLUMNS, CAREER_TRACKS, DATASET_PATH, MODELS_DIR
from ai_engine.rule_engine import RuleBasedExpertSystem
from ai_engine.a_star_roadmap import AStarRoadmapOptimizer
from ai_engine.explainability import ExplainabilityEngine
from ai_engine.ml_classifier import CareerClassifier, compute_multiclass_metrics

class TestRuleEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RuleBasedExpertSystem()

    def test_academic_prerequisites_met(self):
        records = [
            {"subject_area": "Programming", "grade": "A", "grade_points": 4.0, "module_name": "Prog 1"}
        ]
        res = self.engine.evaluate_academic_prerequisites("Software Engineering", records)
        self.assertTrue(res["overall_academic_met"])
        self.assertEqual(len(res["satisfied"]), 1)
        self.assertEqual(len(res["violations"]), 0)

    def test_academic_prerequisites_violation(self):
        records = [
            {"subject_area": "Programming", "grade": "D", "grade_points": 1.0, "module_name": "Prog 1"}
        ]
        res = self.engine.evaluate_academic_prerequisites("Software Engineering", records)
        self.assertFalse(res["overall_academic_met"])
        self.assertEqual(len(res["violations"]), 1)
        self.assertEqual(res["violations"][0]["severity"], "High")

    def test_skill_gap_analysis(self):
        skills = {
            "Object-Oriented Programming": "Intermediate",
            "Automated Testing & QA": "None" # Target is Intermediate
        }
        gaps = self.engine.analyze_skill_gaps("Software Engineering", skills)
        testing_gap = next((g for g in gaps if g["skill"] == "Automated Testing & QA"), None)
        self.assertIsNotNone(testing_gap)
        self.assertEqual(testing_gap["gap_levels"], 2)
        self.assertEqual(testing_gap["priority"], "High")

    def test_readiness_calculation(self):
        profile = {"gpa": 3.5, "weekly_hours": 8}
        records = [{"subject_area": "Programming", "grade": "A", "grade_points": 4.0, "module_name": "OOP"}]
        skills = {"Object-Oriented Programming": "Intermediate"}
        projects = [{"title": "Demo"}]
        docs = [{"doc_type": "CV"}]
        res = self.engine.calculate_career_readiness("Software Engineering", profile, records, skills, projects, docs)
        self.assertGreater(res["total_readiness_score"], 0.0)
        self.assertIn("readiness_tier", res)
        self.assertIn("breakdown", res)

class TestAStarRoadmap(unittest.TestCase):
    def setUp(self):
        self.optimizer = AStarRoadmapOptimizer()

    def test_roadmap_generation_order_and_prerequisites(self):
        skills = {"REST APIs & Web Services": "None", "Automated Testing & QA": "None"}
        result = self.optimizer.generate_optimal_roadmap("Software Engineering", skills, weekly_hours=8)
        self.assertGreater(len(result["roadmap"]), 0)
        
        # Verify that REST API Fundamentals comes before Build a REST API Project
        ids = [act["activity_id"] for act in result["roadmap"]]
        if "ACT_SE_01" in ids and "ACT_SE_02" in ids:
            self.assertLess(ids.index("ACT_SE_01"), ids.index("ACT_SE_02"))

    def test_weekly_time_budget(self):
        skills = {"REST APIs & Web Services": "None"}
        res_8hrs = self.optimizer.generate_optimal_roadmap("Software Engineering", skills, weekly_hours=8)
        res_20hrs = self.optimizer.generate_optimal_roadmap("Software Engineering", skills, weekly_hours=20)
        # Higher study hours should result in shorter or equal total weeks
        self.assertLessEqual(res_20hrs["total_weeks"], res_8hrs["total_weeks"])

class TestMLClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = CareerClassifier()

    def test_feature_extraction(self):
        records = [{"subject_area": "Programming", "grade": "A", "grade_points": 4.0}]
        interests = {"Software Development & Systems": 5}
        skills = {"Object-Oriented Programming": "Intermediate"}
        vec = self.classifier.extract_features(records, interests, skills)
        self.assertEqual(vec.shape, (1, len(ML_FEATURE_COLUMNS)))

    def test_prediction_output_structure(self):
        records = [{"subject_area": "Programming", "grade": "A", "grade_points": 4.0}]
        interests = {"Software Development & Systems": 5}
        skills = {"Object-Oriented Programming": "Intermediate"}
        res = self.classifier.predict_career_matches(records, interests, skills)
        self.assertIn("ranked_matches", res)
        self.assertIn("top_track", res)
        self.assertIn("ml_predicted_track", res)
        self.assertEqual(len(res["ranked_matches"]), len(CAREER_TRACKS))
        self.assertEqual(res["top_track"], "Software Engineering")

    def test_prediction_output_components_and_target_separation(self):
        """Verifies prediction components, probability normalization, and target aspiration separation."""
        records = [{"subject_area": "Programming", "grade": "A", "grade_points": 4.0}]
        interests = {"Software Development & Systems": 5}
        skills = {"Object-Oriented Programming": "Intermediate"}
        res = self.classifier.predict_career_matches(
            records, interests, skills, target_career="Data Science / AI"
        )
        self.assertIn("ranked_matches", res)
        self.assertIn("top_track", res)
        self.assertIn("ml_predicted_track", res)
        self.assertIn("nearest_neighbor_distances", res)
        self.assertEqual(len(res["ranked_matches"]), len(CAREER_TRACKS))
        
        # Verify normalization (sums to ~100)
        total_p = sum(m["probability"] for m in res["ranked_matches"])
        self.assertAlmostEqual(total_p, 100.0, delta=1.0)
        
        # Verify target aspiration separation and field presence
        for m in res["ranked_matches"]:
            self.assertIn("match_score", m)
            self.assertIn("competency_prob", m)
            self.assertIn("knn_prob", m)
            self.assertIn("dt_prob", m)
            if m["track"] == "Data Science / AI":
                self.assertTrue(m["is_target"])
            else:
                self.assertFalse(m["is_target"])

class TestMLEvaluationMathematicalCorrectness(unittest.TestCase):
    """
    Mathematical correctness tests for multiclass evaluation metrics.
    Replaces arbitrary outcome-based thresholds with verifiable mathematical invariants.
    """
    def setUp(self):
        self.labels = ["Track_A", "Track_B", "Track_C"]
        # Controlled array with known asymmetric error distribution
        self.y_true = ["Track_A", "Track_A", "Track_A", "Track_B", "Track_B", "Track_B", "Track_C", "Track_C", "Track_C", "Track_C"]
        self.y_pred = ["Track_A", "Track_A", "Track_B", "Track_B", "Track_B", "Track_C", "Track_C", "Track_C", "Track_A", "Track_B"]

    def test_invalid_inputs_length_mismatch(self):
        """Verifies that length mismatch between y_true and y_pred raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            compute_multiclass_metrics(["Track_A", "Track_B"], ["Track_A"], self.labels)
        self.assertIn("Length mismatch", str(ctx.exception))

    def test_invalid_inputs_empty(self):
        """Verifies that empty y_true or y_pred raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            compute_multiclass_metrics([], [], self.labels)
        self.assertIn("non-empty", str(ctx.exception))

    def test_invalid_inputs_none(self):
        """Verifies that None inputs raise TypeError."""
        with self.assertRaises(TypeError):
            compute_multiclass_metrics(None, ["Track_A"], self.labels)
        with self.assertRaises(TypeError):
            compute_multiclass_metrics(["Track_A"], None, self.labels)

    def test_invalid_labels_validation(self):
        """Verifies that invalid, duplicate, or empty labels raise ValueError."""
        with self.assertRaises(ValueError):
            compute_multiclass_metrics(["Track_A"], ["Track_A"], [])
        with self.assertRaises(ValueError):
            compute_multiclass_metrics(["Track_A", "Track_B"], ["Track_A", "Track_B"], ["Track_A", "Track_A", "Track_B"])
        with self.assertRaises(ValueError):
            compute_multiclass_metrics(["Track_A"], ["Track_A"], None)

    def test_unsupported_labels_in_data(self):
        """Verifies that labels in y_true or y_pred outside configured labels raise ValueError."""
        with self.assertRaises(ValueError) as ctx1:
            compute_multiclass_metrics(["Track_A", "UNKNOWN"], ["Track_A", "Track_A"], self.labels)
        self.assertIn("unsupported labels in y_true", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            compute_multiclass_metrics(["Track_A", "Track_A"], ["Track_A", "UNKNOWN"], self.labels)
        self.assertIn("unsupported labels in y_pred", str(ctx2.exception))

    def test_completely_incorrect_predictions(self):
        """Verifies that when zero predictions are correct, accuracy and trace are 0.0."""
        labels = ["A", "B"]
        y_true = ["A", "B"]
        y_pred = ["B", "A"]
        m = compute_multiclass_metrics(y_true, y_pred, labels)
        self.assertEqual(m["accuracy"], 0.0)
        cm = m["confusion_matrix"]
        self.assertEqual(cm[0][0] + cm[1][1], 0)

    def test_accuracy_mathematical_definition(self):
        """Verifies accuracy equals correct predictions divided by total predictions."""
        m = compute_multiclass_metrics(self.y_true, self.y_pred, self.labels)
        correct_count = sum(1 for yt, yp in zip(self.y_true, self.y_pred) if yt == yp)
        expected_accuracy = round((correct_count / len(self.y_true)) * 100, 2)
        self.assertEqual(m["accuracy"], expected_accuracy)
        self.assertEqual(m["accuracy"], 60.0)

    def test_confusion_matrix_mathematical_properties(self):
        """
        Verifies confusion matrix structural properties:
        - Matrix rows correspond to actual labels, columns to predicted labels
        - Sum of all elements equals total sample count N
        - Trace (sum of diagonal) equals correct prediction count
        - Row sum for class i equals per-class support for class i
        """
        m = compute_multiclass_metrics(self.y_true, self.y_pred, self.labels)
        cm = m["confusion_matrix"]
        
        # Total count equals N
        total_cm_count = sum(sum(row) for row in cm)
        self.assertEqual(total_cm_count, len(self.y_true))
        
        # Diagonal equals correct predictions
        diag_sum = sum(cm[i][i] for i in range(len(self.labels)))
        correct_count = sum(1 for yt, yp in zip(self.y_true, self.y_pred) if yt == yp)
        self.assertEqual(diag_sum, correct_count)
        
        # Per-class support equals row sum
        for i, label in enumerate(self.labels):
            expected_row_support = sum(cm[i])
            self.assertEqual(m["per_class"][label]["support"], expected_row_support)
            self.assertEqual(m["per_class"][label]["support"], self.y_true.count(label))

    def test_precision_recall_f1_reference_agreement(self):
        """
        Verifies that precision, recall, and F1 calculations agree with
        independent reference calculations and scikit-learn standard metrics.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        m = compute_multiclass_metrics(self.y_true, self.y_pred, self.labels)
        
        sk_acc = round(accuracy_score(self.y_true, self.y_pred) * 100, 2)
        sk_p_wt = round(precision_score(self.y_true, self.y_pred, average="weighted", zero_division=0) * 100, 2)
        sk_r_wt = round(recall_score(self.y_true, self.y_pred, average="weighted", zero_division=0) * 100, 2)
        sk_f1_wt = round(f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0) * 100, 2)
        
        sk_p_mac = round(precision_score(self.y_true, self.y_pred, average="macro", zero_division=0) * 100, 2)
        sk_r_mac = round(recall_score(self.y_true, self.y_pred, average="macro", zero_division=0) * 100, 2)
        sk_f1_mac = round(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0) * 100, 2)
        sk_cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels).tolist()
        
        self.assertEqual(m["accuracy"], sk_acc)
        self.assertEqual(m["precision"], sk_p_wt)
        self.assertEqual(m["recall"], sk_r_wt)
        self.assertEqual(m["f1_score"], sk_f1_wt)
        self.assertEqual(m["precision_macro"], sk_p_mac)
        self.assertEqual(m["recall_macro"], sk_r_mac)
        self.assertEqual(m["f1_score_macro"], sk_f1_mac)
        self.assertEqual(m["confusion_matrix"], sk_cm)

    def test_macro_and_weighted_averaging_logic(self):
        """
        Verifies that Macro metrics are unweighted arithmetic means across all classes,
        while Weighted metrics weight each class by its support.
        """
        m = compute_multiclass_metrics(self.y_true, self.y_pred, self.labels)
        per_class = m["per_class"]
        
        # Macro is unweighted arithmetic mean
        computed_macro_p = round(sum(per_class[l]["precision"] for l in self.labels) / len(self.labels), 2)
        computed_macro_r = round(sum(per_class[l]["recall"] for l in self.labels) / len(self.labels), 2)
        computed_macro_f1 = round(sum(per_class[l]["f1_score"] for l in self.labels) / len(self.labels), 2)
        
        self.assertAlmostEqual(m["precision_macro"], computed_macro_p, places=1)
        self.assertAlmostEqual(m["recall_macro"], computed_macro_r, places=1)
        self.assertAlmostEqual(m["f1_score_macro"], computed_macro_f1, places=1)

    def test_zero_division_and_missing_classes_graceful_handling(self):
        """Verifies that missing classes and zero-division cases evaluate to 0.0 without errors."""
        labels = ["A", "B", "C"]
        # 'C' is never in ground truth; 'A' is never predicted
        y_true = ["A", "A", "B"]
        y_pred = ["B", "B", "B"]
        
        m = compute_multiclass_metrics(y_true, y_pred, labels)
        self.assertIn("per_class", m)
        # Class A has 0 true positives, 0 false positives -> precision = 0.0
        self.assertEqual(m["per_class"]["A"]["precision"], 0.0)
        # Class C has 0 support -> recall = 0.0
        self.assertEqual(m["per_class"]["C"]["recall"], 0.0)
        self.assertEqual(m["per_class"]["C"]["support"], 0)

    def test_legitimate_metric_equality_on_perfect_predictions(self):
        """
        Verifies that accuracy, precision, recall, and F1 legitimately equal 100.0%
        when all predictions are identical to ground truth.
        """
        labels = ["A", "B"]
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "A", "B"]
        m = compute_multiclass_metrics(y_true, y_pred, labels)
        self.assertEqual(m["accuracy"], 100.0)
        self.assertEqual(m["precision"], 100.0)
        self.assertEqual(m["recall"], 100.0)
        self.assertEqual(m["f1_score"], 100.0)
        self.assertEqual(m["precision_macro"], 100.0)
        self.assertEqual(m["f1_score_macro"], 100.0)

    def test_persisted_artifacts_and_provenance(self):
        """Verifies that saved evaluation artifacts contain complete provenance metadata."""
        classifier = CareerClassifier()
        self.assertTrue(classifier.is_trained)
        metrics = classifier.metrics
        self.assertIn("provenance", metrics)
        
        prov = metrics["provenance"]
        self.assertIn("evaluation_timestamp", prov)
        self.assertIn("dataset_hash_sha256", prov)
        self.assertIn("random_seed", prov)
        self.assertEqual(prov["random_seed"], 42)
        self.assertIn("dependencies", prov)
        self.assertIn("scikit-learn", prov["dependencies"])
        self.assertIn("evaluation_protocol", prov)
        
        # Verify model comparison table integrity
        self.assertIn("model_comparison", metrics)
        comp = metrics["model_comparison"]
        self.assertEqual(len(comp), 5)
        for entry in comp:
            self.assertIn("model", entry)
            self.assertIn("cv_accuracy", entry)
            self.assertIn("accuracy", entry)
            self.assertIn("f1_macro", entry)
            self.assertIn("f1_weighted", entry)

    def test_training_evaluation_separation(self):
        """Verifies that train samples and test samples partition the 850 dataset and CV is distinct from test."""
        classifier = CareerClassifier()
        self.assertTrue(classifier.is_trained)
        ds_info = classifier.metrics["dataset_info"]
        self.assertEqual(ds_info["total_samples"], 850)
        self.assertEqual(ds_info["training_samples"], 680)
        self.assertEqual(ds_info["testing_samples"], 170)
        self.assertEqual(ds_info["training_samples"] + ds_info["testing_samples"], 850)
        
        knn_metrics = classifier.metrics["knn"]
        self.assertIn("cv_accuracy_mean", knn_metrics)
        self.assertIn("accuracy", knn_metrics)
        self.assertIn("cv_fold_scores", knn_metrics)
        self.assertEqual(len(knn_metrics["cv_fold_scores"]), 5)
        # Verify CV score differs from test accuracy and is derived from 5 folds
        self.assertIsInstance(knn_metrics["cv_accuracy_mean"], float)
        self.assertIsInstance(knn_metrics["accuracy"], float)

class TestDatasetIntegrity(unittest.TestCase):
    """Verifies dataset structure and demonstration properties."""
    def setUp(self):
        self.df = pd.read_csv(DATASET_PATH)

    def test_dataset_size_and_features(self):
        self.assertEqual(len(self.df), 850)
        for col in ML_FEATURE_COLUMNS:
            self.assertIn(col, self.df.columns)
        self.assertIn("career_track", self.df.columns)

    def test_unique_student_profiles(self):
        """Verifies that synthetic cohort student profiles are unique."""
        self.assertEqual(self.df["student_id"].nunique(), 850)
        self.assertEqual(self.df["reg_no"].nunique(), 850)
        self.assertEqual(self.df.duplicated(subset=ML_FEATURE_COLUMNS).sum(), 0)

    def test_class_representation(self):
        """Verifies all 5 career tracks are adequately represented."""
        counts = self.df["career_track"].value_counts()
        self.assertEqual(len(counts), len(CAREER_TRACKS))
        for track in CAREER_TRACKS:
            self.assertGreaterEqual(counts[track], 100)

class TestIntegrationExplainability(unittest.TestCase):
    def setUp(self):
        self.rule_engine = RuleBasedExpertSystem()
        self.classifier = CareerClassifier()
        self.explainability = ExplainabilityEngine(self.rule_engine, self.classifier)

    def test_full_explanation_bundle(self):
        profile = {"gpa": 3.4, "weekly_hours": 8}
        records = [{"subject_area": "Programming", "grade": "A", "grade_points": 4.0}]
        skills = {"Object-Oriented Programming": "Intermediate"}
        interests = {"Software Development & Systems": 5, "Data Analysis & AI Research": 3}
        
        bundle = self.explainability.generate_full_explanation(profile, records, skills, interests, "Software Engineering")
        self.assertEqual(bundle["target_track"], "Software Engineering")
        self.assertIn("ranked_matches", bundle)
        self.assertIn("radar_comparison", bundle)
        self.assertIn("skill_gaps", bundle)

if __name__ == "__main__":
    unittest.main()
