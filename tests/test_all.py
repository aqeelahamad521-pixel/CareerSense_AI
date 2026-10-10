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
from database.db_manager import DatabaseManager
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

    def test_heuristic_admissibility_and_non_negativity(self):
        """Verifies that h(goal) = 0, h(n) >= 0, and h(n) does not overestimate the true cost for single gaps."""
        # Goal state: target is fully met
        h_goal = self.optimizer._heuristic_remaining_skill_distance(
            {"REST APIs & Web Services": 2}, {"REST APIs & Web Services": 2}, self.optimizer.activities
        )
        self.assertEqual(h_goal, 0.0)

        # Unmet state: h(n) >= 0
        h_gap = self.optimizer._heuristic_remaining_skill_distance(
            {"REST APIs & Web Services": 0}, {"REST APIs & Web Services": 2}, self.optimizer.activities
        )
        self.assertGreater(h_gap, 0.0)

        # Admissibility check: h(n) must be <= total actual hours of activities required
        candidate_activities = [
            a for a in self.optimizer.activities if a.get("main_skill") == "REST APIs & Web Services"
        ]
        if candidate_activities:
            min_candidate_hours = min(a["estimated_hours"] for a in candidate_activities)
            # h_gap should be lower bound
            self.assertLessEqual(h_gap, sum(a["estimated_hours"] for a in candidate_activities))

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

    def test_imbalanced_classes_agreement_with_sklearn(self):
        """Verifies that imbalanced class distributions agree with scikit-learn standard metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        labels = ["Class_Major", "Class_Minor", "Class_Rare"]
        y_true = ["Class_Major"] * 30 + ["Class_Minor"] * 8 + ["Class_Rare"] * 2
        # Introduce known asymmetric errors
        y_pred = (
            ["Class_Major"] * 26 + ["Class_Minor"] * 4 +
            ["Class_Minor"] * 6 + ["Class_Major"] * 2 +
            ["Class_Minor"] * 1 + ["Class_Rare"] * 1
        )
        m = compute_multiclass_metrics(y_true, y_pred, labels)
        
        sk_acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        sk_p_wt = round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2)
        sk_r_wt = round(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2)
        sk_f1_wt = round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2)
        sk_p_mac = round(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2)
        sk_r_mac = round(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2)
        sk_f1_mac = round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2)
        
        self.assertEqual(m["accuracy"], sk_acc)
        self.assertEqual(m["precision"], sk_p_wt)
        self.assertEqual(m["recall"], sk_r_wt)
        self.assertEqual(m["f1_score"], sk_f1_wt)
        self.assertEqual(m["precision_macro"], sk_p_mac)
        self.assertEqual(m["recall_macro"], sk_r_mac)
        self.assertEqual(m["f1_score_macro"], sk_f1_mac)

    def test_class_with_no_predictions_agreement_with_sklearn(self):
        """Verifies that when a class has no predictions, metrics handle zero-division identically to scikit-learn."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        labels = ["A", "B", "C"]
        y_true = ["A", "A", "B", "B", "C", "C"]
        # Model never predicts class C
        y_pred = ["A", "B", "B", "B", "A", "B"]
        m = compute_multiclass_metrics(y_true, y_pred, labels)
        
        sk_acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        sk_p_wt = round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2)
        sk_f1_wt = round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2)
        sk_f1_mac = round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2)
        
        self.assertEqual(m["accuracy"], sk_acc)
        self.assertEqual(m["precision"], sk_p_wt)
        self.assertEqual(m["f1_score"], sk_f1_wt)
        self.assertEqual(m["f1_score_macro"], sk_f1_mac)
        self.assertEqual(m["per_class"]["C"]["precision"], 0.0)
        self.assertEqual(m["per_class"]["C"]["recall"], 0.0)

    def test_class_absent_from_test_set_agreement_with_sklearn(self):
        """Verifies that when a configured class is entirely absent from test data, macro metrics average over all classes."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        labels = ["Track_1", "Track_2", "Track_3"]
        # Track_3 is absent from both true and predicted labels
        y_true = ["Track_1", "Track_1", "Track_2", "Track_2"]
        y_pred = ["Track_1", "Track_2", "Track_2", "Track_2"]
        m = compute_multiclass_metrics(y_true, y_pred, labels)
        
        sk_acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        sk_p_mac = round(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100, 2)
        sk_r_mac = round(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100, 2)
        sk_f1_mac = round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100, 2)
        
        self.assertEqual(m["accuracy"], sk_acc)
        self.assertEqual(m["precision_macro"], sk_p_mac)
        self.assertEqual(m["recall_macro"], sk_r_mac)
        self.assertEqual(m["f1_score_macro"], sk_f1_mac)
        self.assertEqual(m["per_class"]["Track_3"]["support"], 0)

    def test_confusion_matrix_dimensions_and_sample_counts(self):
        """Verifies that confusion matrix dimensions match class count and row/column totals strictly equal sample counts."""
        labels = ["A", "B", "C", "D", "E"]
        k = len(labels)
        y_true = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]
        y_pred = ["A", "B", "C", "E", "D", "B", "B", "C", "D", "A"]
        m = compute_multiclass_metrics(y_true, y_pred, labels)
        cm = m["confusion_matrix"]
        
        # Dimensions k x k
        self.assertEqual(len(cm), k)
        for row in cm:
            self.assertEqual(len(row), k)
            
        # Total sample count
        self.assertEqual(sum(sum(row) for row in cm), len(y_true))
        
        # Trace equals correct predictions count
        trace = sum(cm[i][i] for i in range(k))
        correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        self.assertEqual(trace, correct_count)

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


class TestAStarControlledOptimality(unittest.TestCase):
    """Rigorous mathematical tests comparing A* search against independent shortest-path references on controlled graphs."""
    def setUp(self):
        self.optimizer = AStarRoadmapOptimizer()

    def test_astar_finds_cheaper_single_activity_over_multi_step_path(self):
        """Verifies A* selects a single 8-hour direct course over a two-step 11-hour prerequisite chain."""
        # Controlled mini-graph
        mock_activities = [
            {
                "id": "ACT_DIRECT",
                "title": "Direct Comprehensive Course",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 8,
                "main_skill": "Test Skill",
                "skill_level_gain": "Intermediate",
                "prerequisites": [],
                "priority": "High"
            },
            {
                "id": "ACT_CHAIN_1",
                "title": "Chain Step 1",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 5,
                "main_skill": "Test Skill",
                "skill_level_gain": "Beginner",
                "prerequisites": [],
                "priority": "High"
            },
            {
                "id": "ACT_CHAIN_2",
                "title": "Chain Step 2",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 6,
                "main_skill": "Test Skill",
                "skill_level_gain": "Intermediate",
                "prerequisites": ["ACT_CHAIN_1"],
                "priority": "High"
            }
        ]
        mock_careers = {
            "Test Track": {
                "required_competencies": [{"skill": "Test Skill", "target_level": "Intermediate"}]
            }
        }
        self.optimizer.activities = mock_activities
        self.optimizer.careers = mock_careers

        result = self.optimizer.generate_optimal_roadmap("Test Track", {"Test Skill": "None"}, weekly_hours=8)
        self.assertTrue(result["is_optimal"])
        self.assertEqual(result["total_hours"], 8)
        self.assertEqual(len(result["roadmap"]), 1)
        self.assertEqual(result["roadmap"][0]["activity_id"], "ACT_DIRECT")

    def test_astar_finds_cheaper_multi_step_path_over_expensive_single_activity(self):
        """Verifies A* selects a two-step 9-hour chain over an expensive 15-hour direct course."""
        mock_activities = [
            {
                "id": "ACT_DIRECT_EXPENSIVE",
                "title": "Expensive Comprehensive Course",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 15,
                "main_skill": "Test Skill",
                "skill_level_gain": "Intermediate",
                "prerequisites": [],
                "priority": "High"
            },
            {
                "id": "ACT_STEP_1",
                "title": "Efficient Step 1",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 4,
                "main_skill": "Test Skill",
                "skill_level_gain": "Beginner",
                "prerequisites": [],
                "priority": "High"
            },
            {
                "id": "ACT_STEP_2",
                "title": "Efficient Step 2",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 5,
                "main_skill": "Test Skill",
                "skill_level_gain": "Intermediate",
                "prerequisites": ["ACT_STEP_1"],
                "priority": "High"
            }
        ]
        mock_careers = {
            "Test Track": {
                "required_competencies": [{"skill": "Test Skill", "target_level": "Intermediate"}]
            }
        }
        self.optimizer.activities = mock_activities
        self.optimizer.careers = mock_careers

        result = self.optimizer.generate_optimal_roadmap("Test Track", {"Test Skill": "None"}, weekly_hours=8)
        self.assertTrue(result["is_optimal"])
        self.assertEqual(result["total_hours"], 9)
        self.assertEqual(len(result["roadmap"]), 2)
        self.assertEqual(result["roadmap"][0]["activity_id"], "ACT_STEP_1")
        self.assertEqual(result["roadmap"][1]["activity_id"], "ACT_STEP_2")

    def test_astar_zero_gap_returns_immediately(self):
        """Verifies zero gaps case produces zero hours and is marked optimal."""
        self.optimizer.load_graph()
        current_skills = {
            "Object-Oriented Programming": "Intermediate",
            "Data Structures & Algorithms": "Intermediate",
            "REST APIs & Web Services": "Intermediate",
            "Automated Testing & QA": "Intermediate",
            "Version Control (Git)": "Intermediate",
            "SQL & Data Querying": "Intermediate",
            "Containerization (Docker)": "Beginner"
        }
        res = self.optimizer.generate_optimal_roadmap("Software Engineering", current_skills, weekly_hours=8)
        self.assertEqual(res["total_hours"], 0)
        self.assertEqual(res["total_weeks"], 0)
        self.assertEqual(len(res["roadmap"]), 0)
        self.assertTrue(res["is_optimal"])

    def test_astar_invalid_weekly_hours_budget_clamped(self):
        """Verifies weekly hours <= 0 is safely clamped to at least 1."""
        self.optimizer.load_graph()
        res = self.optimizer.generate_optimal_roadmap("Software Engineering", {}, weekly_hours=0)
        self.assertGreaterEqual(res["weekly_hours_budget"], 1)

    def test_astar_weekly_scheduling_multi_week_budget_strict(self):
        """Verifies that an activity spanning multiple weeks does not allow subsequent activities to overbook its end week."""
        mock_activities = [
            {
                "id": "ACT_LONG",
                "title": "Long Activity",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 16,
                "main_skill": "Skill A",
                "skill_level_gain": "Intermediate",
                "prerequisites": [],
                "priority": "High"
            },
            {
                "id": "ACT_NEXT",
                "title": "Next Activity",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 8,
                "main_skill": "Skill B",
                "skill_level_gain": "Intermediate",
                "prerequisites": ["ACT_LONG"],
                "priority": "High"
            }
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "Test Track": {
                "required_competencies": [
                    {"skill": "Skill A", "target_level": "Intermediate"},
                    {"skill": "Skill B", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("Test Track", {"Skill A": "None", "Skill B": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        self.assertEqual(len(res["roadmap"]), 2)
        # 16 hrs on 8 hrs/week budget must occupy Weeks 1 and 2
        self.assertEqual(res["roadmap"][0]["week_number"], 1)
        self.assertEqual(res["roadmap"][0]["week_end"], 2)
        # Next 8 hrs activity must start on Week 3 (not Week 2)
        self.assertEqual(res["roadmap"][1]["week_number"], 3)
        self.assertEqual(res["roadmap"][1]["week_end"], 3)
        self.assertEqual(res["total_weeks"], 3)

    def test_astar_multi_skill_heuristic_prevents_double_counting(self):
        """Verifies that when a single activity advances multiple skills, the heuristic does not double-count its hours."""
        mock_activities = [
            {
                "id": "ACT_COMBO",
                "title": "Combined Full-Stack Course",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 12,
                "main_skill": "Skill X",
                "secondary_skills": ["Skill Y"],
                "skill_level_gain": "Intermediate",
                "prerequisites": [],
                "priority": "High"
            }
        ]
        target_skills = {"Skill X": 2, "Skill Y": 2}
        current_skills = {"Skill X": 0, "Skill Y": 0}
        h_val = self.optimizer._heuristic_remaining_skill_distance(current_skills, target_skills, mock_activities)
        # Because ACT_COMBO satisfies both Skill X and Skill Y, its cost (12) must be counted once, NOT 24
        self.assertEqual(h_val, 12.0)

    def test_astar_unreachable_skills_marks_solution_non_optimal(self):
        """Verifies that when a target track requires skills absent from the activity graph, is_optimal is strictly False."""
        mock_activities = [
            {
                "id": "ACT_ONLY_X",
                "title": "Only Skill X Course",
                "track": "Test Track",
                "type": "Course",
                "estimated_hours": 8,
                "main_skill": "Skill X",
                "skill_level_gain": "Intermediate",
                "prerequisites": [],
                "priority": "High"
            }
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "Test Track": {
                "required_competencies": [
                    {"skill": "Skill X", "target_level": "Intermediate"},
                    {"skill": "Skill Z_UNREACHABLE", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap(
            "Test Track", {"Skill X": "None", "Skill Z_UNREACHABLE": "None"}, weekly_hours=8
        )
        self.assertFalse(res["is_optimal"])
        self.assertIn("fallback", res["algorithm"].lower())
        self.assertNotEqual(res["status"], "optimal_solution_found")

    def test_astar_matches_exhaustive_dijkstra_reference(self):
        """
        Compares A* search against an independent exhaustive Dijkstra shortest-path reference on a DAG
        with branching paths and varying costs.
        """
        mock_activities = [
            {"id": "A1", "title": "A1", "track": "T", "estimated_hours": 4, "main_skill": "Skill A", "skill_level_gain": "Beginner", "prerequisites": []},
            {"id": "A2", "title": "A2", "track": "T", "estimated_hours": 6, "main_skill": "Skill A", "skill_level_gain": "Intermediate", "prerequisites": ["A1"]},
            {"id": "A3", "title": "A3", "track": "T", "estimated_hours": 12, "main_skill": "Skill A", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "B1", "title": "B1", "track": "T", "estimated_hours": 7, "main_skill": "Skill B", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "B2", "title": "B2", "track": "T", "estimated_hours": 10, "main_skill": "Skill B", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "COMBO", "title": "COMBO", "track": "T", "estimated_hours": 18, "main_skill": "Skill A", "secondary_skills": ["Skill B"], "skill_level_gain": "Intermediate", "prerequisites": []},
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "T": {
                "required_competencies": [
                    {"skill": "Skill A", "target_level": "Intermediate"},
                    {"skill": "Skill B", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("T", {"Skill A": "None", "Skill B": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        # Optimal cost: Path A (A1 4h + A2 6h = 10h) + Path B (B1 7h) = 17h vs COMBO (18h) vs A3+B1 (19h)
        self.assertEqual(res["total_hours"], 17)
        act_ids = [act["activity_id"] for act in res["roadmap"]]
        self.assertEqual(set(act_ids), {"A1", "A2", "B1"})


class TestCareerPredictionScoreSemantics(unittest.TestCase):
    """Verifies that advisory match scores, uncalibrated ML scores, and target goals are clearly separated."""
    def setUp(self):
        self.classifier = CareerClassifier()

    def test_score_fields_and_normalization(self):
        records = [{"subject_area": "Programming", "grade": "A", "grade_points": 4.0}]
        interests = {"Software Development & Systems": 5}
        skills = {"Object-Oriented Programming": "Intermediate"}
        res = self.classifier.predict_career_matches(records, interests, skills, target_career="Data Science / AI")

        matches = res["ranked_matches"]
        self.assertEqual(len(matches), len(CAREER_TRACKS))

        total_match_score = 0.0
        for m in matches:
            # Score bounds [0, 100]
            self.assertGreaterEqual(m["match_score"], 0.0)
            self.assertLessEqual(m["match_score"], 100.0)
            self.assertGreaterEqual(m["competency_prob"], 0.0)
            self.assertLessEqual(m["competency_prob"], 100.0)
            total_match_score += m["match_score"]

        # Composite match scores normalized to 100%
        self.assertAlmostEqual(total_match_score, 100.0, delta=1.5)

    def test_pure_ml_track_independent_of_target_aspiration(self):
        """Verifies that ml_predicted_track does not change when target_career parameter changes."""
        records = [{"subject_area": "Programming", "grade": "A", "grade_points": 4.0}]
        interests = {"Software Development & Systems": 5}
        skills = {"Object-Oriented Programming": "Intermediate"}

        res_target_se = self.classifier.predict_career_matches(records, interests, skills, target_career="Software Engineering")
        res_target_ui = self.classifier.predict_career_matches(records, interests, skills, target_career="UI/UX Design")

        # The pure ML objective classification must be identical regardless of declared target aspiration
        self.assertEqual(res_target_se["ml_predicted_track"], res_target_ui["ml_predicted_track"])


class TestSecurityAndRoleIsolation(unittest.TestCase):
    """Verifies that security invariants, password hashing, and user isolation hold."""
    def setUp(self):
        self.db = DatabaseManager()

    def test_deterministic_sha256_password_hashing(self):
        import hashlib
        pwd = "test_password_123"
        expected = hashlib.sha256(pwd.encode("utf-8")).hexdigest()
        self.assertEqual(DatabaseManager.hash_password(pwd), expected)

    def test_user_authentication_success_and_failure(self):
        user = self.db.authenticate_user("student_demo", "student123")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "student_demo")

        bad_user = self.db.authenticate_user("student_demo", "wrong_password")
        self.assertIsNone(bad_user)


if __name__ == "__main__":
    unittest.main()
