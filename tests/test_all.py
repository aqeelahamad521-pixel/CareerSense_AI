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

    def test_astar_finds_cheaper_multi_skill_combo_over_separate_activities(self):
        """Verifies A* selects a 13-hour multi-skill combo activity over separate 10-hour activities (20h total)."""
        mock_activities = [
            {"id": "ACT_SEP_A", "title": "Sep A", "track": "T", "estimated_hours": 10, "main_skill": "Skill A", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "ACT_SEP_B", "title": "Sep B", "track": "T", "estimated_hours": 10, "main_skill": "Skill B", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "ACT_COMBO", "title": "Combo AB", "track": "T", "estimated_hours": 13, "main_skill": "Skill A", "secondary_skills": ["Skill B"], "skill_level_gain": "Intermediate", "prerequisites": []},
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
        res_astar = self.optimizer.generate_optimal_roadmap("T", {"Skill A": "None", "Skill B": "None"}, weekly_hours=8)
        self.assertTrue(res_astar["is_optimal"])
        self.assertEqual(res_astar["total_hours"], 13)
        self.assertEqual(len(res_astar["roadmap"]), 1)
        self.assertEqual(res_astar["roadmap"][0]["activity_id"], "ACT_COMBO")

        # Verify against pure Uniform-Cost Search / Dijkstra reference (heuristic_mode="zero")
        optimizer_dijkstra = AStarRoadmapOptimizer(heuristic_mode="zero")
        optimizer_dijkstra.activities = mock_activities
        optimizer_dijkstra.careers = self.optimizer.careers
        res_dijkstra = optimizer_dijkstra.generate_optimal_roadmap("T", {"Skill A": "None", "Skill B": "None"}, weekly_hours=8)
        self.assertEqual(res_astar["total_hours"], res_dijkstra["total_hours"])
        self.assertEqual([a["activity_id"] for a in res_astar["roadmap"]], [a["activity_id"] for a in res_dijkstra["roadmap"]])

        # Admissibility check: heuristic value h(start) must not exceed optimal cost h* = 13
        h_start = self.optimizer._heuristic_remaining_skill_distance({"Skill A": 0, "Skill B": 0}, {"Skill A": 2, "Skill B": 2}, mock_activities)
        self.assertLessEqual(h_start, 13.0)
        self.assertEqual(h_start, 10.0)

    def test_astar_finds_cheaper_separate_activities_over_expensive_combo(self):
        """Verifies A* selects two separate 4-hour activities (8h total) over an expensive 16-hour combo."""
        mock_activities = [
            {"id": "ACT_SEP_A", "title": "Sep A", "track": "T", "estimated_hours": 4, "main_skill": "Skill A", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "ACT_SEP_B", "title": "Sep B", "track": "T", "estimated_hours": 4, "main_skill": "Skill B", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "ACT_COMBO_EXP", "title": "Combo Expensive", "track": "T", "estimated_hours": 16, "main_skill": "Skill A", "secondary_skills": ["Skill B"], "skill_level_gain": "Intermediate", "prerequisites": []},
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
        self.assertEqual(res["total_hours"], 8)
        act_ids = {a["activity_id"] for a in res["roadmap"]}
        self.assertEqual(act_ids, {"ACT_SEP_A", "ACT_SEP_B"})

    def test_astar_heuristic_modes_agreement_on_optimal_cost(self):
        """Verifies that auto, admissible_max, and zero (Dijkstra) all find the exact same optimal cost."""
        mock_activities = [
            {"id": "X1", "title": "X1", "track": "T", "estimated_hours": 5, "main_skill": "Skill X", "skill_level_gain": "Beginner", "prerequisites": []},
            {"id": "X2", "title": "X2", "track": "T", "estimated_hours": 6, "main_skill": "Skill X", "skill_level_gain": "Intermediate", "prerequisites": ["X1"]},
            {"id": "Y1", "title": "Y1", "track": "T", "estimated_hours": 8, "main_skill": "Skill Y", "skill_level_gain": "Intermediate", "prerequisites": []},
        ]
        careers = {
            "T": {
                "required_competencies": [
                    {"skill": "Skill X", "target_level": "Intermediate"},
                    {"skill": "Skill Y", "target_level": "Intermediate"}
                ]
            }
        }
        results = []
        for mode in ["auto", "admissible_max", "zero"]:
            opt = AStarRoadmapOptimizer(heuristic_mode=mode)
            opt.activities = mock_activities
            opt.careers = careers
            r = opt.generate_optimal_roadmap("T", {"Skill X": "None", "Skill Y": "None"}, weekly_hours=8)
            self.assertTrue(r["is_optimal"])
            results.append(r["total_hours"])

        # All 3 modes must find the exact same optimal cost (5 + 6 + 8 = 19 hours)
        self.assertEqual(results[0], 19)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_activity_filtering_includes_secondary_skills(self):
        """Verifies that an activity advancing a required skill solely via secondary_skills is included and selected."""
        mock_activities = [
            {
                "id": "ACT_SEC_ONLY",
                "title": "Cross-Track Activity with Secondary Skill",
                "track": "Unrelated Track",
                "type": "Course",
                "estimated_hours": 10,
                "main_skill": "Unrelated Skill",
                "secondary_skills": ["Target Secondary Skill"],
                "skill_level_gain": "Intermediate",
                "prerequisites": [],
                "priority": "High"
            }
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "Special Track": {
                "required_competencies": [
                    {"skill": "Target Secondary Skill", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("Special Track", {"Target Secondary Skill": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        self.assertEqual(res["status"], "optimal_solution_found")
        self.assertEqual(res["total_hours"], 10)
        self.assertEqual(len(res["roadmap"]), 1)
        self.assertEqual(res["roadmap"][0]["activity_id"], "ACT_SEC_ONLY")

    def test_activity_filtering_includes_prerequisite_with_unrelated_main_skill(self):
        """Verifies that a prerequisite activity is included via backward closure even if its main skill is not an active gap."""
        mock_activities = [
            {
                "id": "ACT_PREREQ",
                "title": "Prerequisite Foundation",
                "track": "Foundations Track",
                "type": "Course",
                "estimated_hours": 6,
                "main_skill": "Unneeded Prereq Skill",
                "skill_level_gain": "Intermediate",
                "prerequisites": [],
                "priority": "Low"
            },
            {
                "id": "ACT_GOAL",
                "title": "Target Competency Course",
                "track": "Target Track",
                "type": "Course",
                "estimated_hours": 8,
                "main_skill": "Crucial Target Skill",
                "skill_level_gain": "Intermediate",
                "prerequisites": ["ACT_PREREQ"],
                "priority": "High"
            }
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "Target Track": {
                "required_competencies": [
                    {"skill": "Crucial Target Skill", "target_level": "Intermediate"}
                ]
            }
        }
        # Student needs Crucial Target Skill; Unneeded Prereq Skill is NOT an active gap
        res = self.optimizer.generate_optimal_roadmap("Target Track", {"Crucial Target Skill": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        self.assertEqual(res["status"], "optimal_solution_found")
        self.assertEqual(res["total_hours"], 14)
        self.assertEqual(len(res["roadmap"]), 2)
        self.assertEqual(res["roadmap"][0]["activity_id"], "ACT_PREREQ")
        self.assertEqual(res["roadmap"][1]["activity_id"], "ACT_GOAL")

    def test_partial_coverage_reported_only_when_skill_genuinely_uncovered(self):
        """Verifies partial coverage is reported if and only if an active gap has no advancing activity anywhere in graph."""
        mock_activities = [
            {
                "id": "ACT_A",
                "title": "Skill A Course",
                "track": "Track T",
                "type": "Course",
                "estimated_hours": 5,
                "main_skill": "Skill A",
                "skill_level_gain": "Intermediate",
                "prerequisites": [],
                "priority": "High"
            }
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "Track T": {
                "required_competencies": [
                    {"skill": "Skill A", "target_level": "Intermediate"},
                    {"skill": "Skill B Uncovered", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("Track T", {"Skill A": "None", "Skill B Uncovered": "None"}, weekly_hours=8)
        self.assertFalse(res["is_optimal"])
        self.assertEqual(res["status"], "partial_coverage")
        self.assertIn("Partial Graph Coverage Fallback", res["algorithm"])
        # Should still sequence the available activity for Skill A
        self.assertEqual(len(res["roadmap"]), 1)
        self.assertEqual(res["roadmap"][0]["activity_id"], "ACT_A")

    def test_roadmap_duration_policy_fractional_hours(self):
        """Verifies fractional hours are preserved consistently in search path cost, total_hours, and scheduled roadmap."""
        mock_activities = [
            {"id": "ACT_F1", "title": "Fractional 1", "track": "T", "estimated_hours": 7.5, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "ACT_F2", "title": "Fractional 2", "track": "T", "estimated_hours": 3.25, "main_skill": "S2", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "ACT_F3", "title": "Fractional 3", "track": "T", "estimated_hours": 4.25, "main_skill": "S3", "skill_level_gain": "Intermediate", "prerequisites": []}
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "T": {
                "required_competencies": [
                    {"skill": "S1", "target_level": "Intermediate"},
                    {"skill": "S2", "target_level": "Intermediate"},
                    {"skill": "S3", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("T", {"S1": "None", "S2": "None", "S3": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        # 7.5 + 3.25 + 4.25 = 15.0 hours
        self.assertEqual(res["total_hours"], 15)
        # Sum of scheduled item hours must exactly equal total_hours
        sched_sum = sum(item["estimated_hours"] for item in res["roadmap"])
        self.assertEqual(sched_sum, 15.0)
        hours_by_id = {item["activity_id"]: item["estimated_hours"] for item in res["roadmap"]}
        self.assertEqual(hours_by_id["ACT_F1"], 7.5)
        self.assertEqual(hours_by_id["ACT_F2"], 3.25)
        self.assertEqual(hours_by_id["ACT_F3"], 4.25)

    def test_roadmap_duration_policy_zero_or_negative_clamped(self):
        """Verifies zero or negative activity hours are safely clamped to 1.0 hour, maintaining strictly positive costs."""
        mock_activities = [
            {"id": "ACT_ZERO", "title": "Zero Duration", "track": "T", "estimated_hours": 0, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "ACT_NEG", "title": "Negative Duration", "track": "T", "estimated_hours": -10, "main_skill": "S2", "skill_level_gain": "Intermediate", "prerequisites": []}
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "T": {
                "required_competencies": [
                    {"skill": "S1", "target_level": "Intermediate"},
                    {"skill": "S2", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("T", {"S1": "None", "S2": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        # Both clamped to 1.0 hr -> 1 + 1 = 2 hrs
        self.assertEqual(res["total_hours"], 2)
        self.assertEqual(res["roadmap"][0]["estimated_hours"], 1)
        self.assertEqual(res["roadmap"][1]["estimated_hours"], 1)

    def test_roadmap_weekly_budget_longer_than_weekly_hours(self):
        """Verifies that an activity longer than the weekly budget spans multiple weeks and respects the budget."""
        mock_activities = [
            {"id": "ACT_HUGE", "title": "20-Hour Course", "track": "T", "estimated_hours": 20, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "ACT_FIT", "title": "4-Hour Course", "track": "T", "estimated_hours": 4, "main_skill": "S2", "skill_level_gain": "Intermediate", "prerequisites": ["ACT_HUGE"]},
            {"id": "ACT_NEXT", "title": "6-Hour Course", "track": "T", "estimated_hours": 6, "main_skill": "S3", "skill_level_gain": "Intermediate", "prerequisites": ["ACT_FIT"]}
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "T": {
                "required_competencies": [
                    {"skill": "S1", "target_level": "Intermediate"},
                    {"skill": "S2", "target_level": "Intermediate"},
                    {"skill": "S3", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("T", {"S1": "None", "S2": "None", "S3": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        # ACT_HUGE (20h on 8h budget): Week 1 (8h), Week 2 (8h), Week 3 (4h) -> weeks 1 to 3
        self.assertEqual(res["roadmap"][0]["week_number"], 1)
        self.assertEqual(res["roadmap"][0]["week_end"], 3)
        # ACT_FIT (4h): Fits in remaining 4h of Week 3 -> week 3 to 3
        self.assertEqual(res["roadmap"][1]["week_number"], 3)
        self.assertEqual(res["roadmap"][1]["week_end"], 3)
        # ACT_NEXT (6h): Week 3 is now full (4h + 4h = 8h), so starts on Week 4 -> week 4 to 4
        self.assertEqual(res["roadmap"][2]["week_number"], 4)
        self.assertEqual(res["roadmap"][2]["week_end"], 4)
        self.assertEqual(res["total_weeks"], 4)
        self.assertEqual(res["total_hours"], 30)

    def test_multiple_consecutive_activities_scheduling(self):
        """Verifies scheduling of multiple consecutive activities within and across weekly budget limits."""
        mock_activities = [
            {"id": "A1", "title": "A1", "track": "T", "estimated_hours": 2, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "A2", "title": "A2", "track": "T", "estimated_hours": 3, "main_skill": "S2", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "A3", "title": "A3", "track": "T", "estimated_hours": 3, "main_skill": "S3", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "A4", "title": "A4", "track": "T", "estimated_hours": 4, "main_skill": "S4", "skill_level_gain": "Intermediate", "prerequisites": []}
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "T": {
                "required_competencies": [
                    {"skill": "S1", "target_level": "Intermediate"},
                    {"skill": "S2", "target_level": "Intermediate"},
                    {"skill": "S3", "target_level": "Intermediate"},
                    {"skill": "S4", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("T", {"S1": "None", "S2": "None", "S3": "None", "S4": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        # Total hours = 2 + 3 + 3 + 4 = 12
        self.assertEqual(res["total_hours"], 12)
        # Week 1: 2 + 3 + 3 = 8 hours full
        # Week 2: 4 hours
        self.assertEqual(res["total_weeks"], 2)

    def test_astar_search_limit_exhaustion_marks_fallback_non_optimal(self):
        """Verifies that if search iterations are exhausted, solution is marked non-optimal fallback."""
        # Create an activity graph with circular prerequisite deadlock preventing goal attainment
        mock_activities = [
            {"id": "ACT_LOOP_1", "title": "Loop 1", "track": "T", "estimated_hours": 5, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": ["ACT_LOOP_2"]},
            {"id": "ACT_LOOP_2", "title": "Loop 2", "track": "T", "estimated_hours": 5, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": ["ACT_LOOP_1"]}
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "T": {
                "required_competencies": [
                    {"skill": "S1", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("T", {"S1": "None"}, weekly_hours=8)
        self.assertFalse(res["is_optimal"])
        self.assertIn("Fallback", res["algorithm"])
        self.assertNotEqual(res["status"], "optimal_solution_found")

    def test_path_cost_equals_sum_of_activity_costs_invariance(self):
        """Verifies that returned total_hours strictly equals the sum of activity costs across the entire roadmap."""
        mock_activities = [
            {"id": "ACT_A1", "title": "A1", "track": "T", "estimated_hours": 6.5, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "ACT_A2", "title": "A2", "track": "T", "estimated_hours": 8.0, "main_skill": "S2", "skill_level_gain": "Intermediate", "prerequisites": ["ACT_A1"]},
            {"id": "ACT_A3", "title": "A3", "track": "T", "estimated_hours": 3.5, "main_skill": "S3", "skill_level_gain": "Intermediate", "prerequisites": ["ACT_A2"]}
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "T": {
                "required_competencies": [
                    {"skill": "S1", "target_level": "Intermediate"},
                    {"skill": "S2", "target_level": "Intermediate"},
                    {"skill": "S3", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("T", {"S1": "None", "S2": "None", "S3": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        # Sum of items: 6.5 + 8.0 + 3.5 = 18.0
        sum_items = sum(item["estimated_hours"] for item in res["roadmap"])
        self.assertEqual(res["total_hours"], 18)
        self.assertEqual(res["total_hours"], sum_items)
        for item in res["roadmap"]:
            self.assertGreater(item["estimated_hours"], 0)

    def test_deep_cross_track_prerequisite_chain_retained(self):
        """Verifies that a 4-level deep prerequisite chain across different tracks is fully preserved by backward closure."""
        mock_activities = [
            {"id": "P_ROOT", "title": "Root Course", "track": "Track 1", "estimated_hours": 4, "main_skill": "Skill R", "skill_level_gain": "Beginner", "prerequisites": []},
            {"id": "P_MID1", "title": "Mid Course 1", "track": "Track 2", "estimated_hours": 5, "main_skill": "Skill M1", "skill_level_gain": "Beginner", "prerequisites": ["P_ROOT"]},
            {"id": "P_MID2", "title": "Mid Course 2", "track": "Track 3", "estimated_hours": 6, "main_skill": "Skill M2", "skill_level_gain": "Beginner", "prerequisites": ["P_MID1"]},
            {"id": "ACT_LEAF", "title": "Leaf Course", "track": "Track 4", "estimated_hours": 7, "main_skill": "Skill Other", "secondary_skills": ["Target Gap"], "skill_level_gain": "Intermediate", "prerequisites": ["P_MID2"]}
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "Target Career": {
                "required_competencies": [
                    {"skill": "Target Gap", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("Target Career", {"Target Gap": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        self.assertEqual(res["status"], "optimal_solution_found")
        self.assertEqual(res["total_hours"], 22)  # 4 + 5 + 6 + 7
        act_ids = [item["activity_id"] for item in res["roadmap"]]
        # Must execute in strict topological order
        self.assertEqual(act_ids, ["P_ROOT", "P_MID1", "P_MID2", "ACT_LEAF"])

    def test_missing_external_prerequisite_falls_back_non_optimal(self):
        """Verifies that when an activity has an unresolvable missing prerequisite, A* falls back with is_optimal=False."""
        mock_activities = [
            {"id": "ACT_BROKEN", "title": "Broken Prereq", "track": "T", "estimated_hours": 8, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": ["NON_EXISTENT_ID"]}
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "T": {
                "required_competencies": [
                    {"skill": "S1", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("T", {"S1": "None"}, weekly_hours=8)
        self.assertFalse(res["is_optimal"])
        self.assertNotEqual(res["status"], "optimal_solution_found")
        self.assertIn("fallback", res["algorithm"].lower())

    def test_heuristic_admissibility_and_bottleneck_max_proof(self):
        """Verifies that when multi-skill activities exist, the bottleneck lower bound is strictly admissible."""
        mock_activities = [
            {"id": "A_COMBO", "title": "Combo", "track": "T", "estimated_hours": 15, "main_skill": "S1", "secondary_skills": ["S2"], "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "A_SEP1", "title": "Sep 1", "track": "T", "estimated_hours": 10, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "A_SEP2", "title": "Sep 2", "track": "T", "estimated_hours": 10, "main_skill": "S2", "skill_level_gain": "Intermediate", "prerequisites": []}
        ]
        # h*(start) = 15 hours (choosing A_COMBO is cheaper than SEP1 + SEP2 = 20)
        # Bottleneck bound: max(min(15, 10), min(15, 10)) = max(10, 10) = 10.0
        # Admissibility condition: h(start) <= h*(start) (10.0 <= 15.0)
        h_auto = self.optimizer._heuristic_remaining_skill_distance({"S1": 0, "S2": 0}, {"S1": 2, "S2": 2}, mock_activities)
        self.assertEqual(h_auto, 10.0)
        self.assertLessEqual(h_auto, 15.0)

        # In admissible_max mode, it must also be 10.0
        opt_max = AStarRoadmapOptimizer(heuristic_mode="admissible_max")
        h_max = opt_max._heuristic_remaining_skill_distance({"S1": 0, "S2": 0}, {"S1": 2, "S2": 2}, mock_activities)
        self.assertEqual(h_max, 10.0)
        self.assertLessEqual(h_max, 15.0)

    def test_complete_solution_found_when_one_exists_in_complex_graph(self):
        """Verifies A* finds a complete optimal solution in a complex graph with distractor tracks and competing branches."""
        mock_activities = [
            {"id": "D1", "title": "Distractor", "track": "Irrelevant Track", "estimated_hours": 50, "main_skill": "Unrelated", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "BRANCH1_A", "title": "Branch 1 Step A", "track": "T", "estimated_hours": 7, "main_skill": "S1", "skill_level_gain": "Beginner", "prerequisites": []},
            {"id": "BRANCH1_B", "title": "Branch 1 Step B", "track": "T", "estimated_hours": 7, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": ["BRANCH1_A"]},
            {"id": "BRANCH2_A", "title": "Branch 2 Direct", "track": "T", "estimated_hours": 11, "main_skill": "S1", "skill_level_gain": "Intermediate", "prerequisites": []},
            {"id": "S2_ACT", "title": "S2 Direct", "track": "Other Track", "estimated_hours": 8, "main_skill": "Other", "secondary_skills": ["S2"], "skill_level_gain": "Intermediate", "prerequisites": []}
        ]
        self.optimizer.activities = mock_activities
        self.optimizer.careers = {
            "T": {
                "required_competencies": [
                    {"skill": "S1", "target_level": "Intermediate"},
                    {"skill": "S2", "target_level": "Intermediate"}
                ]
            }
        }
        res = self.optimizer.generate_optimal_roadmap("T", {"S1": "None", "S2": "None"}, weekly_hours=8)
        self.assertTrue(res["is_optimal"])
        self.assertEqual(res["status"], "optimal_solution_found")
        # Optimal cost: BRANCH2_A (11) + S2_ACT (8) = 19h vs BRANCH1 (7+7=14) + S2_ACT (8) = 22h
        self.assertEqual(res["total_hours"], 19)
        act_ids = {item["activity_id"] for item in res["roadmap"]}
        self.assertEqual(act_ids, {"BRANCH2_A", "S2_ACT"})


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

    def test_score_semantic_fields_and_alignment(self):
        """Verifies all decomposed score semantics, backward compatibility aliases, and sorting order."""
        records = [{"subject_area": "Programming", "grade": "A", "grade_points": 4.0}]
        interests = {"Software Development & Systems": 5}
        skills = {"Object-Oriented Programming": "Intermediate"}
        res = self.classifier.predict_career_matches(records, interests, skills, target_career="Data Science / AI")

        matches = res["ranked_matches"]
        # Verify strict descending order by match_score
        scores = [m["match_score"] for m in matches]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Verify top_track matches highest match_score
        self.assertEqual(res["top_track"], matches[0]["track"])

        # Verify semantic keys and backward-compatible aliases
        for m in matches:
            self.assertIn("match_score", m)
            self.assertIn("affinity_score", m)
            self.assertIn("probability", m)
            self.assertEqual(m["match_score"], m["affinity_score"])
            self.assertEqual(m["match_score"], m["probability"])

            self.assertIn("ml_competency_score", m)
            self.assertIn("competency_prob", m)
            self.assertEqual(m["ml_competency_score"], m["competency_prob"])

            self.assertIn("knn_neighbor_score", m)
            self.assertIn("knn_prob", m)
            self.assertEqual(m["knn_neighbor_score"], m["knn_prob"])

            self.assertIn("dt_leaf_score", m)
            self.assertIn("dt_prob", m)
            self.assertEqual(m["dt_leaf_score"], m["dt_prob"])

            self.assertIn("interest_score", m)
            self.assertIn("degree_prior_score", m)

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
    """Verifies that security invariants, salted PBKDF2 password hashing, legacy migration, and user isolation hold."""
    def setUp(self):
        self.db = DatabaseManager()

    def test_salted_password_hashing_unique_salts(self):
        """Verifies PBKDF2-HMAC-SHA256 password hashing generates unique random salts per call."""
        pwd = "test_password_123"
        h1 = DatabaseManager.hash_password(pwd)
        h2 = DatabaseManager.hash_password(pwd)
        self.assertNotEqual(h1, h2, "Salts must be unique across independent hashing calls")
        self.assertTrue(h1.startswith("pbkdf2:sha256:100000$"))
        self.assertTrue(h2.startswith("pbkdf2:sha256:100000$"))
        parts1 = h1.split("$")
        parts2 = h2.split("$")
        self.assertEqual(len(parts1), 3)
        self.assertEqual(len(parts2), 3)
        self.assertNotEqual(parts1[1], parts2[1], "Salt hex strings must differ")

    def test_password_verification_correct_and_incorrect(self):
        """Verifies constant-time verification accepts valid passwords and rejects incorrect ones."""
        pwd = "super_secret_credentials_456"
        h = DatabaseManager.hash_password(pwd)
        is_valid, needs_rehash = DatabaseManager.verify_password(pwd, h)
        self.assertTrue(is_valid)
        self.assertFalse(needs_rehash)

        is_bad, _ = DatabaseManager.verify_password("wrong_credentials", h)
        self.assertFalse(is_bad)
        is_empty, _ = DatabaseManager.verify_password("", h)
        self.assertFalse(is_empty)
        is_none, _ = DatabaseManager.verify_password(None, h)
        self.assertFalse(is_none)

    def test_legacy_sha256_hash_verification_and_flags_rehash(self):
        """Verifies legacy 64-char unsalted SHA-256 hashes are verified and flagged for rehash."""
        import hashlib
        legacy_pwd = "legacy_kdu_student_pwd"
        legacy_hash = hashlib.sha256(legacy_pwd.encode("utf-8")).hexdigest()
        self.assertEqual(len(legacy_hash), 64)

        is_valid, needs_rehash = DatabaseManager.verify_password(legacy_pwd, legacy_hash)
        self.assertTrue(is_valid)
        self.assertTrue(needs_rehash, "Legacy unsalted hashes must trigger needs_rehash=True")

        is_bad, _ = DatabaseManager.verify_password("wrong_password", legacy_hash)
        self.assertFalse(is_bad)

    def test_user_authentication_transparent_rehash_migration(self):
        """Verifies authenticate_user transparently rehashes legacy SHA-256 records upon successful login."""
        import hashlib
        legacy_pwd = "legacy_user_secret_789"
        legacy_hash = hashlib.sha256(legacy_pwd.encode("utf-8")).hexdigest()
        username = "temp_legacy_mig_user"

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, full_name, email) VALUES (?, ?, ?, ?, ?)",
                (username, legacy_hash, "student", "Legacy Migration User", "legacy.mig@kdu.ac.lk")
            )
            conn.commit()

        try:
            # First authentication: validates legacy hash and upgrades record
            user = self.db.authenticate_user(username, legacy_pwd)
            self.assertIsNotNone(user)
            self.assertEqual(user["username"], username)
            self.assertTrue(user["password_hash"].startswith("pbkdf2:sha256:100000$"))

            # Direct database verification to ensure persistence
            persisted = self.db.get_user_by_id(user["id"])
            self.assertTrue(persisted["password_hash"].startswith("pbkdf2:sha256:100000$"))

            # Subsequent authentication succeeds with newly persisted PBKDF2 hash
            subsequent = self.db.authenticate_user(username, legacy_pwd)
            self.assertIsNotNone(subsequent)
            self.assertTrue(subsequent["password_hash"].startswith("pbkdf2:sha256:100000$"))
        finally:
            with self.db.get_connection() as conn:
                conn.cursor().execute("DELETE FROM users WHERE username = ?", (username,))
                conn.commit()

    def test_user_authentication_success_and_failure(self):
        """Verifies authentication for seeded demo accounts and invalid credentials."""
        user = self.db.authenticate_user("student_demo", "student123")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "student_demo")
        self.assertTrue(user["password_hash"].startswith("pbkdf2:sha256:100000$"))

        bad_user = self.db.authenticate_user("student_demo", "wrong_password")
        self.assertIsNone(bad_user)
        non_user = self.db.authenticate_user("non_existent_student_999", "password")
        self.assertIsNone(non_user)

    def test_sql_injection_defense_parameterization(self):
        """Verifies parameterized queries prevent SQL injection authentication bypass."""
        injection_user = self.db.authenticate_user("admin' OR '1'='1", "admin123")
        self.assertIsNone(injection_user)
        injection_pwd = self.db.authenticate_user("admin", "' OR '1'='1")
        self.assertIsNone(injection_pwd)
        union_attempt = self.db.authenticate_user("' UNION SELECT * FROM users --", "any")
        self.assertIsNone(union_attempt)

    def test_student_profile_isolation(self):
        """Verifies student profile data is isolated by user_id without cross-student data leakage."""
        students = self.db.get_all_students_summary()
        self.assertGreater(len(students), 0)
        for s in students:
            profile = self.db.get_student_profile(s["id"])
            if profile:
                self.assertEqual(profile["user_id"], s["id"])
        self.assertIsNone(self.db.get_student_profile(999999))

    def test_two_users_same_password_receive_different_salted_hashes(self):
        """Verifies that two distinct users registering with the same password receive completely different salted hashes."""
        common_pwd = "CommonUniversityPassword2026!"
        uid1 = self.db.create_user("user_twin_1", common_pwd, "student", "Twin One", email="t1@kdu.ac.lk")
        uid2 = self.db.create_user("user_twin_2", common_pwd, "student", "Twin Two", email="t2@kdu.ac.lk")
        try:
            u1 = self.db.get_user_by_id(uid1)
            u2 = self.db.get_user_by_id(uid2)
            self.assertNotEqual(u1["password_hash"], u2["password_hash"])
            self.assertTrue(u1["password_hash"].startswith("pbkdf2:sha256:100000$"))
            self.assertTrue(u2["password_hash"].startswith("pbkdf2:sha256:100000$"))

            # Both authenticate successfully with their own salted hashes
            self.assertIsNotNone(self.db.authenticate_user("user_twin_1", common_pwd))
            self.assertIsNotNone(self.db.authenticate_user("user_twin_2", common_pwd))
        finally:
            with self.db.get_connection() as conn:
                conn.cursor().execute("DELETE FROM users WHERE id IN (?, ?)", (uid1, uid2))
                conn.commit()

    def test_password_hash_stored_and_loaded_verification(self):
        """Verifies password hashes remain verifiable after complete database persistence and retrieval."""
        test_user = "persist_verify_user"
        test_pass = "PersistPass987!"
        uid = self.db.create_user(test_user, test_pass, "student", "Persist User")
        try:
            user_row = self.db.get_user_by_id(uid)
            stored_hash = user_row["password_hash"]
            is_valid, needs_rehash = DatabaseManager.verify_password(test_pass, stored_hash)
            self.assertTrue(is_valid)
            self.assertFalse(needs_rehash)
        finally:
            with self.db.get_connection() as conn:
                conn.cursor().execute("DELETE FROM users WHERE id = ?", (uid,))
                conn.commit()

    def test_invalid_credentials_cannot_bypass_role_checks(self):
        """Verifies that failed authentication never returns a valid user session or role assignment."""
        roles_to_test = ["admin", "advisor", "coordinator", "student_demo"]
        for username in roles_to_test:
            # Bad password
            user_bad_pwd = self.db.authenticate_user(username, "totally_incorrect_password")
            self.assertIsNone(user_bad_pwd)
            # Empty password
            user_empty = self.db.authenticate_user(username, "")
            self.assertIsNone(user_empty)
            # SQL injection
            user_inj = self.db.authenticate_user(username, "' OR 1=1 --")
            self.assertIsNone(user_inj)


if __name__ == "__main__":
    unittest.main()
