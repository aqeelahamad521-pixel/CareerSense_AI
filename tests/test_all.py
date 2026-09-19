"""
Comprehensive Test Suite for CareerSense AI.
Validates:
1. Rule-Based Expert System (Prerequisites, Skill Gaps, Dependencies, Readiness)
2. A* Search Roadmap Optimizer (Dependency order, time budget constraints, heuristics)
3. Machine Learning Classifier (Feature extraction, predictions, format)
4. End-to-End Workflow Integration
"""
import unittest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import ML_FEATURE_COLUMNS, CAREER_TRACKS
from ai_engine.rule_engine import RuleBasedExpertSystem
from ai_engine.a_star_roadmap import AStarRoadmapOptimizer
from ai_engine.explainability import ExplainabilityEngine

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
        from ai_engine.ml_classifier import CareerClassifier
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
        self.assertEqual(len(res["ranked_matches"]), len(CAREER_TRACKS))
        self.assertEqual(res["top_track"], "Software Engineering")

class TestIntegrationExplainability(unittest.TestCase):
    def setUp(self):
        from ai_engine.ml_classifier import CareerClassifier
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

