"""
Explainability & Model Interpretation Engine for CareerSense AI.
Provides multi-dimensional, transparent reasoning for:
1. Ranked career predictions and confidence estimation
2. Feature attribution (academic grades, interests, existing skills)
3. Prerequisite rule justifications
4. Visual competency radar metrics
5. Roadmap activity rationale
"""
from typing import Dict, List, Any
import numpy as np

class ExplainabilityEngine:
    def __init__(self, rule_engine, ml_classifier=None):
        self.rule_engine = rule_engine
        self.ml_classifier = ml_classifier

    def generate_full_explanation(self, student_profile: dict, academic_records: list, student_skills: dict, student_interests: dict, target_track: str = None) -> Dict[str, Any]:
        """
        Generates a comprehensive explainability bundle for a student.
        """
        # 1. Run ML classification (if available) or fallback to rule heuristic
        if self.ml_classifier and self.ml_classifier.is_trained:
            ml_results = self.ml_classifier.predict_career_matches(academic_records, student_interests, student_skills)
        else:
            ml_results = self._heuristic_career_ranking(academic_records, student_interests, student_skills)

        # Selected or top recommended track
        chosen_track = target_track or (ml_results["ranked_matches"][0]["track"] if ml_results.get("ranked_matches") else "Software Engineering")

        # 2. Rule Engine Prerequisite Evaluation
        prereq_eval = self.rule_engine.evaluate_academic_prerequisites(chosen_track, academic_records)

        # 3. Rule Engine Skill Gap Analysis
        skill_gaps = self.rule_engine.analyze_skill_gaps(chosen_track, student_skills)

        # 4. Inter-skill dependency alerts
        dep_alerts = self.rule_engine.verify_skill_dependencies(student_skills)

        # 5. Influential Factor Attribution
        influential_factors = self._extract_influential_factors(chosen_track, academic_records, student_interests, student_skills)

        # 6. Radar Chart Comparison Data
        radar_data = self._generate_radar_comparison_data(chosen_track, student_skills)

        return {
            "target_track": chosen_track,
            "ranked_matches": ml_results.get("ranked_matches", []),
            "model_type": ml_results.get("model_used", "K-NN & Decision Tree Pipeline"),
            "influential_factors": influential_factors,
            "prerequisite_evaluation": prereq_eval,
            "skill_gaps": skill_gaps,
            "dependency_alerts": dep_alerts,
            "radar_comparison": radar_data,
            "explainability_summary": self._synthesize_summary(chosen_track, ml_results, prereq_eval, skill_gaps)
        }

    def _heuristic_career_ranking(self, academic_records: list, interests: dict, skills: dict) -> dict:
        """Heuristic baseline when ML models are being initialized."""
        from config import CAREER_TRACKS
        matches = []
        for track in CAREER_TRACKS:
            score = 65.0
            if track == "Software Engineering":
                score += (interests.get("Software Development & Systems", 3) - 3) * 6
            elif track == "Data Science / AI":
                score += (interests.get("Data Analysis & AI Research", 3) - 3) * 6
            elif track == "Cybersecurity":
                score += (interests.get("Cybersecurity & Threat Defense", 3) - 3) * 6
            elif track == "Cloud / DevOps":
                score += (interests.get("Cloud Infrastructure & Automation", 3) - 3) * 6
            elif track == "UI/UX Design":
                score += (interests.get("UI/UX Design & User Experience", 3) - 3) * 6
            matches.append({"track": track, "probability": round(min(98.0, max(20.0, score)), 1)})

        matches.sort(key=lambda x: -x["probability"])
        return {"ranked_matches": matches, "model_used": "Knowledge-Based Heuristic"}

    def _extract_influential_factors(self, track: str, academics: list, interests: dict, skills: dict) -> dict:
        """
        Extracts positive drivers and constraining factors influencing the career recommendation.
        """
        positive_drivers = []
        growth_areas = []

        # Analyze relevant grades
        for rec in academics:
            subj = rec.get("subject_area", "")
            grade = rec.get("grade", "C")
            pts = rec.get("grade_points", 2.0)
            if pts >= 3.3: # B+ or higher
                positive_drivers.append(f"Strong academic performance in {subj} (Grade: {grade}) provides solid foundations.")
            elif pts < 2.0:
                growth_areas.append(f"Lower score in {subj} (Grade: {grade}) requires revision for technical interviews.")

        # Analyze domain interests
        for category, rating in interests.items():
            if rating >= 4:
                positive_drivers.append(f"High self-reported affinity ({rating}/5) for '{category}'.")
            elif rating <= 2 and track.lower() in category.lower():
                growth_areas.append(f"Moderate interest ({rating}/5) logged in primary discipline '{category}'.")

        # Analyze core skills
        advanced_skills = [k for k, v in skills.items() if v in ("Intermediate", "Advanced")]
        if advanced_skills:
            positive_drivers.append(f"Demonstrated competency in key tools: {', '.join(advanced_skills[:4])}.")

        return {
            "strengths": positive_drivers if positive_drivers else ["Consistent overall academic baseline across general computing modules."],
            "limitations": growth_areas if growth_areas else ["No critical academic deficiencies detected; focus on targeted practical portfolio building."]
        }

    def _generate_radar_comparison_data(self, track: str, student_skills: dict) -> dict:
        """
        Produces aligned arrays for Plotly radar visualization:
        - Labels (competencies)
        - Student current levels (0-3)
        - Benchmark target levels (0-3)
        """
        career_def = self.rule_engine.careers.get(track, {})
        req_list = career_def.get("required_competencies", [])
        
        from config import SKILL_LEVELS

        labels = []
        student_vals = []
        benchmark_vals = []

        for req in req_list:
            labels.append(req["skill"])
            benchmark_vals.append(SKILL_LEVELS.get(req["target_level"], 2))
            student_vals.append(SKILL_LEVELS.get(student_skills.get(req["skill"], "None"), 0))

        return {
            "categories": labels,
            "student_values": student_vals,
            "benchmark_values": benchmark_vals
        }

    def _synthesize_summary(self, track: str, ml_res: dict, prereq_eval: dict, gaps: list) -> str:
        high_gaps = [g["skill"] for g in gaps if g["priority"] == "High"]
        prereq_status = "All foundational academic prerequisites are satisfied." if prereq_eval.get("overall_academic_met") else "Some academic prerequisite thresholds require attention."
        
        gap_text = f"Top priority skill gaps identified: {', '.join(high_gaps)}." if high_gaps else "All core competencies currently meet or exceed the benchmark levels."
        
        return f"{track} is evaluated as a suitable career pathway. {prereq_status} {gap_text}"
