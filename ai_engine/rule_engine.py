"""
AI Layer 2: Rule-Based Expert System for CareerSense AI.
Provides transparent forward-chaining inference for:
1. Academic prerequisite validation
2. Competency gap analysis & priority assignment
3. Inter-skill dependency verification
4. Portfolio & profile readiness evaluation
5. Explainable reasoning generation
"""
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import DATA_DIR, SKILL_LEVELS, LEVEL_TO_NAME, GRADE_POINTS

class RuleBasedExpertSystem:
    def __init__(self):
        self.rules_path = DATA_DIR / "rules_knowledge_base.json"
        self.careers_path = DATA_DIR / "career_definitions.json"
        self.load_knowledge_base()

    def load_knowledge_base(self):
        with open(self.rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)
        with open(self.careers_path, "r", encoding="utf-8") as f:
            self.careers = json.load(f)

    def evaluate_academic_prerequisites(self, target_track: str, academic_records: list) -> dict:
        """
        Validates whether student meets minimum academic prerequisite grades for target track.
        """
        results = {
            "satisfied": [],
            "violations": [],
            "overall_academic_met": True
        }
        
        # Build map of subject -> highest grade points achieved
        subject_grades = {}
        for rec in academic_records:
            subj = rec["subject_area"]
            pts = rec.get("grade_points", 0.0)
            if subj not in subject_grades or pts > subject_grades[subj]["points"]:
                subject_grades[subj] = {
                    "points": pts,
                    "grade": rec.get("grade", "N/A"),
                    "module_name": rec.get("module_name", "N/A")
                }

        # Check against academic prerequisite rules
        for rule in self.rules.get("academic_prerequisite_rules", []):
            if rule["target_track"].lower() == target_track.lower():
                subj = rule["subject"]
                min_pts = rule["minimum_points"]
                min_grade = rule["minimum_grade"]
                
                if subj in subject_grades:
                    student_pts = subject_grades[subj]["points"]
                    student_grade = subject_grades[subj]["grade"]
                    
                    if student_pts >= min_pts:
                        results["satisfied"].append({
                            "rule_id": rule["id"],
                            "subject": subj,
                            "required_grade": min_grade,
                            "student_grade": student_grade,
                            "status": "Pass",
                            "message": f"Prerequisite satisfied: Grade {student_grade} in {subj} meets required minimum {min_grade}."
                        })
                    else:
                        results["overall_academic_met"] = False
                        results["violations"].append({
                            "rule_id": rule["id"],
                            "subject": subj,
                            "required_grade": min_grade,
                            "student_grade": student_grade,
                            "status": "Fail",
                            "severity": rule.get("severity", "High"),
                            "message": rule["message"],
                            "remedy": rule["remedy"]
                        })
                else:
                    # Subject not yet taken
                    results["overall_academic_met"] = False
                    results["violations"].append({
                        "rule_id": rule["id"],
                        "subject": subj,
                        "required_grade": min_grade,
                        "student_grade": "Not Taken",
                        "status": "Missing",
                        "severity": rule.get("severity", "High"),
                        "message": f"Foundational subject '{subj}' has not been recorded in academic profile.",
                        "remedy": rule["remedy"]
                    })
                    
        return results

    def analyze_skill_gaps(self, target_track: str, student_skills: dict) -> list:
        """
        Compares student's current proficiency levels with benchmark competencies required for target track.
        Assigns gap size (0 to 3 levels) and priority (High, Medium, Low).
        """
        career_def = self.careers.get(target_track)
        if not career_def:
            return []

        gaps = []
        for req in career_def.get("required_competencies", []):
            skill_name = req["skill"]
            target_lvl_str = req["target_level"]
            target_lvl_int = SKILL_LEVELS.get(target_lvl_str, 2)
            
            curr_lvl_str = student_skills.get(skill_name, "None")
            curr_lvl_int = SKILL_LEVELS.get(curr_lvl_str, 0)
            
            gap_size = max(0, target_lvl_int - curr_lvl_int)
            
            # Determine Priority based on gap size and requirement priority
            if gap_size == 0:
                priority = "Satisfied"
                status = "Met"
            elif req.get("priority") == "High" or gap_size >= 2:
                priority = "High"
                status = "Critical Gap"
            else:
                priority = "Medium"
                status = "Moderate Gap"

            gaps.append({
                "skill": skill_name,
                "current_level": curr_lvl_str,
                "current_val": curr_lvl_int,
                "required_level": target_lvl_str,
                "required_val": target_lvl_int,
                "gap_levels": gap_size,
                "priority": priority,
                "status": status,
                "reason": f"Required for {target_track} role ({target_lvl_str} proficiency benchmark)."
            })

        # Sort gaps: High priority first, then by gap size descending
        priority_order = {"High": 0, "Medium": 1, "Satisfied": 2}
        gaps.sort(key=lambda x: (priority_order.get(x["priority"], 3), -x["gap_levels"]))
        return gaps

    def verify_skill_dependencies(self, student_skills: dict) -> list:
        """
        Checks inter-skill dependencies (e.g. Automated Testing requires OOP).
        """
        dependency_alerts = []
        for dep in self.rules.get("competency_dependency_rules", []):
            target_skill = dep["skill"]
            prereq_skill = dep["prerequisite_skill"]
            min_lvl_str = dep["min_prereq_level"]
            min_lvl_val = SKILL_LEVELS.get(min_lvl_str, 1)

            target_val = SKILL_LEVELS.get(student_skills.get(target_skill, "None"), 0)
            prereq_val = SKILL_LEVELS.get(student_skills.get(prereq_skill, "None"), 0)

            # If student is attempting target skill without meeting prerequisite
            if target_val > 0 and prereq_val < min_lvl_val:
                dependency_alerts.append({
                    "rule_id": dep["id"],
                    "target_skill": target_skill,
                    "prerequisite_skill": prereq_skill,
                    "required_level": min_lvl_str,
                    "current_prereq_level": LEVEL_TO_NAME.get(prereq_val, "None"),
                    "message": dep["message"],
                    "priority": dep.get("priority", "High")
                })
        return dependency_alerts

    def calculate_career_readiness(self, target_track: str, student_profile: dict, academic_records: list, student_skills: dict, projects: list, documents: list) -> dict:
        """
        Computes the holistic Career Readiness Score (0% to 100%) broken down into 4 components:
        1. Academic Score (25%)
        2. Competency Score (50%)
        3. Project Evidence Score (15%)
        4. Profile & Document Score (10%)
        """
        career_def = self.careers.get(target_track, {})
        req_competencies = career_def.get("required_competencies", [])
        
        # 1. Competency Coverage (50% max)
        total_target_pts = 0
        student_achieved_pts = 0
        for req in req_competencies:
            t_val = SKILL_LEVELS.get(req["target_level"], 2)
            c_val = SKILL_LEVELS.get(student_skills.get(req["skill"], "None"), 0)
            total_target_pts += t_val
            student_achieved_pts += min(c_val, t_val)
        
        competency_ratio = (student_achieved_pts / total_target_pts) if total_target_pts > 0 else 0.5
        competency_score = round(competency_ratio * 50.0, 1)

        # 2. Academic Score (25% max)
        acad_results = self.evaluate_academic_prerequisites(target_track, academic_records)
        gpa = student_profile.get("gpa", 0.0)
        # Scale GPA (max 4.0) into base points
        base_acad = (min(gpa, 4.0) / 4.0) * 20.0
        # Bonus for meeting all prerequisites
        prereq_bonus = 5.0 if acad_results.get("overall_academic_met", True) else 0.0
        academic_score = round(min(25.0, base_acad + prereq_bonus), 1)

        # 3. Project Evidence Score (15% max)
        proj_count = len(projects)
        project_score = round(min(15.0, proj_count * 7.5), 1) # 2 projects = full score

        # 4. Profile Completeness & Documents (10% max)
        doc_count = len(documents)
        cv_present = any(d.get("doc_type") == "CV" for d in documents) or bool(student_profile.get("cv_filename"))
        doc_pts = 5.0 if cv_present else 0.0
        doc_pts += min(5.0, (doc_count - (1 if cv_present else 0)) * 2.5)
        document_score = round(min(10.0, doc_pts), 1)

        total_readiness = round(competency_score + academic_score + project_score + document_score, 1)

        # Generate readiness label
        if total_readiness >= 80:
            readiness_tier = "Job / Internship Ready"
            tier_color = "green"
        elif total_readiness >= 60:
            readiness_tier = "Near Ready (Minor Gaps)"
            tier_color = "blue"
        elif total_readiness >= 40:
            readiness_tier = "Developing (Targeted Training Needed)"
            tier_color = "orange"
        else:
            readiness_tier = "Foundational Stage (Significant Gaps)"
            tier_color = "red"

        # Check portfolio & workload rules
        advisory_warnings = []
        weekly_hours = student_profile.get("weekly_hours", 8)
        gap_count = sum(1 for g in self.analyze_skill_gaps(target_track, student_skills) if g["gap_levels"] > 0)
        
        if proj_count == 0:
            advisory_warnings.append({
                "type": "Portfolio Notice",
                "message": "No projects recorded. Real-world projects significantly enhance internship readiness."
            })
        if not cv_present:
            advisory_warnings.append({
                "type": "Documentation Notice",
                "message": "No CV uploaded. Uploading a CV allows advisor verification."
            })
        if weekly_hours < 8 and gap_count > 3:
            advisory_warnings.append({
                "type": "Study Allocation Notice",
                "message": f"Weekly study time ({weekly_hours} hrs/wk) is low for {gap_count} remaining gaps. Consider increasing study time."
            })

        return {
            "total_readiness_score": total_readiness,
            "readiness_tier": readiness_tier,
            "tier_color": tier_color,
            "breakdown": {
                "competency_score": competency_score,
                "competency_max": 50.0,
                "academic_score": academic_score,
                "academic_max": 25.0,
                "project_score": project_score,
                "project_max": 15.0,
                "document_score": document_score,
                "document_max": 10.0
            },
            "academic_eval": acad_results,
            "advisory_warnings": advisory_warnings
        }
