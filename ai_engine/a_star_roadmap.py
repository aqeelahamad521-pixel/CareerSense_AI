"""
AI Layer 3: A* Search Algorithm for Learning Roadmap Optimization.
Constructs an optimal, dependency-aware, and time-budgeted learning path
from the student's current skill profile to the target career competency benchmark.
"""
import json
import heapq
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import DATA_DIR, SKILL_LEVELS, LEVEL_TO_NAME

class AStarRoadmapOptimizer:
    def __init__(self):
        self.learning_graph_path = DATA_DIR / "learning_graph.json"
        self.careers_path = DATA_DIR / "career_definitions.json"
        self.load_graph()

    def load_graph(self):
        with open(self.learning_graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.activities = data.get("activities", [])
            self.activity_dict = {a["id"]: a for a in self.activities}

        with open(self.careers_path, "r", encoding="utf-8") as f:
            self.careers = json.load(f)

    def _heuristic_remaining_skill_distance(self, current_skills: Dict[str, int], target_skills: Dict[str, int], remaining_activities: List[dict]) -> float:
        """
        Admissible heuristic h(n):
        Estimates the minimum study hours required to bridge all remaining skill gaps.
        For each unsatisfied skill gap, finds the minimum estimated hours among available
        activities that can actually advance that skill beyond its current level.
        If no activity can advance an unsatisfied skill gap, 0.0 is contributed so as not
        to violate admissibility (never overestimating actual reachable graph cost).
        """
        total_heuristic = 0.0
        for skill, target_val in target_skills.items():
            curr_val = current_skills.get(skill, 0)
            if curr_val < target_val:
                gap = target_val - curr_val
                # Find available activities that actually advance this skill beyond current level
                candidate_hours = [
                    act["estimated_hours"] for act in remaining_activities
                    if act.get("main_skill") == skill and SKILL_LEVELS.get(act.get("skill_level_gain", "Beginner"), 1) > curr_val
                ]
                if candidate_hours:
                    # Admissible lower bound (minimum hours among activities that advance this skill)
                    min_act_hours = min(candidate_hours)
                    total_heuristic += min_act_hours
                else:
                    # If no remaining activity can advance this skill in this graph, contribute 0.0 to stay admissible
                    total_heuristic += 0.0
        return float(total_heuristic)

    def generate_optimal_roadmap(self, target_track: str, current_skills: Dict[str, str], weekly_hours: int = 8) -> Dict:
        """
        Runs A* search to find the optimal sequence of learning activities
        that satisfy all skill gaps for the target career track.
        """
        weekly_hours = max(1, int(weekly_hours or 8))
        career_def = self.careers.get(target_track, {})
        req_competencies = career_def.get("required_competencies", [])
        
        # Determine target numeric skill levels
        target_skills_numeric = {
            c["skill"]: SKILL_LEVELS.get(c["target_level"], 2)
            for c in req_competencies
        }
        
        # Convert student current skills to numeric
        student_skills_numeric = {
            s: SKILL_LEVELS.get(current_skills.get(s, "None"), 0)
            for s in target_skills_numeric
        }

        # Identify actual gaps
        active_gaps = {
            s: target_skills_numeric[s] - student_skills_numeric[s]
            for s in target_skills_numeric
            if target_skills_numeric[s] > student_skills_numeric[s]
        }

        if not active_gaps:
            return {
                "target_track": target_track,
                "total_hours": 0,
                "total_weeks": 0,
                "weekly_hours_budget": weekly_hours,
                "message": f"All benchmark competencies for {target_track} are already satisfied!",
                "roadmap": [],
                "algorithm": "A* Search with Admissible Skill-Distance Heuristic",
                "is_optimal": True
            }

        # Filter activities relevant to this track or identified gaps
        relevant_activities = [
            a for a in self.activities
            if a.get("track") == target_track or a.get("main_skill") in active_gaps
        ]
        if not relevant_activities:
            # Fallback to all activities
            relevant_activities = self.activities

        # Priority Queue for A*: (f_score, g_cost, state_id, current_skills_tuple, completed_ids_tuple, path)
        counter = 0
        initial_skills_tuple = tuple(sorted(student_skills_numeric.items()))
        initial_completed = tuple()
        initial_h = self._heuristic_remaining_skill_distance(student_skills_numeric, target_skills_numeric, relevant_activities)
        
        open_set = []
        heapq.heappush(open_set, (initial_h, 0, counter, initial_skills_tuple, initial_completed, []))
        
        visited_states = set()
        best_path = None
        best_cost = float("inf")

        max_iterations = 2000
        iteration = 0

        while open_set and iteration < max_iterations:
            iteration += 1
            f, g, _, curr_skills_tup, completed_ids_tup, path = heapq.heappop(open_set)
            curr_skills_dict = dict(curr_skills_tup)
            completed_set = set(completed_ids_tup)

            # Check if Goal State reached: All target skills satisfied
            goal_reached = True
            for skill, target_val in target_skills_numeric.items():
                if curr_skills_dict.get(skill, 0) < target_val:
                    goal_reached = False
                    break

            if goal_reached:
                best_path = path
                best_cost = g
                break

            state_key = (curr_skills_tup, completed_ids_tup)
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            # Explore valid candidate activities (transitions)
            for act in relevant_activities:
                act_id = act["id"]
                if act_id in completed_set:
                    continue
                
                # Check prerequisites: all required prerequisite activity IDs must be completed
                prereqs = act.get("prerequisites", [])
                prereqs_met = all(p in completed_set for p in prereqs)
                if not prereqs_met:
                    continue

                # Transition: Take this activity
                new_g = g + act["estimated_hours"]
                new_completed = tuple(sorted(list(completed_set | {act_id})))
                
                # Update skill levels from this activity
                new_skills_dict = dict(curr_skills_dict)
                gain_skill = act["main_skill"]
                gain_lvl_val = SKILL_LEVELS.get(act.get("skill_level_gain", "Beginner"), 1)
                new_skills_dict[gain_skill] = max(new_skills_dict.get(gain_skill, 0), gain_lvl_val)
                new_skills_tup = tuple(sorted(new_skills_dict.items()))

                # Compute remaining activities for heuristic
                remaining_acts = [a for a in relevant_activities if a["id"] not in new_completed]
                new_h = self._heuristic_remaining_skill_distance(new_skills_dict, target_skills_numeric, remaining_acts)
                new_f = new_g + new_h

                counter += 1
                heapq.heappush(open_set, (new_f, new_g, counter, new_skills_tup, new_completed, path + [act]))

        # Fallback: if search limit reached or goal unreachable due to graph sparsity,
        # sort relevant activities by prerequisite topological order & priority
        a_star_optimal = (best_path is not None)
        if not best_path:
            best_path = self._topological_fallback(relevant_activities, active_gaps)

        # Build week-by-week schedule based on student's weekly study budget
        scheduled_roadmap = []
        current_week = 1
        current_week_hours_accum = 0
        total_hours = 0

        for act in best_path:
            hours = act["estimated_hours"]
            total_hours += hours
            
            # If adding this activity exceeds current week budget and we already have work in current week
            if current_week_hours_accum + hours > weekly_hours and current_week_hours_accum > 0:
                current_week += 1
                current_week_hours_accum = 0

            start_week = current_week
            # Span multiple weeks if activity alone exceeds budget
            span_weeks = max(1, (hours + weekly_hours - 1) // weekly_hours)
            end_week = start_week + span_weeks - 1
            current_week = end_week
            current_week_hours_accum += (hours % weekly_hours)

            scheduled_roadmap.append({
                "activity_id": act["id"],
                "activity_title": act["title"],
                "activity_type": act["type"],
                "estimated_hours": hours,
                "main_skill": act["main_skill"],
                "skill_level_gain": act.get("skill_level_gain", "Intermediate"),
                "priority": act.get("priority", "Medium"),
                "prerequisites": act.get("prerequisites", []),
                "week_number": start_week,
                "week_end": end_week,
                "description": act.get("description", ""),
                "resource_url": act.get("resource_url", "#"),
                "status": "Planned"
            })

        total_weeks = max([item["week_end"] for item in scheduled_roadmap], default=0)

        return {
            "target_track": target_track,
            "total_hours": total_hours,
            "total_weeks": total_weeks,
            "weekly_hours_budget": weekly_hours,
            "roadmap": scheduled_roadmap,
            "algorithm": "A* Search with Admissible Skill-Distance Heuristic" if a_star_optimal else "Topological Prerequisite Ordering (Fallback)",
            "is_optimal": a_star_optimal
        }

    def _topological_fallback(self, activities: List[dict], active_gaps: Dict[str, int]) -> List[dict]:
        """Simple topological sort fallback to guarantee a valid sequenced roadmap."""
        ordered = []
        added_ids = set()
        
        # Sort candidates prioritizing those closing gaps
        candidates = sorted(activities, key=lambda a: (0 if a.get("main_skill") in active_gaps else 1, a.get("estimated_hours", 10)))
        
        for _ in range(len(candidates)):
            for act in candidates:
                if act["id"] in added_ids:
                    continue
                prereqs = act.get("prerequisites", [])
                if all(p in added_ids for p in prereqs):
                    ordered.append(act)
                    added_ids.add(act["id"])
                    break
        return ordered
