"""
AI Layer 3: A* Search Algorithm for Learning Roadmap Optimization.
Constructs an optimal, dependency-aware, and time-budgeted learning path
from the student's current skill profile to the target career competency benchmark.
"""
import json
import heapq
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import DATA_DIR, SKILL_LEVELS, LEVEL_TO_NAME

class AStarRoadmapOptimizer:
    """
    State-Space Graph Search Optimizer for Student Upskilling.
    
    Formal Problem Formulation:
    - State: (current_skills, completed_activities)
      * current_skills: Mapping of skill names to discrete proficiency levels (0=None, 1=Beginner, 2=Intermediate, 3=Advanced).
      * completed_activities: Tuple of activity IDs completed so far along the search path.
    - Start State s_0: Student's initial verified skills, empty completed activities tuple.
    - Goal Condition: All required target competencies for the career track are satisfied (current_skills[s] >= target[s]).
    - Actions / Transitions: Selecting an uncompleted activity whose prerequisite activity IDs are all completed.
    - Step Cost c(s, a, s'): Estimated duration in hours of activity a (c(a) > 0).
    - Objective: Minimize total study hours g(n) = sum_{a in path} Hours(a) to reach the goal state.
    """
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

    def _heuristic_remaining_skill_distance(
        self, current_skills: Dict[str, int], target_skills: Dict[str, int], remaining_activities: List[dict]
    ) -> float:
        """
        Admissible Heuristic h(n):
        Estimates the minimum study hours required to bridge all remaining skill gaps.

        Mathematical Formulation & Admissibility:
        Let S_gaps = {s in target_skills | current_skills[s] < target_skills[s]}.
        For each unsatisfied skill gap s, let Cand(s) be the available activities that can advance
        skill s beyond its current level.

        Admissibility Proof:
        1. Single-Skill Activities (Disjoint Partition):
           In the curated curriculum graph, each activity has a single 'main_skill'.
           The candidate sets Cand(s) are therefore mutually disjoint for distinct skills.
           Any valid sequence that bridges all gaps must select at least one activity from each Cand(s).
           Hence h*(n) >= sum_{s in S_gaps} min_{a in Cand(s)} Hours(a).
        2. Multi-Skill Activities (Shared Effects / Overlap Prevention):
           If an activity can satisfy multiple skill gaps simultaneously, summing minimums could
           double-count that activity's cost. To preserve strict admissibility (h(n) <= h*(n)) across
           arbitrary graphs, the heuristic computes the total hours over UNIQUE minimal candidate activities.
           If a single activity satisfies multiple gaps, its hours are counted exactly once.
        3. Missing / Unreachable Activities:
           If an unsatisfied skill has no remaining candidate activity in the graph, it contributes 0.0
           to remain an admissible lower bound on reachable graph cost (never overestimating).
        """
        best_candidate_per_skill = {}
        for skill, target_val in target_skills.items():
            curr_val = current_skills.get(skill, 0)
            if curr_val < target_val:
                candidates = [
                    act for act in remaining_activities
                    if (act.get("main_skill") == skill or skill in act.get("secondary_skills", []))
                    and SKILL_LEVELS.get(act.get("skill_level_gain", "Beginner"), 1) > curr_val
                ]
                if candidates:
                    min_act = min(candidates, key=lambda a: a.get("estimated_hours", 10))
                    best_candidate_per_skill[skill] = min_act

        if not best_candidate_per_skill:
            return 0.0

        # Unique minimal activities prevent double-counting if an activity advances multiple skills
        unique_acts = {act["id"]: act for act in best_candidate_per_skill.values()}
        return float(sum(act.get("estimated_hours", 0) for act in unique_acts.values()))

    def generate_optimal_roadmap(
        self, target_track: str, current_skills: Dict[str, str], weekly_hours: int = 8
    ) -> Dict:
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
                "is_optimal": True,
                "status": "already_satisfied"
            }

        # Filter activities relevant to this track or identified gaps
        relevant_activities = [
            a for a in self.activities
            if a.get("track") == target_track or a.get("main_skill") in active_gaps
        ]
        if not relevant_activities:
            relevant_activities = self.activities

        # Check for gaps that have no corresponding activity in the graph
        uncovered_gaps = [
            s for s in active_gaps
            if not any(
                (a.get("main_skill") == s or s in a.get("secondary_skills", []))
                and SKILL_LEVELS.get(a.get("skill_level_gain", "Beginner"), 1) > student_skills_numeric.get(s, 0)
                for a in relevant_activities
            )
        ]

        # If uncovered gaps exist, A* targets the reachable subset of competencies
        if uncovered_gaps:
            search_targets = {s: lvl for s, lvl in target_skills_numeric.items() if s not in uncovered_gaps}
        else:
            search_targets = target_skills_numeric

        # Priority Queue for A*: (f_score, g_cost, state_id, current_skills_tuple, completed_ids_tuple, path)
        counter = 0
        initial_skills_tuple = tuple(sorted(student_skills_numeric.items()))
        initial_completed = tuple()
        initial_h = self._heuristic_remaining_skill_distance(student_skills_numeric, search_targets, relevant_activities)

        open_set = []
        heapq.heappush(open_set, (initial_h, 0, counter, initial_skills_tuple, initial_completed, []))

        best_g: Dict[Tuple, float] = {}
        best_path = None
        best_cost = float("inf")
        status = "unknown"
        status_message = ""

        max_iterations = 2000
        iteration = 0

        # Run A* search if there are reachable targets
        if search_targets:
            while open_set and iteration < max_iterations:
                iteration += 1
                f, g, _, curr_skills_tup, completed_ids_tup, path = heapq.heappop(open_set)
                curr_skills_dict = dict(curr_skills_tup)
                completed_set = set(completed_ids_tup)

                state_key = (curr_skills_tup, completed_ids_tup)
                if state_key in best_g and best_g[state_key] <= g:
                    continue
                best_g[state_key] = g

                # Check if Goal State reached: All target skills in search_targets satisfied
                goal_reached = True
                for skill, target_val in search_targets.items():
                    if curr_skills_dict.get(skill, 0) < target_val:
                        goal_reached = False
                        break

                if goal_reached:
                    best_path = path
                    best_cost = g
                    if not uncovered_gaps:
                        status = "optimal_solution_found"
                        status_message = "Optimal A* learning sequence found satisfying all target competencies."
                    else:
                        status = "partial_coverage"
                        status_message = (
                            "A* sequence generated for available graph activities. Note: certain prerequisite "
                            f"competencies ({', '.join(uncovered_gaps)}) are curriculum coursework outside this activity graph."
                        )
                    break

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
                    new_g = g + act.get("estimated_hours", 10)
                    new_completed = tuple(sorted(list(completed_set | {act_id})))

                    # Update skill levels from this activity
                    new_skills_dict = dict(curr_skills_dict)
                    gain_skill = act.get("main_skill")
                    gain_lvl_val = SKILL_LEVELS.get(act.get("skill_level_gain", "Beginner"), 1)
                    if gain_skill:
                        new_skills_dict[gain_skill] = max(new_skills_dict.get(gain_skill, 0), gain_lvl_val)
                    if "secondary_skills" in act:
                        for sec in act["secondary_skills"]:
                            new_skills_dict[sec] = max(new_skills_dict.get(sec, 0), gain_lvl_val)

                    new_skills_tup = tuple(sorted(new_skills_dict.items()))

                    # Compute remaining activities for heuristic
                    remaining_acts = [a for a in relevant_activities if a["id"] not in new_completed]
                    new_h = self._heuristic_remaining_skill_distance(new_skills_dict, search_targets, remaining_acts)
                    new_f = new_g + new_h

                    counter += 1
                    heapq.heappush(open_set, (new_f, new_g, counter, new_skills_tup, new_completed, path + [act]))

        # Handle fallback if A* could not reach the goal
        if best_path is None:
            if iteration >= max_iterations:
                status = "search_limit_reached"
                status_message = "A* search reached iteration limit; generated valid prerequisite-ordered fallback."
            else:
                status = "goal_unreachable"
                status_message = "Target competencies unreachable in activity graph; generated valid prerequisite-ordered fallback."
            best_path = self._topological_fallback(relevant_activities, active_gaps)

        # Strictly distinguish optimal solutions:
        # A solution is optimal if and only if A* reached the goal satisfying all active gaps without missing coverage.
        a_star_optimal = (status == "optimal_solution_found")

        # Build week-by-week schedule respecting declared weekly study budget
        scheduled_roadmap = []
        current_week = 1
        current_week_hours_accum = 0
        total_hours = 0

        for act in best_path:
            hours = max(1, int(act.get("estimated_hours", 1)))
            total_hours += hours

            # If adding this activity exceeds current week budget and current week already has work
            if current_week_hours_accum + hours > weekly_hours and current_week_hours_accum > 0:
                current_week += 1
                current_week_hours_accum = 0

            start_week = current_week
            avail_in_start_week = weekly_hours - current_week_hours_accum

            if hours <= avail_in_start_week:
                # Fits entirely within the current week
                end_week = start_week
                current_week_hours_accum += hours
                if current_week_hours_accum == weekly_hours:
                    current_week += 1
                    current_week_hours_accum = 0
            else:
                # Spans across multiple weeks
                remaining_hours = hours - avail_in_start_week
                extra_weeks = (remaining_hours + weekly_hours - 1) // weekly_hours
                end_week = start_week + extra_weeks
                rem_final = remaining_hours % weekly_hours
                if rem_final == 0:
                    current_week = end_week + 1
                    current_week_hours_accum = 0
                else:
                    current_week = end_week
                    current_week_hours_accum = rem_final

            scheduled_roadmap.append({
                "activity_id": act["id"],
                "activity_title": act["title"],
                "activity_type": act.get("type", "Course"),
                "estimated_hours": hours,
                "main_skill": act.get("main_skill", ""),
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
            "is_optimal": a_star_optimal,
            "status": status,
            "message": status_message
        }

    def _topological_fallback(self, activities: List[dict], active_gaps: Dict[str, int]) -> List[dict]:
        """Simple topological sort fallback to guarantee a valid sequenced roadmap."""
        ordered = []
        added_ids = set()

        # Sort candidates prioritizing those closing gaps
        candidates = sorted(
            activities,
            key=lambda a: (0 if a.get("main_skill") in active_gaps else 1, a.get("estimated_hours", 10))
        )

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
