# Graph Report - CareerSense_AI  (2026-10-15)

## Corpus Check
- Corpus is ~45,237 words - fits in a single context window. You may not need a graph.

## Summary
- 392 nodes · 621 edges · 25 communities (13 shown, 12 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 29 edges (avg confidence: 0.92)
- Token cost: unavailable (the host-agent extraction API did not expose usage counts; zero placeholders were not treated as real usage)

## Graph Health Note
- Diagnostic warning: 77 dangling-endpoint edges, 2 self-loops, and 4 endpoint-pair collapses were detected in the raw extraction. The graph remains useful for navigation, but architectural conclusions were manually checked against source code and tests.

## Community Hubs (Navigation)
- Product Claims and CI
- ML Evaluation Tests
- Web App Orchestration
- Project Module Dependencies
- SQLite Data Persistence
- Rules and Explainability
- Security and Role Isolation
- ML Classification Semantics
- A* Controlled Tests
- A* Roadmap Engine
- PDF Report Generation
- Synthetic Dataset Validation
- Basic A* Tests
- A* Optimality Proofs
- Duration Clamping
- Search Exhaustion Fallback
- Deep Prerequisite Chains
- Missing Prerequisite Fallback
- Direct Course Optimization
- Multi-Step Path Optimization
- Zero-Gap Handling
- Weekly Budget Validation
- Dijkstra Parity Validation
- Secondary Skill Inclusion
- Partial Coverage Reporting

## God Nodes (most connected - your core abstractions)
1. `DatabaseManager` - 38 edges
2. `TestAStarControlledOptimality` - 28 edges
3. `TestMLEvaluationMathematicalCorrectness` - 22 edges
4. `AStarRoadmapOptimizer` - 20 edges
5. `compute_multiclass_metrics()` - 20 edges
6. `CareerClassifier` - 20 edges
7. `TestSecurityAndRoleIsolation` - 19 edges
8. `RuleBasedExpertSystem` - 14 edges
9. `ExplainabilityEngine` - 12 edges
10. `get_system_components()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Final Model Evaluation Benchmarks` --semantically_similar_to--> `Final Held-Out Model Benchmarks`  [INFERRED] [semantically similar]
  README.md → docs/final_report.md
- `Three-Layer AI Architecture` --semantically_similar_to--> `Three-Layer AI Pipeline`  [INFERRED] [semantically similar]
  docs/presentation_script.md → README.md
- `get_system_components()` --uses--> `AStarRoadmapOptimizer`  [INFERRED]
  app.py → ai_engine/a_star_roadmap.py
- `TestAStarControlledOptimality` --uses--> `AStarRoadmapOptimizer`  [INFERRED]
  tests/test_all.py → ai_engine/a_star_roadmap.py
- `TestAStarRoadmap` --uses--> `AStarRoadmapOptimizer`  [INFERRED]
  tests/test_all.py → ai_engine/a_star_roadmap.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three Complementary AI Layers** — docs_final_report_machine_learning_classification, docs_final_report_rule_based_expert_system, docs_final_report_a_star_search_algorithm [EXTRACTED 1.00]
- **CareerSense AI Project Documentation** — readme_careersense_ai, docs_final_report_careersense_ai_final_project_report, docs_presentation_script_project_presentation, docs_progress_review_report_stage_2_progress_review_report [INFERRED 0.85]
- **CareerSense AI Python Runtime Stack** — requirements_streamlit, requirements_scikit_learn, requirements_pandas_numpy, requirements_networkx, requirements_plotly [EXTRACTED 1.00]

## Communities (25 total, 12 thin omitted)

### Community 0 - "Product Claims and CI"
Cohesion: 0.05
Nodes (53): Automated unittest Execution, CareerSense AI Test Suite Workflow, Python 3.11 CI Environment, tests/test_all.py Test Suite, A* Search Algorithm, Adaptive Admissible A* Heuristic, Application of Machine Learning Technique in Career Prediction, Artificial Intelligence: A Modern Approach (+45 more)

### Community 1 - "ML Evaluation Tests"
Cohesion: 0.07
Nodes (22): compute_multiclass_metrics(), Trains and evaluates K-NN, Decision Tree, Logistic Regression, Random Forest,…, Computes mathematically rigorous multiclass classification metrics. Guarantees…, Mathematical correctness tests for multiclass evaluation metrics. Replaces…, Verifies that length mismatch between y_true and y_pred raises ValueError., Verifies that empty y_true or y_pred raises ValueError., Verifies that None inputs raise TypeError., Verifies that invalid, duplicate, or empty labels raise ValueError. (+14 more)

### Community 2 - "Web App Orchestration"
Cohesion: 0.10
Nodes (31): get_system_components(), main(), CareerSense AI - Main Application Entrypoint An Explainable AI-Powered Career…, Initializes and caches database, AI models, rule engine, and A* optimizer., cache_resource, seed_database(), datetime, System Administrator & AI Model Explorer View for CareerSense AI. Provides… (+23 more)

### Community 3 - "Project Module Dependencies"
Cohesion: 0.10
Nodes (27): AI Layer 3: A* Search Algorithm for Learning Roadmap Optimization. Constructs…, Explainability & Model Interpretation Engine for CareerSense AI. Provides…, AI Layer 1: Machine Learning Career Classifier for CareerSense AI. Implements:…, AI Layer 2: Rule-Based Expert System for CareerSense AI. Provides transparent…, asyncio, CareerSense AI - Configuration & Constants Defines global configurations,…, Database Management Module for CareerSense AI. Uses SQLite to store…, Seed Data Script for CareerSense AI. Populates the database with system users… (+19 more)

### Community 4 - "SQLite Data Persistence"
Cohesion: 0.10
Nodes (7): DatabaseManager, Initializes all database tables., Replaces or sets module records for student., Cryptographically hashes password using PBKDF2-HMAC-SHA256 with a unique random…, Saves generated roadmap activities., Verifies password against stored_hash in constant time using…, Verifies constant-time verification accepts valid passwords and rejects…

### Community 5 - "Rules and Explainability"
Cohesion: 0.08
Nodes (13): ExplainabilityEngine, Extracts positive drivers and constraining factors influencing the career…, Produces aligned arrays for Plotly radar visualization: - Labels (competencies)…, Generates a comprehensive explainability bundle for a student., Heuristic baseline when ML models are being initialized., Compares student's current proficiency levels with benchmark competencies…, Checks inter-skill dependencies (e.g. Automated Testing requires OOP)., Computes the holistic Career Readiness Score (0% to 100%) broken down into 4… (+5 more)

### Community 6 - "Security and Role Isolation"
Cohesion: 0.06
Nodes (16): Verifies that security invariants, salted PBKDF2 password hashing, legacy…, Verifies PBKDF2-HMAC-SHA256 password hashing generates unique random salts per…, Verifies legacy 64-char unsalted SHA-256 hashes are verified and flagged for…, Verifies authenticate_user transparently rehashes legacy SHA-256 records upon…, Verifies failed login on a legacy account returns None without modifying the…, Verifies authentication for seeded demo accounts and invalid credentials., Verifies parameterized queries prevent SQL injection authentication bypass., Verifies student profile data is isolated by user_id without cross-student data… (+8 more)

### Community 7 - "ML Classification Semantics"
Cohesion: 0.09
Nodes (12): CareerClassifier, Transforms raw student attributes into a feature vector matching…, Loads trained models and saved metrics. Returns True if successful, False…, Generates advisory pathway match scores for a student. Score Semantics &…, ndarray, Verifies that advisory match scores, uncalibrated ML scores, and target goals…, Verifies all decomposed score semantics, backward compatibility aliases, and…, Verifies that ml_predicted_track does not change when target_career parameter… (+4 more)

### Community 8 - "A* Controlled Tests"
Cohesion: 0.09
Nodes (12): Verifies that an activity longer than the weekly budget spans multiple weeks…, Verifies scheduling of multiple consecutive activities within and across weekly…, Verifies that returned total_hours strictly equals the sum of activity costs…, Verifies A* finds a complete optimal solution in a complex graph with…, Rigorous mathematical tests comparing A* search against independent shortest-…, Verifies that an activity spanning multiple weeks does not allow subsequent…, Verifies that when a single activity advances multiple skills, the heuristic…, Verifies that when a target track requires skills absent from the activity… (+4 more)

### Community 9 - "A* Roadmap Engine"
Cohesion: 0.16
Nodes (9): AStarRoadmapOptimizer, Builds the eligible activity set for A* search. Includes every activity that…, State-Space Graph Search Optimizer for Student Upskilling. Formal Problem…, Runs A* search to find the optimal sequence of learning activities that satisfy…, Normalizes activity duration to a strictly positive numeric value. Policy: -…, Simple topological sort fallback to guarantee a valid sequenced roadmap., Formats hours as integer if whole, or float rounded to 2 decimals., heuristic_mode: - "auto": dynamically selects proven admissible lower bound… (+1 more)

### Community 10 - "PDF Report Generation"
Cohesion: 0.16
Nodes (8): reportlab_lib, reportlab_lib_pagesizes, reportlab_lib_styles, reportlab_lib_units, reportlab_pdfgen, reportlab_platypus, NumberedCanvas, Script to generate a publication-quality PDF for CareerSense AI Stage 2…

### Community 11 - "Synthetic Dataset Validation"
Cohesion: 0.25
Nodes (4): Verifies dataset structure and demonstration properties., Verifies that synthetic cohort student profiles are unique., Verifies all 5 career tracks are adequately represented., TestDatasetIntegrity

### Community 13 - "A* Optimality Proofs"
Cohesion: 0.20
Nodes (3): Verifies that when multi-skill activities exist, the bottleneck lower bound is…, Verifies A* selects a 13-hour multi-skill combo activity over separate 10-hour…, Verifies that auto, admissible_max, and zero (Dijkstra) all find the exact same…

## Ambiguous Edges - Review These
- `Final Held-Out Model Benchmarks` → `HTML Stage 2 Model Evaluation Results`  [AMBIGUOUS]
  docs/progress_review_report.html · relation: conceptually_related_to
- `Final Held-Out Model Benchmarks` → `Stage 2 Model Evaluation Results`  [AMBIGUOUS]
  docs/progress_review_report.md · relation: conceptually_related_to

## Knowledge Gaps
- **10 isolated node(s):** `Python 3.11 CI Environment`, `tests/test_all.py Test Suite`, `Rule-Based Expert System`, `Weekly Time-Budgeted Learning Roadmap`, `Application of Machine Learning Technique in Career Prediction` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 167 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Final Held-Out Model Benchmarks` and `HTML Stage 2 Model Evaluation Results`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Final Held-Out Model Benchmarks` and `Stage 2 Model Evaluation Results`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `TestAStarControlledOptimality` connect `A* Controlled Tests` to `Project Module Dependencies`, `A* Roadmap Engine`, `A* Optimality Proofs`, `Duration Clamping`, `Search Exhaustion Fallback`, `Deep Prerequisite Chains`, `Missing Prerequisite Fallback`, `Direct Course Optimization`, `Multi-Step Path Optimization`, `Zero-Gap Handling`, `Weekly Budget Validation`, `Dijkstra Parity Validation`, `Secondary Skill Inclusion`, `Partial Coverage Reporting`?**
  _High betweenness centrality (0.186) - this node is a cross-community bridge._
- **Why does `DatabaseManager` connect `SQLite Data Persistence` to `Web App Orchestration`, `Project Module Dependencies`, `Security and Role Isolation`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `TestSecurityAndRoleIsolation` connect `Security and Role Isolation` to `Project Module Dependencies`, `SQLite Data Persistence`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `DatabaseManager` (e.g. with `get_system_components()` and `render_login_and_registration()`) actually correct?**
  _`DatabaseManager` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `AStarRoadmapOptimizer` (e.g. with `get_system_components()` and `TestAStarControlledOptimality`) actually correct?**
  _`AStarRoadmapOptimizer` has 3 INFERRED edges - model-reasoned connections that need verification._
