# CareerSense AI: Project Idea, Solution Audit, Risks, and Recommendations

**Audit date:** 2026-10-15
**Scope:** Product concept, implementation architecture, model/evaluation validity, roadmap algorithm, data integrity, security boundaries, documentation, and automated tests.

## Executive assessment

CareerSense AI is a well-structured **academic decision-support prototype** for undergraduate computing students. Its core idea is to turn a student profile into three outputs:

1. a ranked set of five computing career pathways;
2. an explainable competency and prerequisite gap analysis; and
3. a dependency-aware learning roadmap scheduled around weekly study time.

The implementation is functional and unusually well tested for a student project. The audit reproduced **74 passing tests**, confirmed that the Streamlit server reaches its ready state, and found a modular design with clear separation among ML classification, expert rules, A* planning, persistence, and role-based views.

The main limitation is **external validity, not basic software correctness**. The classifier is trained and tested on synthetic profiles whose labels are chosen first and whose grades, interests, and skills are then generated conditionally from those labels. The reported 88.82% held-out accuracy therefore measures how well the model recovers the project's own synthetic generation logic; it does not demonstrate real-world career suitability, employment outcomes, or benefit to students. The final user-facing composite score is also not the same model that was evaluated.

**Overall judgement:** strong educational prototype; not yet suitable for consequential student guidance without real-data validation, evidence-governance changes, and tighter deployment controls.

## 1. What problem the project addresses

The project targets the weakness of generic career advice for computing undergraduates. It assumes that useful guidance should consider:

- completed coursework and grades;
- current technical skills;
- domain interests and intended career direction;
- competency prerequisites and gaps;
- time available for further study; and
- human advisor oversight.

This is a sensible problem framing. Career guidance is not only a classification task: a student also needs to understand *why* a path may fit, what is missing, and what to do next.

## 2. How the solution works

```text
Student profile
  (grades, skills, interests, degree, target, time)
        |
        +--> ML layer: K-NN + Decision Tree scores five pathways
        |
        +--> Rule layer: prerequisites, gaps, readiness, explanations
        |
        +--> A* layer: minimum-hour activity sequence + weekly schedule
        |
        +--> SQLite: profile, roadmap, progress, notes
        |
        +--> Student / advisor / coordinator / admin dashboards
```

### Layer 1: pathway ranking

`CareerClassifier` extracts 36 grade, interest, and skill features, then combines K-NN and Decision Tree outputs. The application further combines that ML score with target aspiration, domain interests, and degree priors to create a normalized advisory affinity ranking. The UI correctly warns that this score is not an employment probability.

### Layer 2: rule-based decision support

`RuleBasedExpertSystem` checks academic prerequisites, compares current skills with career benchmarks, checks a small set of skill dependencies, and calculates a four-part readiness index covering competencies, academics, projects, and documents.

### Layer 3: A* learning plan

`AStarRoadmapOptimizer` treats learning activities as actions with prerequisite edges and study-hour costs. It uses an admissible heuristic, searches for the minimum-hour sequence that can close represented skill gaps, and schedules the result across weeks. If full graph coverage is impossible, it labels the output as partial or falls back to topological ordering.

### Application and users

The Streamlit application provides dashboards for students, advisors, coordinators, and administrators. SQLite stores profiles, self-assessments, roadmaps, progress, document metadata, and advisor notes.

## 3. What is done well

| Area | Evidence-backed strength |
|---|---|
| Architecture | Clear separation between ML, rules, planning, explanations, persistence, and views. No import cycles were detected. |
| Evaluation mechanics | Train/test stratification and fold-local scaling are implemented correctly for the evaluated estimators. |
| Baselines | The project compares K-NN against dummy, tree, logistic regression, and random forest baselines instead of reporting one isolated score. |
| Score disclosure | The source explicitly says affinity scores are uncalibrated and are not employment probabilities. |
| A* correctness | Tests cover admissibility, optimal-cost parity with Dijkstra, multi-skill activities, prerequisites, missing coverage, search limits, fractional durations, and weekly scheduling. |
| Security basics | Passwords use salted PBKDF2, SQL queries are parameterized, malformed hashes fail safely, and legacy hashes migrate after valid login. |
| Transparency | The README discloses that the dataset is synthetic and the roadmap reports partial coverage rather than falsely marking every result optimal. |
| Automated verification | All 74 tests passed locally in 64.692 seconds. |

## 4. Key issues and risks

### High: synthetic accuracy is being asked to support a real guidance claim

The generator selects a `true_track` from degree priors and then generates grades, interests, and skills based on that selected track. The classifier is later evaluated on a random split of those generated records. This is not direct target-column leakage, and the fold-local scaling is correct, but it is still a **closed synthetic loop**: the evaluation rewards reconstruction of assumptions embedded in the generator.

Consequences:

- 88.82% accuracy cannot be interpreted as career-guidance accuracy.
- The split is not independent at the level that matters: all rows come from the same generator, seed logic, priors, and feature distributions.
- There is no measured outcome such as student satisfaction, persistence, internship attainment, advisor agreement, or later role fit.
- There is no subgroup or fairness evaluation across degree, year, gender, socioeconomic background, disability, or other relevant groups.

**Recommendation:** describe the result as *synthetic benchmark recovery*, not validated career prediction. Obtain consented, de-identified, multi-cohort data; define an outcome with advisors and students; reserve temporal or institution-level external validation; and report calibration and subgroup error, not accuracy alone.

### High: the deployed composite recommendation is not the evaluated model

The benchmark evaluates individual classifiers, but production ranking uses:

- 70% K-NN + 30% Decision Tree for the ML component;
- an additional domain-interest component;
- a degree prior; and
- up to 45% direct target-aspiration weight in cold-start mode.

That final composite is never evaluated against a held-out outcome. There are also important double-counting paths:

- interests are already five ML input features and are then added again as a separate weighted component;
- registration raises the interest associated with the selected target to 5, while target aspiration is also separately weighted;
- the synthetic label is sampled using degree priors, and degree priors are again added at inference time.

This design can turn guidance into confirmation: the student chooses a target, the system increases the matching interest, and the final score strongly favors that target.

**Recommendation:** keep competency evidence, stated preference, and curriculum prior as separate displayed dimensions. If one ranking is required, pre-register or learn its weights on an independent validation set and evaluate the exact final formula. Add sensitivity tests showing how rankings change when only target, degree, or interest changes.

### High: “pure ML competency” still includes self-reported interests

The documentation says the pure ML path is independent of aspiration and evaluates coursework and skills, but `ML_FEATURE_COLUMNS` includes all five interest ratings. The result is independent of the `target_career` parameter, but not independent of preferences or aspirations expressed through interests.

**Recommendation:** either rename it to “profile-based ML alignment” or train a genuinely competency-only model using verified grades and skills. Show both competency and preference views instead of implying objectivity.

### High: evidence is marked verified without verification

The document workflow validates extension and size but does not persist or parse the uploaded bytes. It writes metadata with `"verified": True` immediately. Project entries are self-reported, and completing a roadmap item automatically increases a skill level without an assessment or advisor approval. The increment also ignores the activity's declared `skill_level_gain`; an Intermediate student completing any activity can be moved to Advanced.

The readiness index then awards points for document and project counts and can label a student “Job / Internship Ready.” This makes a consequential label depend on unverified counts and self-reported state.

**Recommendation:** until verification exists, rename fields and UI text to “self-reported” or “submitted.” Do not set `verified=True`. Store documents securely only if there is a clear need and retention policy; otherwise store neither files nor verification claims. Require evidence or advisor approval before skill promotion, and use the activity's explicit gain level. Replace “Job / Internship Ready” with “Prototype Readiness Index” until validated against outcomes.

### High for deployment: model artifacts are not reproducibly compatible

The committed artifacts record Python 3.14.5, scikit-learn 1.9.1, NumPy 2.4.6, and pandas 3.0.3. `requirements.txt` only supplies lower bounds. On the available Python 3.9 runtime it installed scikit-learn 1.6.1, and every model load emitted `InconsistentVersionWarning`. Pickled scikit-learn objects are not a stable cross-version model format.

**Recommendation:** choose and document supported Python versions; use a lock file with exact compatible versions; retrain artifacts in the supported environment; verify the stored dataset/model hash at load; fail clearly on incompatible provenance; and consider a safer, versioned export format where practical. Add a CI matrix matching the claimed Python range.

### Medium-high: every career roadmap is partial for a zero-skill student

The current learning graph does not cover every required competency:

| Career track | Required | Covered | Missing from activity graph |
|---|---:|---:|---|
| Software Engineering | 7 | 4 | OOP, Git, SQL |
| Data Science / AI | 6 | 4 | SQL, OOP |
| Cybersecurity | 6 | 4 | Secure Coding, Monitoring & Logging |
| Cloud / DevOps | 6 | 5 | Monitoring & Logging |
| UI/UX Design | 5 | 3 | Design Principles, Frontend Awareness |

The implementation honestly returns `partial_coverage`, but the product still presents the feature as an optimized complete roadmap. Also, “optimal” only means minimum declared study hours among represented activities. It does not optimize learning effectiveness, cost, availability, difficulty, accessibility, deadlines, or student preference. Weekly hours schedule the chosen plan after search; they do not change which path is selected.

**Recommendation:** complete the activity graph or explicitly classify some competencies as verified curriculum prerequisites. Version and review resource links and estimated hours. Consider multi-objective planning and make the narrow definition of optimality visible beside every plan.

### Medium: coordinator recommendations are hard-coded, not analytical

The coordinator page always states “60%+” pathway demand, “over 45%” testing deficiency, and an “8.5 hours” average regardless of the database. In the current demo database there is one student, the average is 8.0 hours, and no such percentages can be supported.

**Recommendation:** compute each statement from the current aggregate with minimum-cohort thresholds and suppress small-group insights to protect privacy. If the text is illustrative, label it “example insight.”

### Medium: SQLite foreign keys are declared but disabled

`get_connection()` opens SQLite without `PRAGMA foreign_keys = ON`. Runtime verification returned `foreign_keys = 0`, so declared cascades and referential constraints are not enforced.

**Recommendation:** enable and verify foreign keys for every connection and add orphan/cascade tests.

### Medium: authorization is concentrated in the UI layer

Data methods accept arbitrary user or roadmap IDs and do not enforce actor identity or role. This is workable for a single-process local demonstration, but it is not a safe service boundary. Advisor access also covers every student, not assigned advisees.

**Recommendation:** add an authorization-aware service layer, scope mutations by both record ID and authenticated user ID, model advisor assignments, and keep demo persona switching unavailable outside an explicit demo configuration.

### Medium: generated HTML is not escaped

Names, registration numbers, gap text, and roadmap values are interpolated directly into downloadable HTML. User-controlled HTML can therefore be embedded in the report.

**Recommendation:** HTML-escape all dynamic text and strictly validate numeric/style values before interpolation.

### Medium-low: tests emphasize internal invariants over product validity

The suite is strong for A* mathematics, metric formulas, password handling, and data isolation. However, many tests check structure or consistency rather than whether the career advice is correct or beneficial. There are no tests for:

- the exact final composite ranking against an independent target;
- abstention on insufficient evidence;
- fairness or subgroup performance;
- hard-coded coordinator claims;
- false document verification;
- the readiness-score CV edge case; or
- incompatible serialized-model versions.

This is not a criticism of the existing 74 tests; it identifies what a larger test count cannot prove.

## 5. Specific correctness issue in the readiness calculation

If `cv_filename` exists in the profile but the document table is empty, the formula starts with five CV points and then subtracts 2.5 because `doc_count - 1` is negative. A present CV can therefore receive only 2.5/10 rather than the intended 5/10.

**Recommendation:** count non-CV documents with `max(0, ...)`, or derive a distinct set of verified document types before scoring.

## 6. Recommended improvement plan

### Phase 1 — make claims match current evidence

1. Rename “career prediction” to “career pathway alignment prototype.”
2. Rename the readiness tier and remove employment-readiness claims.
3. Clearly label grades, skills, projects, and documents as self-reported unless verified.
4. Separate competency, interest, degree context, and aspiration in the UI.
5. Replace hard-coded coordinator insights with calculated, privacy-thresholded values.

### Phase 2 — fix technical reliability

1. Pin Python and dependencies; rebuild models in that environment.
2. Validate artifact provenance before loading.
3. Enable SQLite foreign keys and add authorization-aware services.
4. Respect `skill_level_gain`; require evidence/approval for promotion.
5. Escape generated HTML.
6. Add tests for the concrete issues in this report.

### Phase 3 — improve recommendation validity

1. Define the decision the system supports: exploration, course planning, internship preparation, or role prediction. These require different targets.
2. Create an advisor-reviewed competency rubric with versioned sources and review dates.
3. Evaluate the exact production composite, including ablation and sensitivity tests.
4. Add an abstain/low-evidence state instead of always ranking five paths.
5. Validate with multiple cohorts and measure calibration, subgroup errors, and human outcomes.

### Phase 4 — strengthen the roadmap

1. Cover all required competencies or distinguish coursework from optional activities.
2. Add resource availability, cost, difficulty, modality, expiration/review dates, and accessibility.
3. Let students optimize for more than hours, and show trade-offs rather than one supposedly best plan.
4. Re-plan after verified progress while preserving an audit trail of why the plan changed.

## 7. Verification performed

- Graphify scan: 30 supported files, approximately 45,237 words.
- Knowledge graph: 392 nodes, 621 edges, 25 communities.
- Graph benchmark: approximately 3.7x fewer tokens per representative query.
- Graph integrity warning: 77 dangling-endpoint edges, 2 self-loops, and 4 endpoint-pair collapses; all important conclusions above were checked directly against source files.
- Automated tests: **74 passed** in **64.692s**.
- Runtime: Streamlit reached its ready state on a local test port.
- Model compatibility: repeated scikit-learn version warnings reproduced.
- Roadmap coverage: all five zero-skill scenarios reproduced as `partial_coverage`.
- Database integrity setting: `PRAGMA foreign_keys` reproduced as `0`.

## 8. Final recommendation

Keep the three-layer design. It is the project's best idea: prediction/alignment, transparent rules, and actionable planning solve different parts of the student problem. But position the current system as an **explainable exploration and planning prototype**, not a validated predictor of career success.

The most valuable next step is not another algorithm. It is to validate the decision framework with real students and advisors, evaluate the exact production score, and introduce evidence quality and abstention. Those changes would improve both scientific credibility and student safety more than optimizing the already well-tested A* implementation.

## Source pointers

- Product claims and stated limitations: `README.md:22`, `README.md:154`, `README.md:174`
- Synthetic label and feature generation: `scripts/generate_dataset.py:82`
- ML features and degree priors: `config.py:111`, `config.py:243`
- Training and evaluation: `ai_engine/ml_classifier.py:269`
- Production score composition: `ai_engine/ml_classifier.py:510`
- Rule/readiness formula: `ai_engine/rule_engine.py:176`
- A* objective and partial coverage: `ai_engine/a_star_roadmap.py:18`, `ai_engine/a_star_roadmap.py:246`
- Document metadata and automatic skill promotion: `modules/student_view.py:315`, `modules/student_view.py:552`
- Hard-coded coordinator insights: `modules/coordinator_view.py:88`
- SQLite connection and record mutation boundaries: `database/db_manager.py:29`, `database/db_manager.py:442`
- Unescaped report interpolation: `modules/report_generator.py:18`
- Dependency policy and stored provenance: `requirements.txt:1`, `ai_engine/saved_models/evaluation_metrics.json:273`
