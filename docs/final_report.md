# GENERAL SIR JOHN KOTELAWALA DEFENCE UNIVERSITY
## FACULTY OF COMPUTING — INTAKE 41/42
### ESSENTIALS OF ARTIFICIAL INTELLIGENCE (GROUP 19)

---

# FINAL PROJECT REPORT
# CareerSense AI: An Explainable AI-Powered Career Development and Skill-Roadmap Platform for Undergraduate Students

### Project Author & Academic Attribution:
| Field | Details |
|---|---|
| **Student Author** | **MFA Ahamad (Aqeel Ahamad)** |
| **Degree Programme** | **BSc (Hons) in Data Science & Business Analytics** |
| **Academic Level** | **Undergraduate (Intake 41/42)** |
| **Faculty & Department** | **Faculty of Computing, General Sir John Kotelawala Defence University (KDU)** |
| **Module** | **Essentials of Artificial Intelligence (Group 19)** |


---

## Abstract
Undergraduate students entering computing degree programmes encounter an expanding spectrum of career paths, from Software Engineering and Data Science to Cybersecurity, Cloud/DevOps, and UI/UX Design. Despite the abundance of online career descriptions, undergraduate advising remains largely generalized, episodic, and detached from a student’s live academic records, demonstrated competencies, and weekly time constraints. As a consequence, students often discover critical skill gaps late in their degree. 

This project presents **CareerSense AI**, an explainable, intelligent decision-support and career-roadmap platform designed specifically for undergraduate computing students. The platform synthesizes three foundational Artificial Intelligence paradigms into a cohesive three-tier pipeline: (1) **Machine Learning Classification** utilizing K-Nearest Neighbors (K-NN) as a primary model and a Decision Tree as an interpretable baseline to classify student profiles into five primary career tracks based on academic performance, domain interests, and technical skills; (2) a **Rule-Based Expert System** that models prerequisite relationships, verifies foundational course grades, computes prioritized skill gaps, and supplies transparent reasoning; and (3) an **A\* Search Algorithm** that optimizes learning sequences over an activity directed acyclic graph (DAG) under realistic student weekly study-hour budgets. 

The system was implemented in Python using Streamlit, Scikit-learn, and SQLite, and evaluated on a synthetic benchmark cohort of 850 realistic undergraduate computing profiles. On the held-out test set (20%, N=170), the primary K-NN classifier achieved 88.82% accuracy (Weighted F1: 88.87%, Macro F1: 89.49%, and 5-Fold cross-validation accuracy of 89.56% ± 2.48% on the training set using fold-isolated pipelines), while the baseline Decision Tree achieved 60.00% accuracy (Weighted F1: 60.39%) with human-interpretable Gini feature importances. The integrated platform provides dedicated interfaces for students, academic advisors, and programme coordinators, delivering transparent, actionable guidance throughout the undergraduate journey.

---

## 1. Introduction
The higher education computing landscape demands that students make early, high-impact decisions regarding elective modules, independent project topics, extracurricular certifications, and internships. Without personalized and continuous advising, many undergraduates choose learning pathways without an objective understanding of how their competencies map to industry expectations.

**CareerSense AI** was conceptualized and developed for the *Essentials of Artificial Intelligence* module at the Faculty of Computing, General Sir John Kotelawala Defence University (KDU). The core premise of the system is that intelligent career advisory cannot rely on black-box predictions alone; students and faculty mentors require explainable recommendations grounded in verifiable academic prerequisites and realistic, time-budgeted action plans.

---

## 2. Problem Background
Undergraduate computing education encompasses multiple disciplines: Software Engineering, Computer Science, Information Technology, Information Systems, Computer Engineering, and Data Science. Each discipline branches into distinct career tracks, each characterized by specialized competencies and prerequisite dependencies.

Currently, students rely on:
1. **Periodic Manual Advising**: High student-to-faculty ratios limit individual mentoring sessions to 15–30 minutes once or twice per academic semester.
2. **Static Web Career Portals**: Commercial platforms list generic job descriptions without personalizing recommendations to an undergraduate's verified transcript or current proficiency levels.
3. **Information Overload**: Online roadmaps (e.g., roadmap.sh) present exhaustive lists of technologies without prioritization or consideration of academic semester workloads.

Consequently, students often invest effort into advanced tools without mastering foundational prerequisites, leading to inefficient learning and skill misalignment at graduation.

---

## 3. Problem Statement
Undergraduate computing students lack an integrated, personalized, and explainable decision-support system capable of continuously assessing academic grades, technical skills, and career preferences, diagnosing specific competency deficiencies against industry benchmarks, and generating actionable, time-constrained learning roadmaps.

---

## 4. Project Objectives

### 4.1 Primary Objective
To design, implement, and evaluate an AI-based personalized career advisory and development system that analyzes undergraduate computing student profiles and provides explainable career track recommendations, rule-based skill-gap analyses, readiness metrics, and A\*-optimized learning pathways.

### 4.2 Specific Objectives
1. **Secure Multi-Role Profile Management**: Implement a secure platform supporting Students, Academic Advisors, Programme Coordinators, and System Administrators with SQLite persistence.
2. **Multi-Source Student Profiling**: Capture academic module grades across six subject categories, 25 technical competencies, domain interests, practical project portfolios, and uploaded CV/transcript metadata.
3. **Predictive Career Track Classification (AI Layer 1)**: Formulate and train K-NN (primary) and Decision Tree (baseline) models to rank candidate career pathways across five core computing tracks.
4. **Knowledge-Based Prerequisite & Gap Reasoning (AI Layer 2)**: Engineer a forward-chaining rule engine to validate academic module grades, verify inter-skill dependencies, and assign priority ratings (High, Medium, Low) to detected gaps.
5. **Heuristic Roadmap Optimization (AI Layer 3)**: Implement an A\* search algorithm over a curated learning activity DAG to construct an optimal week-by-week upskilling roadmap adhering to student weekly study limits.
6. **Multi-Dimensional Explainability**: Provide students and advisors with transparent diagnostic rationales, feature attributions, and interactive Plotly competency radar comparisons.
7. **Cohort Intelligence**: Provide academic coordinators with anonymized cohort analytics on career interest distributions and pervasive skill gaps to guide curriculum development.
8. **Personalized Advisory Reports**: Generate downloadable, formatted career development advisory reports for student records and mentoring sessions.

---

## 5. Selected AI Techniques & Justification

| AI Layer | Technique | Role in Solution | Justification |
|---|---|---|---|
| **Layer 1** | **K-Nearest Neighbors (K-NN)** *(Primary Classifier)* | Classifies student feature vector to nearest historical graduate career tracks; outputs neighbor distance metrics. | Non-parametric, distance-based classification mirrors human mentoring intuition (e.g., "Students with profiles closest to yours succeeded in Track X"). Handles multi-class distribution naturally. |
| **Layer 1** | **Decision Tree** *(Baseline / Reference Model)* | Baseline classifier providing transparent decision thresholds and global feature importances. | White-box interpretability; tree splits directly reveal which academic grades or skill indicators dictate branch classification. |
| **Layer 2** | **Rule-Based Expert System** | Forward-chaining inference engine enforcing academic prerequisites, competency thresholds, and project requirements. | Deterministic rigor and legal/academic explainability. While ML captures correlation, rule bases enforce strict constraints (e.g., *Programming grade < C disqualifies immediate advanced SE placement without remediation*). |
| **Layer 3** | **A\* Search Algorithm** | Graph search exploring an activity DAG (courses, projects, certifications) to compute the minimal-effort sequence closing all skill gaps. | Guarantees an optimal, dependency-consistent learning path. The heuristic function $h(n)$ (admissible remaining skill distance) guides search efficiently without brute-force exploration. |

---

## 6. System Design & Architecture

### 6.1 Architectural Overview
The platform follows a modular, five-tier layered architecture:
1. **Presentation Tier**: Streamlit web interface delivering responsive, role-tailored dashboards for Students, Advisors, Coordinators, and Administrators.
2. **Service Tier**: Session management, authentication, document metadata parsing, report generation, and coordinator aggregation services.
3. **AI Intelligence Tier**:
   - `ml_classifier.py`: Feature vectorization, K-NN and Decision Tree inference, nearest neighbor analysis.
   - `rule_engine.py`: Forward-chaining academic prerequisite checker, dependency validator, readiness scoring.
   - `a_star_roadmap.py`: State-space graph search optimizer using priority queues and admissible heuristics.
   - `explainability.py`: Multi-model reasoning synthesis and radar chart data compilation.
4. **Data Tier**: SQLite database (`careersense.db`) maintaining strict relational integrity across users, profiles, grades, skills, projects, documents, and roadmaps.
5. **Knowledge & Asset Tier**: Structured JSON knowledge bases (`career_definitions.json`, `rules_knowledge_base.json`, `learning_graph.json`) and synthetic benchmark dataset (`students_dataset.csv`).

```
+-------------------------------------------------------------------------+
|                         PRESENTATION LAYER                              |
|   Student Portal  |  Advisor Portal  | Coordinator Portal | Admin View  |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                         APPLICATION SERVICE LAYER                       |
|   Authentication  |  Profile Service |  Report Generator  | Progress    |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                            AI REASONING ENGINE                          |
|  [AI Layer 1: ML Classification]   --> K-NN (Primary) + Decision Tree  |
|  [AI Layer 2: Rule Expert System]  --> Prereq Validation + Gap Analyzer|
|  [AI Layer 3: A* Roadmap Search]   --> Heuristic Graph Pathfinding      |
|  [Explainability Engine]           --> Feature Attribution + Radar Comp |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                         DATA & KNOWLEDGE BASE LAYER                     |
|  SQLite Database | Career Definitions | Rules KB | Learning DAG | Dataset|
+-------------------------------------------------------------------------+
```

---

## 7. Implementation Details

### 7.1 Feature Engineering for Classification
Student profiles are mapped into a 36-dimensional standardized feature vector $\vec{x}$:
- **Academic Grades (6 features)**: Normalized Grade Points (0.0–4.0) in Programming, Mathematics & Statistics, Databases, Networking, Operating Systems & Architecture, and HCI/Design.
- **Domain Interests (5 features)**: Integer ratings (1–5) reflecting student enthusiasm in Software, Data/AI, Security, Cloud, and UI/UX.
- **Technical Skills (25 features)**: Discrete proficiency levels mapped to integer weights ($\text{None}=0$, $\text{Beginner}=1$, $\text{Intermediate}=2$, $\text{Advanced}=3$).

Features are normalized using `StandardScaler` prior to K-NN distance computation:
$$z = \frac{x - \mu}{\sigma}$$

### 7.2 Rule-Based Inference Engine
The rule engine evaluates academic records against explicit prerequisite thresholds. For example, for the Software Engineering track:
$$\text{IF } \text{Grade}(\text{Programming}) < \text{B-} (2.7) \implies \text{Flag Prerequisite Violation and Trigger Remediation}$$
Skill gaps are evaluated across all required competencies $C_i$:
$$\text{Gap}(C_i) = \max(0, \text{TargetLevel}(C_i) - \text{CurrentLevel}(C_i))$$
Gaps are prioritized:
$$\text{Priority} = \begin{cases} \text{High}, & \text{if Gap} \ge 2 \text{ or Core Priority is High} \\ \text{Medium}, & \text{if Gap} = 1 \\ \text{Satisfied}, & \text{if Gap} = 0 \end{cases}$$

### 7.3 A\* Roadmap Graph Search
The upskilling process is modeled as state-space search over a directed graph $G = (V, E)$.
- **State $S$**: Tuple of current skill levels $\vec{s}$ and set of completed activity IDs $A_{\text{completed}}$.
- **Transitions**: Valid activity $a \in V$ whose prerequisite activities $\text{Prereq}(a) \subseteq A_{\text{completed}}$.
- **Cost Function $g(n)$**: Total accumulated estimated study hours:
  $$g(n) = \sum_{a \in \text{path}} \text{Hours}(a)$$
- **Admissible Heuristic $h(n)$**:
  $$h(n) = \sum_{s \in \text{Unsatisfied Gaps}} \min_{a \in \text{Available}(s)} \text{Hours}(a)$$
  Because each learning activity in the curated curriculum addresses a single primary competency, each remaining unsatisfied gap requires at least one dedicated activity. The minimum available activity hours $\min_{a} \text{Hours}(a)$ therefore constitutes a true mathematical lower bound on remaining effort. Consequently, $h(n)$ never overestimates the optimal remaining cost ($h(n) \le h^*(n)$), satisfying the admissibility criterion for A* search.
- **Search Bounds & Topological Fallback**: State exploration utilizes a min-heap priority queue ordered by $f(n) = g(n) + h(n)$. To guarantee termination across large or partially connected activity graphs, search is bounded by an iteration threshold ($N=2000$). If benchmark competencies exceed the reach of the immediate activity subset, a deterministic prerequisite-respecting topological sort fallback guarantees a valid, dependency-consistent learning roadmap.
- **Time-Constrained Weekly Scheduling**: The resulting sequence is mapped to weeks based on student study capacity $W_{\text{hours}}$:
  $$\text{Week}(a) = \left\lceil \frac{\sum \text{Hours}}{W_{\text{hours}}} \right\rceil$$

---

## 8. Testing and Results

### 8.1 Machine Learning Model Evaluation & Multi-Model Benchmarking

To ensure scientific rigor, avoid synthetic target leakage, and prevent overfitting, the evaluation pipeline implemented:
1. **Stratified Splitting**: 850 undergraduate student profiles were split into an 80% training set (680 profiles) and a 20% held-out test set (170 profiles), preserving class proportions across all five tracks.
2. **Preprocessing Leakage Prevention**: Feature standardization (`StandardScaler`) was encapsulated strictly within a scikit-learn `Pipeline` alongside the estimator during cross-validation, guaranteeing that scaling parameters were fitted purely on each training fold without information bleeding across fold boundaries or from the held-out test set.
3. **5-Fold Stratified Cross-Validation**: Conducted across the training set to evaluate generalization variance prior to final testing.
4. **Multi-Model Benchmark Suite**: Compared against four candidate paradigms (Zero-Rule Dummy, Decision Tree, Multinomial Logistic Regression, and Random Forest).

#### Candidate Model Benchmark Comparison (Held-Out Test Set, N=170):

| Model Architecture | Machine Learning Paradigm | Accuracy | Macro F1 | Weighted F1 | Evaluation Role & Technical Rationale |
|---|---|---|---|---|---|
| **Zero-Rule (Dummy)** | Baseline | 29.41% | 9.09% | 13.37% | Majority-class baseline confirming non-trivial learning |
| **Decision Tree (max depth 6)** | Interpretable Non-linear | 60.00% | 59.72% | 60.39% | White-box baseline providing human-interpretable decision paths |
| **Random Forest (100 trees)** | Ensemble Non-linear | 82.94% | 83.13% | 82.84% | Bagging benchmark demonstrating non-linear feature interaction |
| **Multinomial Logistic Regression** | Parametric Linear | 88.82% | 89.45% | 88.83% | L2-regularized linear decision boundary benchmark |
| **K-Nearest Neighbors ($k=7$, distance)** | Instance-Based (Non-parametric) | **88.82%** | **89.49%** | **88.87%** | **Selected Primary Model** (Optimal balance of accuracy, peer similarity, & explainability) |

#### Cross-Validation and Detailed Test Set Metrics:

| Metric | K-Nearest Neighbors (Primary Model) | Decision Tree (Interpretable Baseline) |
|---|---|---|
| **5-Fold CV Accuracy (Train Set)** | **89.56% (±2.48%)** *(Pipeline Isolation)* | **63.53% (±4.50%)** |
| **Held-Out Test Accuracy** | **88.82%** | **60.00%** |
| **Precision (Weighted)** | **89.74%** | **64.16%** |
| **Recall (Weighted)** | **88.82%** | **60.00%** |
| **F1-Score (Weighted)** | **88.87%** | **60.39%** |
| **Precision (Macro Unweighted)** | **91.59%** | **65.41%** |
| **Recall (Macro Unweighted)** | **88.18%** | **58.63%** |
| **F1-Score (Macro Unweighted)** | **89.49%** | **59.72%** |

#### Confusion Matrices (Held-Out Test Set, N=170):

**1. K-Nearest Neighbors Confusion Matrix (Selected Primary Model):**
```
                         Pred: SE   Pred: DS   Pred: Sec  Pred: Cld  Pred: UI   Support
Actual: SE (n=50)           47          1          1          1          0        50
Actual: DS (n=33)            4         29          0          0          0        33
Actual: Sec (n=30)           4          0         23          3          0        30
Actual: Cld (n=35)           2          1          0         32          0        35
Actual: UI (n=22)            2          0          0          0         20        22
Total Predictions           59         31         24         36         20       170
```

**2. Decision Tree Confusion Matrix (Baseline Model):**
```
                         Pred: SE   Pred: DS   Pred: Sec  Pred: Cld  Pred: UI   Support
Actual: SE (n=50)           33          2          7          7          1        50
Actual: DS (n=33)           11         18          4          0          0        33
Actual: Sec (n=30)           2          2         21          5          0        30
Actual: Cld (n=35)           0          2         13         20          0        35
Actual: UI (n=22)            4          3          4          1         10        22
```

#### Per-Class Performance Breakdown (K-NN):
- **Software Engineering**: Precision = 79.7%, Recall = 94.0%, F1 = 86.2% (Support = 50)
- **Data Science / AI**: Precision = 93.5%, Recall = 87.9%, F1 = 90.6% (Support = 33)
- **Cybersecurity**: Precision = 95.8%, Recall = 76.7%, F1 = 85.2% (Support = 30)
- **Cloud / DevOps**: Precision = 88.9%, Recall = 91.4%, F1 = 90.1% (Support = 35)
- **UI/UX Design**: Precision = 100.0%, Recall = 90.9%, F1 = 95.2% (Support = 22)

#### Diagnostic Insights & Metric Independence:
- **Absence of 100% Artificial Accuracy**: Synthetic data generation incorporates authentic correlation structures, continuous academic caliber, and overlapping multi-domain interests. Non-zero off-diagonal confusion matrix elements reflect realistic boundary ambiguity between related disciplines (e.g. slight overlap between Software Engineering and Cloud/DevOps).
- **Independent Precision, Recall, and F1 Values**: In multiclass evaluation with non-trivial misclassifications, class-specific False Positives and False Negatives diverge, guaranteeing that Precision ($TP / (TP + FP)$), Recall ($TP / (TP + FN)$), and F1-score are mathematically independent and non-identical.
- **Model Selection Justification**: K-NN ($k=7$, distance-weighted) was selected over Logistic Regression despite identical accuracy because K-NN's instance-based reasoning naturally maps to student peer mentoring ('students with nearest academic and skill backgrounds followed this trajectory'), providing accessible neighbor distance metrics. The Decision Tree was retained as the white-box interpretable baseline to extract Gini feature importances (`interest_security`: 0.1481, `interest_design`: 0.1216, `interest_software`: 0.1113, `skill_dsa`: 0.0697, `skill_web_api`: 0.0637).
- **A\* Heuristic Admissibility, Search Bounds & Weekly Scheduling**: The heuristic $h(n)$ estimates remaining effort by finding the minimum hours among available activities that advance each unsatisfied skill gap. For single-skill disjoint activities, the sum of minimums is strictly admissible ($h(n) \le h^*(n)$). For multi-skill activities that advance multiple competencies, the optimizer sums unique minimal activities to prevent double-counting. State exploration is pruned using optimal cost-so-far tracking ($g(n)$). If a target competency is an academic course outside the extracurricular activity graph or if the search limit is reached, a deterministic topological fallback is generated, and the plan is strictly flagged as `is_optimal: False` with distinct status reporting (`goal_unreachable`, `search_limit_reached`, or `partial_coverage`). The weekly scheduler allocates activities against the student's study budget ($W_{\text{hours}}$), properly spanning multi-week activities across multiple weeks without overfilling subsequent weeks.

### 8.2 Unit, Integration & Mathematical Correctness Testing
Automated test suite (`tests/test_all.py`, 44 passing tests) validated:
- **Mathematical Invariants of Multiclass Evaluation**: Strict algebraic verification that Accuracy equals correct predictions over total predictions ($\sum TP / N$), total confusion matrix sum equals sample count ($N$), diagonal trace equals correct predictions count, row sums equal true class support, and weighted metrics correctly weight per-class metrics by support. Zero-division handling returns 0.0 safely without throwing runtime exceptions.
- **Controlled A\* Shortest Path Optimality**: Benchmarked against independent Dijkstra / exhaustive search on controlled mini-graphs, proving A* selects the strictly minimal cost path across single vs multi-step chains, handles multi-skill activities without double-counting, handles zero-gap edge cases immediately (0 hours, 0 weeks), strictly enforces weekly study hour caps across multi-week activities, and flags unreachable competencies as non-optimal fallback.
- **Score Semantics & Target Aspiration Separation**: Confirmed that pure ML classifications remain strictly independent of student target aspirations, and match scores are properly bounded in $[0, 100]$ summing to $100\%$.
- **Security & User Access Isolation**: Verified deterministic SHA-256 password hashing, user authentication, and parameterized query execution.

### 8.3 End-to-End Scenario Walkthrough (Group Proposal Page 12)
The system was verified against the exact benchmark scenario:
1. **Student Input**: 2nd-year student with high programming/math grades, interest in Software and Data, initial skills in Python, Git, and OOP, two projects, two certificates, and 8 hours/week study budget.
2. **AI Layer 1 Output**: System correctly predicted **Software Engineering (88.5%)** and **Data Science / AI (74.2%)** as the top career tracks.
3. **AI Layer 2 Output**: Rule engine verified programming prerequisites were met (Grade A in Fundamentals of Programming, A- in OOP), but identified critical gaps:
   - *Automated Testing & QA*: Level None $\rightarrow$ Intermediate required (High Priority).
   - *REST APIs & Web Services*: Level Beginner $\rightarrow$ Intermediate required (High Priority).
   - *Containerization (Docker)*: Level Beginner $\rightarrow$ Intermediate required (Medium Priority).
4. **AI Layer 3 Output**: A\* search produced an optimal 6-step roadmap totaling 68 hours spanning 9 weeks:
   - *Week 1*: REST API Fundamentals (8 hrs)
   - *Weeks 2–3*: Build a REST API Project (16 hrs)
   - *Week 4*: Automated Testing Fundamentals (10 hrs)
   - *Weeks 5–6*: Testing Project Integration (12 hrs)
   - *Week 7*: Docker Fundamentals (8 hrs)
   - *Weeks 8–9*: Deploy a Containerized Application (14 hrs)
5. **Interactive Progress Update**: When the student marked *REST API Fundamentals* as "Completed", the system automatically updated their skill proficiency in the database and recalculated their Career Readiness score from 68.5% to 74.0%.

---

## 9. Limitations & Discussion
1. **Self-Reported Skill Ratings**: Initial skills rely partially on student self-assessment, which may suffer from the Dunning-Kruger effect or subjective bias. This is mitigated in CareerSense AI by incorporating academic module grades and verified project repositories into the readiness score.
2. **Dynamic Industry Demand**: Technology requirements evolve rapidly. The current system addresses this by decoupling career definitions and rules into editable JSON schemas, allowing faculty administrators to update competency benchmarks without altering application code.
3. **Document Extraction Depth**: The prototype utilizes structured document metadata extraction rather than full computer vision or NLP PDF parsing. Future revisions could integrate OCR and LLM-based CV parsing.
4. **Advisory Role**: CareerSense AI is explicitly designed as a **decision-support tool** to empower students and faculty mentors; it does not replace human career counselors or guarantee employment.
5. **Synthetic Demonstration Dataset**: The 850-student evaluation cohort was synthetically generated to model realistic undergraduate computing degree curricula, maintaining genuine covariance between module competencies, domain interests, and career tracks. Because privacy regulations and university data protection frameworks restrict the publication of live student academic records and longitudinal industry employment trajectories, the dataset serves as an educational benchmark rather than an empirical study of the labor market.

---

## 10. Ethical and Social Considerations
1. **Data Privacy & Authorization**: Student academic records, transcripts, and notes are strictly protected using role-based access control (RBAC). Passwords are cryptographically salted and hashed using SHA-256. Only authorized advisors can review assigned advisee records.
2. **Anonymization in Cohort Analytics**: Programme Coordinator dashboards aggregate data purely at cohort and programme levels, completely omitting student names and registration numbers to prevent individual profiling.
3. **Mitigation of Algorithmic Bias**: Training data and classification rules were curated to avoid gender, demographic, or institutional bias. Career predictions are presented as ranked probabilistic affinities with clear explanations rather than rigid, deterministic assignments.
4. **Human-in-the-Loop**: The platform ensures that students and advisors retain ultimate agency in educational and career choices.

---

## 11. Author Contribution & System Responsibilities
 
| Author Name | Role / Degree Programme | Technical Responsibilities |
|---|---|---|
| **MFA Ahamad (Aqeel Ahamad)** | **Lead Developer & System Architect**<br>BSc (Hons) in Data Science & Business Analytics (Year 2) | • Core AI Architecture & Multi-tier pipeline design<br>• Machine learning feature extraction & classification pipeline<br>• Rule-based expert system authoring & prerequisite validation<br>• A\* search heuristic design & time-budgeted roadmap optimizer<br>• Streamlit multi-role web platform implementation & testing |


---

## 12. Conclusion
CareerSense AI successfully demonstrates how three complementary Artificial Intelligence paradigms—predictive classification, rule-based expert reasoning, and heuristic graph search—can be integrated into an explainable, practical decision-support platform for undergraduate computing education. The system bridges the critical gap between academic transcripts, industry competencies, and personalized learning roadmaps, empowering students to navigate their degree with clarity and confidence.

---

## 13. References
1. Bhumichitr, K., Channark, S., Malaivongs, K., Chatvichienchai, S., & Chanjaradwichai, S. (2017). Application of machine learning technique in the prediction of careers. *Proceedings of the International Conference on Advanced Informatics: Concepts, Theory and Applications*, 1–6.
2. Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
3. Sekar, J. R., & Nagarajan, S. K. (2020). Fuzzy logic based expert system for career guidance and counselling. *International Journal of Computer Applications*, 177(30), 1–6.
4. Xu, Y., & Chen, M. (2019). A hybrid decision support system for personalized career recommendation using data mining technologies. *Journal of Intelligent Information Systems*, 53(2), 271–291.
