# GENERAL SIR JOHN KOTELAWALA DEFENCE UNIVERSITY
## FACULTY OF COMPUTING
### INTAKE 41 & INTAKE 42

**Essentials of Artificial Intelligence**  
**Intake 41 – IT3182 (IT/IS) | Intake 42 – CS 22032 (DBA) / CS22023 (CS/SE/COE)**

---

# PROGRESS REVIEW REPORT (STAGE 2)

### Project Details
| Field | Information |
|---|---|
| **Project Title** | **CareerSense AI: An Explainable AI-Powered Career Development and Skill-Roadmap Platform for Undergraduate Students** |
| **Domain** | Artificial Intelligence / Decision Support / Educational Technology |
| **Primary Deliverable** | Working AI-Enabled Web Prototype, Trained Models, Knowledge Base, and Roadmap Graph |

### Group & Author Details
| Name | Academic Level | Degree Programme |
|---|---|---|
| **MFA Ahamad (Aqeel Ahamad)** | **Undergraduate (Intake 41/42)** | **Department of Data Science and Business Analytics** |

**Group No:** 19  
**Submission Date:** 21.09.2026  

---

## Table of Contents
1. [Introduction](#1-introduction)  
   1.1 [Project Overview](#11-project-overview)  
   1.2 [Problem Statement](#12-problem-statement)  
2. [Changes Made After Proposal](#2-changes-made-after-proposal)  
   2.1 [Changes to Project Scope](#21-changes-to-project-scope)  
   2.2 [Finalization of AI Techniques](#22-finalization-of-ai-techniques)  
       2.2.1 [Machine Learning Classification](#221-machine-learning-classification)  
       2.2.2 [Rule-Based Expert System](#222-rule-based-expert-system)  
       2.2.3 [A* Search Algorithm for Roadmap Optimization](#223-a-search-algorithm-for-roadmap-optimization)  
3. [System Workflow](#3-system-workflow)  
   3.1 [Workflow Diagram](#31-workflow-diagram)  
   3.2 [Workflow Explanation](#32-workflow-explanation)  
4. [Dataset Details](#4-dataset-details)  
   4.1 [Dataset Overview](#41-dataset-overview)  
   4.2 [Dataset Attributes](#42-dataset-attributes)  
5. [AI Model Development](#5-ai-model-development)  
   5.1 [Machine Learning Classification Model](#51-machine-learning-classification-model)  
       5.1.1 [Purpose of Machine Learning Classification](#511-purpose-of-machine-learning-classification)  
       5.1.2 [Input Features](#512-input-features)  
       5.1.3 [Target Variable](#513-target-variable)  
       5.1.4 [Data Preprocessing](#514-data-preprocessing)  
       5.1.5 [Classification Algorithm](#515-classification-algorithm)  
       5.1.6 [Model Training](#516-model-training)  
       5.1.7 [Model Testing and Evaluation](#517-model-testing-and-evaluation)  
       5.1.8 [Model Output](#518-model-output)  
   5.2 [Rule-Based Expert System](#52-rule-based-expert-system)  
       5.2.1 [Purpose of Rule-Based Reasoning](#521-purpose-of-rule-based-reasoning)  
       5.2.2 [Inputs to the Rule-Based System](#522-inputs-to-the-rule-based-system)  
       5.2.3 [Rule Definitions](#523-rule-definitions)  
       5.2.4 [Rule Evaluation Process](#524-rule-evaluation-process)  
       5.2.5 [Explanation and Diagnostic Reasons](#525-explanation-and-diagnostic-reasons)  
       5.2.6 [Rule-Based Reasoning Output](#526-rule-based-reasoning-output)  
   5.3 [A* Search Algorithm for Learning Roadmap Optimization](#53-a-search-algorithm-for-learning-roadmap-optimization)  
       5.3.1 [Purpose of A* Search in Skill Roadmapping](#531-purpose-of-a-search-in-skill-roadmapping)  
       5.3.2 [Inputs to the A* Search Engine](#532-inputs-to-the-a-search-engine)  
       5.3.3 [State Representation, Cost Function & Admissible Heuristic](#533-state-representation-cost-function--admissible-heuristic)  
       5.3.4 [Graph Search Execution & Prerequisite Adherence](#534-graph-search-execution--prerequisite-adherence)  
       5.3.5 [A* Search Output & Weekly Schedule Generation](#535-a-search-output--weekly-schedule-generation)  
6. [Screenshots and Sample Outputs of Implementation](#6-screenshots-and-sample-outputs-of-implementation)  
   6.1 [Login & Persona Access Interface](#61-login--persona-access-interface)  
   6.2 [Student Profile, Academic Records & Skill Management](#62-student-profile-academic-records--skill-management)  
   6.3 [AI Career Assessment, Radar Comparison & Explainability Output](#63-ai-career-assessment-radar-comparison--explainability-output)  
   6.4 [Rule-Based Prerequisite Validation & Skill Gap Diagnosis](#64-rule-based-prerequisite-validation--skill-gap-diagnosis)  
   6.5 [A* Learning Roadmap & Interactive Progress Tracker](#65-a-learning-roadmap--interactive-progress-tracker)  
   6.6 [Academic Advisor Decision Support Dashboard](#66-academic-advisor-decision-support-dashboard)  
   6.7 [Programme Coordinator Anonymized Cohort Analytics](#67-programme-coordinator-anonymized-cohort-analytics)  
   6.8 [Machine Learning Model Evaluation & Confusion Matrix Output](#68-machine-learning-model-evaluation--confusion-matrix-output)  
7. [Problems Faced During Development](#7-problems-faced-during-development)  
8. [Remaining Work Before Final Submission](#8-remaining-work-before-final-submission)  
9. [Contribution of Group Members](#9-contribution-of-group-members)  

---

## 1. Introduction

### 1.1 Project Overview
Undergraduate computing education encompasses a diverse spectrum of specialized industry disciplines, including Software Engineering, Data Science / AI, Cybersecurity, Cloud / DevOps, and UI/UX Design. Throughout their degree programmes, undergraduate students must make crucial decisions regarding elective courses, extracurricular certifications, practical portfolio projects, and internship placements. 

However, many students struggle with choosing the right pathway due to information overload, inconsistent skill descriptions, and the absence of continuous, personalized academic advising. Traditional advising mechanisms in universities are typically episodic, generalized, and manually conducted, making it challenging for faculty mentors to evaluate every student's granular competencies, academic performance, and weekly time availability. Consequently, students often discover critical competency deficiencies too late in their academic lifecycle.

**CareerSense AI** is an intelligent, explainable decision-support and skill-roadmap platform designed specifically for undergraduate computing students. The system integrates three complementary Artificial Intelligence techniques:
1. **Machine Learning Classification (K-NN & Decision Tree)** to evaluate student academic grades, domain interests, and technical skills to predict suitability across five primary career tracks.
2. **Rule-Based Expert System** to validate university module prerequisites, verify inter-skill dependencies, detect competency gaps, and generate human-understandable explanations.
3. **A\* Search Algorithm** to explore a directed graph of learning activities (courses, practical projects, certifications) and construct an optimal, time-budgeted weekly learning roadmap tailored to the student's available study hours.

The platform provides role-tailored dashboards for **Students**, **Academic Advisors**, **Programme Coordinators**, and **System Administrators**, backed by SQLite persistence and automated downloadable career reports.

### 1.2 Problem Statement
Undergraduate students pursuing computing degrees lack access to an adaptive, personalized, and continuously updated advisory system that correlates academic results, technical proficiencies, portfolio evidence, and weekly time constraints with industry career expectations.

Existing approaches rely on manual, periodic consultations and generalized online roadmaps that fail to relate recommendations to an undergraduate's verified prerequisite coursework. This leads to two major structural issues:
1. **Inefficient Learning & Prerequisite Violations**: Students attempt advanced industry tools (e.g., CI/CD or MLOps) without mastering foundational coursework (e.g., Linux architecture, Docker, or OOP), leading to high failure rates.
2. **Lack of Explainability in Automated Systems**: Black-box matching algorithms offer predictions without explaining *why* a particular career was suggested, what prerequisites are missing, or how many hours are realistically required to bridge identified gaps.

To address these challenges, **CareerSense AI** combines predictive machine learning, deterministic rule-based verification, and heuristic state-space search into a unified platform that delivers actionable, transparent, and time-budgeted guidance.

---

## 2. Changes Made After Proposal

### 2.1 Changes to Project Scope
The fundamental objective of designing an explainable AI career advisory system has remained consistent with the Stage 1 proposal. However, during prototype development and intermediate feedback, several strategic refinements were introduced:

1. **Dual Machine Learning Architecture**: While the proposal outlined K-NN as the primary model and Decision Tree as a baseline, the system was refined to compute an ensemble probability distribution (70% K-NN proximity weighting + 30% Decision Tree rule weighting). This provides both neighbor-based exemplar comparisons and transparent decision paths.
2. **Dynamic Skill Reassessment on Activity Completion**: Originally, progress tracking was planned as a passive logging mechanism. In the developed prototype, when a student toggles an activity status to "Completed", the system automatically upgrades the relevant skill proficiency in the SQLite database and triggers an instant recalculation of the holistic Career Readiness Score.
3. **Downloadable Formatted HTML/Print Career Reports**: A comprehensive report generation module was added, enabling students and advisors to export a complete, printable diagnostic dossier containing student metrics, Plotly radar comparisons, prerequisite audits, and the week-by-week A* roadmap.
4. **Primary Persona Focus**: The primary scenario profile was refined to **Aqeel Ahamad (MFA Ahamad)**, Year 2, enrolled in the **BSc (Hons) in Data Science & Business Analytics** degree, aligning the demonstration with realistic university cohorts.

### 2.2 Finalization of AI Techniques
The three core AI techniques have been finalized, implemented, and benchmarked:

```
+-------------------------------------------------------------------------+
|                       CAREERSENSE AI - 3 AI LAYERS                      |
+-------------------------------------------------------------------------+
|  AI Layer 1: Machine Learning Classification                            |
|  - K-Nearest Neighbors (K-NN) Primary Classifier (k=7, Distance Weights)|
|  - Decision Tree Classifier Interpretable Baseline (Max Depth=6)        |
|  -> Output: Multi-class career probabilities & nearest neighbor metrics |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|  AI Layer 2: Rule-Based Expert System                                   |
|  - Forward-chaining inference engine over JSON knowledge base           |
|  - Academic prerequisite rules (Grade minimums e.g. Programming >= B-)  |
|  - Inter-skill dependency rules (e.g. Automated Testing requires OOP)  |
|  -> Output: Prioritized skill gaps, Pass/Fail audits, Readiness Score   |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|  AI Layer 3: A* Search Algorithm for Roadmap Optimization               |
|  - State-space graph search over learning activity DAG                  |
|  - Cost function g(n): Accumulated study effort hours                   |
|  - Admissible heuristic h(n): Sum of minimum hours for remaining gaps   |
|  -> Output: Time-constrained, week-by-week prioritized learning roadmap |
+-------------------------------------------------------------------------+
```

#### 2.2.1 Machine Learning Classification
* **Algorithms**: K-Nearest Neighbors (Primary) and Decision Tree (Baseline).
* **Input**: 36-dimensional standardized feature vector (6 academic subject grades, 5 domain interests, 25 technical skill levels).
* **Output**: Ranked career track predictions with alignment percentages across 5 tracks: *Software Engineering*, *Data Science / AI*, *Cybersecurity*, *Cloud / DevOps*, and *UI/UX Design*.

#### 2.2.2 Rule-Based Expert System
* **Algorithm**: Forward-chaining production rule engine executing deterministic IF–THEN evaluations.
* **Input**: Academic records, target career benchmark competencies, self-assessed proficiencies, project portfolio, and document metadata.
* **Output**: Prerequisite audit status (Pass/Violation), prioritized skill gaps (High/Medium/Satisfied), diagnostic explanation strings, and composite Career Readiness Score (0%–100%).

#### 2.2.3 A* Search Algorithm for Roadmap Optimization
* **Algorithm**: Heuristic state-space graph search with priority queue expansion.
* **Input**: Current student skill set, target career benchmark levels, student weekly available study hours (e.g., 8 hrs/week), and curated activity DAG.
* **Output**: Optimal, dependency-consistent sequence of learning activities organized into weekly milestones.

---

## 3. System Workflow

### 3.1 Workflow Diagram

```
+---------------------------------------------------------------------------------------------------+
|                                      1. PRESENTATION LAYER (Streamlit)                            |
|  +-----------------------+  +--------------------------+  +------------------------------------+  |
|  | Student Dashboard     |  | Academic Advisor Portal  |  | Programme Coordinator Analytics    |  |
|  | - Profile & Academics |  | - Student Advisee Search |  | - Anonymized Cohort Trends         |  |
|  | - Skills & Portfolio  |  | - AI Explanation Review  |  | - Pervasive Skill Deficiency Heatmap|  |
|  | - A* Roadmap Tracker  |  | - Mentorship Notes Log   |  | - Curriculum Recommendations      |  |
|  +-----------------------+  +--------------------------+  +------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                  2. APPLICATION & SERVICE LAYER                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | Authentication (SHA-256) | Feature Vectorizer | Report Generator | State Persistence Engine |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+---------------------------------------------------+     +-----------------------------------------+
|               3. AI ENGINE PIPELINE               |     |       4. DATABASE & KNOWLEDGE BASE      |
|  +---------------------------------------------+  |     |  +-----------------------------------+  |
|  | 3.1 AI Layer 1: ML Classifier               |  |     |  | SQLite Relational DB              |  |
|  | - K-NN Classifier (Primary)                 |  |     |  | - users, student_profiles         |  |
|  | - Decision Tree (Interpretable Baseline)    |  |     |  | - academic_records, student_skills|  |
|  +---------------------------------------------+  |     |  | - student_projects, roadmaps      |  |
|                        |                          |     |  | - advisor_notes                   |  |
|                        v                          |     |  +-----------------------------------+  |
|  +---------------------------------------------+  |     |  +-----------------------------------+  |
|  | 3.2 AI Layer 2: Rule-Based Expert System    |  |     |  | Knowledge Base Schemas (JSON)     |  |
|  | - Academic Prerequisite Validation Engine   |  |     |  | - career_definitions.json         |  |
|  | - Competency Gap Analysis & Prioritization  |  |<--->|  | - rules_knowledge_base.json       |  |
|  | - Holistic Readiness Scoring Formula        |  |     |  | - learning_graph.json             |  |
|  +---------------------------------------------+  |     |  +-----------------------------------+  |
|                        |                          |     |  +-----------------------------------+  |
|                        v                          |     |  | Serialized Models & Benchmarks    |  |
|  +---------------------------------------------+  |     |  | - knn_model.pkl, dt_model.pkl     |  |
|  | 3.3 AI Layer 3: A* Roadmap Optimizer        |  |     |  | - scaler.pkl, metrics.json        |  |
|  | - Activity DAG State-Space Search           |  |     |  +-----------------------------------+  |
|  | - Admissible Skill-Distance Heuristic       |  |     +-----------------------------------------+
|  | - Weekly Time-Constrained Scheduler         |  |
|  +---------------------------------------------+  |
+---------------------------------------------------+
```

### 3.2 Workflow Explanation

* **Step 1 – Secure Authentication & Role Dispatch**:  
  The user logs into the platform. Based on the authenticated role (`student`, `advisor`, `coordinator`, `admin`), the system loads session state and renders the corresponding view.
* **Step 2 – Student Profile & Evidence Entry**:  
  The student inputs or updates degree programme details, academic semester module grades, self-assessed skill levels, domain interest sliders, portfolio project links, and document metadata.
* **Step 3 – Feature Extraction & Normalization**:  
  The application service extracts the student's raw data and constructs a 36-dimensional numeric feature vector $\vec{x}$, standardizing values using `StandardScaler`.
* **Step 4 – Machine Learning Classification (AI Layer 1)**:  
  The feature vector is fed to both the trained K-NN classifier and the Decision Tree model. The system computes class probabilities, neighbor distances, and ranked pathway affinities.
* **Step 5 – Prerequisite Verification (AI Layer 2)**:  
  The Rule-Based Expert System evaluates student academic records against prerequisite rules for the chosen career track (e.g., verifying that Programming grade $\ge$ B- for Software Engineering).
* **Step 6 – Competency Gap Diagnosis & Prioritization**:  
  The rule engine compares the student's current proficiency with benchmark requirements, computing gap sizes and assigning priority levels (High, Medium, Satisfied).
* **Step 7 – Career Readiness Scoring**:  
  The system computes a composite readiness score (0%–100%) by weighting Competency Alignment (50%), Academic Prerequisites (25%), Project Evidence (15%), and Verified Documents (10%).
* **Step 8 – Heuristic Roadmap Optimization (AI Layer 3)**:  
  Given the student's declared weekly study budget (e.g., 8 hrs/week), the A\* search algorithm explores the learning activity graph, finds the optimal path closing all skill gaps, and organizes activities into weekly milestones.
* **Step 9 – Interactive Execution & Dynamic Reassessment**:  
  The student reviews their roadmap and toggles activity statuses (`Planned` $\rightarrow$ `In-Progress` $\rightarrow$ `Completed`). Marking an activity as "Completed" immediately upgrades the corresponding skill proficiency in SQLite and triggers live score reassessment.
* **Step 10 – Decision Support & Report Generation**:  
  Academic advisors inspect the student's profile, review AI explanations, and record mentorship notes. The student can export an official, printable Career Advisory Report.

---

## 4. Dataset Details

### 4.1 Dataset Overview

| Attribute | Specification |
|---|---|
| **Dataset Source** | Synthesized University Undergraduate Benchmark Dataset |
| **Generation Script** | `scripts/generate_dataset.py` (NumPy & Pandas) |
| **Domain Scope** | Undergraduate Computing Programmes (CS, SE, IT, IS, CE, DBA) |
| **Number of Records** | 850 labeled student profiles |
| **Number of Attributes** | 38 columns (Student ID, Name, Reg No, Degree, Year, GPA, 6 Grades, 5 Interests, 24 Skills, Career Track) |
| **Dataset Format** | CSV (Comma-Separated Values) |
| **Class Balance** | Software Engineering: 238, Data Science / AI: 187, Cybersecurity: 153, Cloud / DevOps: 144, UI/UX Design: 128 |

### 4.2 Dataset Attributes

| Attribute Name | Description | Original Data Type | Feature Type |
|---|---|---|---|
| `student_id` | Unique identifier for student record | String | Identifier |
| `name` | Student full name | String | Nominal |
| `reg_no` | University registration index number | String | Identifier |
| `degree` | Enrolled undergraduate degree programme | String | Categorical |
| `year` | Current academic year (Year 1 to Year 4) | Integer | Ordinal |
| `gpa` | Cumulative Grade Point Average (0.00–4.00) | Float | Numerical |
| `grade_programming` | Grade points in core programming modules | Float | Numerical (0.0–4.0) |
| `grade_math` | Grade points in discrete math & statistics | Float | Numerical (0.0–4.0) |
| `grade_database` | Grade points in database management systems | Float | Numerical (0.0–4.0) |
| `grade_networking` | Grade points in computer networks & protocols | Float | Numerical (0.0–4.0) |
| `grade_systems` | Grade points in operating systems & architecture | Float | Numerical (0.0–4.0) |
| `grade_design` | Grade points in HCI and design concepts | Float | Numerical (0.0–4.0) |
| `interest_software` | Self-reported interest in software systems | Integer | Ordinal (1–5) |
| `interest_data` | Self-reported interest in data analytics & AI | Integer | Ordinal (1–5) |
| `interest_security` | Self-reported interest in cybersecurity defense | Integer | Ordinal (1–5) |
| `interest_cloud` | Self-reported interest in cloud & DevOps | Integer | Ordinal (1–5) |
| `interest_design` | Self-reported interest in UI/UX & design | Integer | Ordinal (1–5) |
| `skill_oop` | Proficiency in Object-Oriented Programming | Integer | Ordinal (0=None, 1=Beg, 2=Int, 3=Adv) |
| `skill_dsa` | Proficiency in Data Structures & Algorithms | Integer | Ordinal (0–3) |
| `skill_web_api` | Proficiency in REST APIs & Web Services | Integer | Ordinal (0–3) |
| `skill_testing` | Proficiency in Automated Testing & QA | Integer | Ordinal (0–3) |
| `skill_git` | Proficiency in Version Control (Git) | Integer | Ordinal (0–3) |
| `skill_python_data` | Proficiency in Pandas & NumPy Stack | Integer | Ordinal (0–3) |
| `skill_ml` | Proficiency in Machine Learning & Modeling | Integer | Ordinal (0–3) |
| `skill_visualization`| Proficiency in Data Visualization & Dashboards | Integer | Ordinal (0–3) |
| `skill_sql` | Proficiency in SQL & Relational Querying | Integer | Ordinal (0–3) |
| `skill_statistics` | Proficiency in Applied Statistical Inference | Integer | Ordinal (0–3) |
| `skill_network_sec` | Proficiency in Network Security Protocols | Integer | Ordinal (0–3) |
| `skill_cryptography` | Proficiency in Cryptographic Fundamentals | Integer | Ordinal (0–3) |
| `skill_linux` | Proficiency in Linux Systems Administration | Integer | Ordinal (0–3) |
| `skill_vuln_assess` | Proficiency in Vulnerability Assessment | Integer | Ordinal (0–3) |
| `skill_docker` | Proficiency in Containerization (Docker) | Integer | Ordinal (0–3) |
| `skill_cicd` | Proficiency in CI/CD Automation Pipelines | Integer | Ordinal (0–3) |
| `skill_cloud_infra` | Proficiency in AWS / GCP Cloud Computing | Integer | Ordinal (0–3) |
| `skill_user_research`| Proficiency in User Research & Usability | Integer | Ordinal (0–3) |
| `skill_figma` | Proficiency in Wireframing & Figma | Integer | Ordinal (0–3) |
| `skill_design_systems`| Proficiency in Component Design Systems | Integer | Ordinal (0–3) |
| `career_track` | Ground-truth career pathway (Target Label) | String | Categorical (5 Classes) |

---

## 5. AI Model Development

### 5.1 Machine Learning Classification Model

#### 5.1.1 Purpose of Machine Learning Classification
The objective of the ML classification model is to predict which computing career tracks an undergraduate student is most naturally aligned with, based on their cumulative academic performance, technical skills, and domain interests.

#### 5.1.2 Input Features
The model consumes 36 numerical and ordinal features:
* **Academic Performance (6 features)**: `grade_programming`, `grade_math`, `grade_database`, `grade_networking`, `grade_systems`, `grade_design`.
* **Domain Interests (5 features)**: `interest_software`, `interest_data`, `interest_security`, `interest_cloud`, `interest_design`.
* **Demonstrated Skills (25 features)**: Discrete proficiency scores (0 to 3) representing core computing proficiencies.

#### 5.1.3 Target Variable
The target variable is `career_track`, representing the 5 primary career classifications:
1. `Software Engineering`
2. `Data Science / AI`
3. `Cybersecurity`
4. `Cloud / DevOps`
5. `UI/UX Design`

#### 5.1.4 Data Preprocessing
Data preprocessing is encapsulated within `ai_engine/ml_classifier.py`:
1. **Feature Extraction**: Maps student academic records and skill strings into column indices.
2. **Missing Value Imputation**: Default baseline grade points (2.5) and median skill weights (0) are applied when attributes are unrecorded.
3. **Feature Standardization**: `StandardScaler` standardizes features to zero mean and unit variance ($z = (x - \mu)/\sigma$).
4. **Stratified Splitting**: 80% training set (680 samples), 20% holdout test set (170 samples), stratified by class label.

#### 5.1.5 Classification Algorithm
* **Primary Classifier**: K-Nearest Neighbors (K-NN) configured with $k=7$, Euclidean distance metric, and inverse distance weighting ($w = 1 / (d + \epsilon)$). K-NN was selected because career alignment naturally resembles nearest-neighbor clustering in competency space.
* **Baseline Classifier**: Decision Tree Classifier configured with `max_depth=6` and Gini impurity splitting criterion. This provides an interpretable benchmark and exposes global feature importances.

#### 5.1.6 Model Training
Model training is executed via `scripts/train_models.py`. Both models fit on the 680 training profiles. Trained artifacts are serialized using Python's `pickle` library:
* `ai_engine/saved_models/knn_model.pkl`
* `ai_engine/saved_models/dt_model.pkl`
* `ai_engine/saved_models/scaler.pkl`
* `ai_engine/saved_models/evaluation_metrics.json`

#### 5.1.7 Model Testing and Evaluation
Evaluated on the 170 holdout test profiles:

| Evaluation Metric | K-Nearest Neighbors (Primary) | Decision Tree (Baseline) |
|---|---|---|
| **Accuracy** | **100.00%** | **98.82%** |
| **Precision (Weighted)** | **100.00%** | **98.82%** |
| **Recall (Weighted)** | **100.00%** | **98.82%** |
| **F1-Score (Weighted)** | **100.00%** | **98.82%** |
| **Training Set Size** | 680 records | 680 records |
| **Testing Set Size** | 170 records | 170 records |

**Decision Tree Feature Importances**:
* `grade_programming`: 0.22
* `grade_math`: 0.18
* `grade_design`: 0.16
* `grade_networking`: 0.15
* `grade_systems`: 0.14
* `interest_software`: 0.05
* `interest_data`: 0.04
* `interest_security`: 0.03

#### 5.1.8 Model Output
The model outputs an ensemble probability distribution across all 5 career tracks, along with nearest-neighbor distance metrics:
* *Example Prediction*:
  * **Data Science / AI**: 84.5% match
  * **Software Engineering**: 71.0% match
  * **Cloud / DevOps**: 38.2% match
  * **Cybersecurity**: 24.0% match
  * **UI/UX Design**: 18.5% match

---

### 5.2 Rule-Based Expert System

#### 5.2.1 Purpose of Rule-Based Reasoning
While machine learning provides statistical classification, academic advising requires **deterministic compliance** with curriculum standards and transparent explanations. The Rule-Based Expert System enforces prerequisite rules, computes exact competency deficiencies, and validates inter-skill dependencies.

#### 5.2.2 Inputs to the Rule-Based System
1. Student Academic Records (Module grades and subject areas)
2. Target Career Track Benchmark Competencies (`data/career_definitions.json`)
3. Student Technical Skill Proficiencies (None, Beginner, Intermediate, Advanced)
4. Portfolio Projects Count & Verified Document Metadata
5. Student Weekly Study Availability (Hours/Week)

#### 5.2.3 Rule Definitions
Rules are codified in `data/rules_knowledge_base.json` across three distinct rule groups:

1. **Academic Prerequisite Rules**:
   * *RULE_ACAD_SE_PROG*: `IF Target = 'Software Engineering' AND Programming Grade < B- (2.7) THEN Flag Prerequisite Violation`
   * *RULE_ACAD_DS_MATH*: `IF Target = 'Data Science / AI' AND Math & Statistics Grade < B- (2.7) THEN Flag Prerequisite Violation`
   * *RULE_ACAD_SEC_NET*: `IF Target = 'Cybersecurity' AND Networking Grade < B- (2.7) THEN Flag Prerequisite Violation`
   * *RULE_ACAD_CLOUD_OS*: `IF Target = 'Cloud / DevOps' AND Operating Systems Grade < B- (2.7) THEN Flag Prerequisite Violation`
   * *RULE_ACAD_UI_HCI*: `IF Target = 'UI/UX Design' AND HCI Grade < B- (2.7) THEN Flag Prerequisite Violation`

2. **Competency Dependency Rules**:
   * *DEP_TESTING_PROG*: `IF Skill('Automated Testing') > None AND Skill('OOP') < Beginner THEN Alert: 'Automated testing requires OOP foundations'`
   * *DEP_CICD_DOCKER*: `IF Skill('CI/CD Pipelines') > None AND Skill('Docker') < Beginner THEN Alert: 'CI/CD deployment requires containerization basics'`
   * *DEP_ML_PYTHON*: `IF Skill('Machine Learning') > None AND Skill('Python Data Stack') < Intermediate THEN Alert: 'ML modeling requires Pandas/NumPy data preparation'`

3. **Portfolio & Workload Feasibility Rules**:
   * *PORTFOLIO_CHECK*: `IF Projects_Count == 0 THEN Severity: Medium, Recommendation: 'Add at least 2 practical projects'`
   * *WORKLOAD_FEASIBILITY*: `IF Weekly_Hours < 8 AND Remaining_Gaps > 3 THEN Notice: 'Weekly study allocation is low; consider extending horizon'`

#### 5.2.4 Rule Evaluation Process
1. Query student academic history and extract the highest grade points achieved per subject area.
2. Evaluate target career prerequisite rules using forward-chaining.
3. Compare student skill levels against benchmark requirements to calculate:
   $$\text{Gap Size} = \max(0, \text{Target Level} - \text{Current Level})$$
4. Categorize gap priority:
   $$\text{Priority} = \begin{cases} \text{High}, & \text{if Gap Size} \ge 2 \text{ or Priority is High} \\ \text{Medium}, & \text{if Gap Size} = 1 \\ \text{Satisfied}, & \text{if Gap Size} = 0 \end{cases}$$
5. Calculate composite Career Readiness Score:
   $$\text{Readiness} = \text{Competency}(50\%) + \text{Academic}(25\%) + \text{Projects}(15\%) + \text{Documents}(10\%)$$

#### 5.2.5 Explanation and Diagnostic Reasons
The rule engine produces structured explanations detailing:
* Specific modules that satisfied or violated prerequisite criteria.
* Exact levels missing for each skill (e.g., *Machine Learning & Modeling: Current 'Beginner' vs Benchmark 'Intermediate'*).
* Contextual remedies guiding the student on how to close each gap.

#### 5.2.6 Rule-Based Reasoning Output
```json
{
  "total_readiness_score": 72.5,
  "readiness_tier": "Near Ready (Minor Gaps)",
  "academic_eval": {
    "overall_academic_met": true,
    "satisfied": [
      {"subject": "Mathematics & Statistics", "required": "B-", "achieved": "A", "status": "Pass"}
    ]
  },
  "skill_gaps": [
    {"skill": "Machine Learning & Modeling", "current": "Beginner", "required": "Intermediate", "gap": 1, "priority": "High"},
    {"skill": "Data Visualization", "current": "Beginner", "required": "Intermediate", "gap": 1, "priority": "Medium"}
  ]
}
```

---

### 5.3 A* Search Algorithm for Learning Roadmap Optimization

#### 5.3.1 Purpose of A* Search in Skill Roadmapping
Rather than providing an unstructured list of recommended courses, CareerSense AI formulates career upskilling as a **state-space pathfinding problem**. The A\* algorithm computes the shortest, prerequisite-consistent sequence of learning activities that bridges all identified skill gaps.

#### 5.3.2 Inputs to the A* Search Engine
1. Student Current Skill Profile
2. Target Benchmark Skill Requirements
3. Weekly Study Hour Allocation (e.g., 8 hours/week)
4. Curated Learning Activity Graph (`data/learning_graph.json`) containing 21 activities with duration, skill gains, and prerequisite activity IDs.

#### 5.3.3 State Representation, Cost Function & Admissible Heuristic
* **State $S$**: Tuple of current skill levels $\vec{s}$ and set of completed activity IDs $A_{\text{completed}}$.
* **Cost Function $g(n)$**: Total accumulated study hours of activities completed along the path:
  $$g(n) = \sum_{a \in \text{path}} \text{EstimatedHours}(a)$$
* **Admissible Heuristic $h(n)$**: Lower-bound estimate of remaining hours needed to satisfy all remaining skill gaps:
  $$h(n) = \sum_{s \in \text{Unsatisfied Gaps}} \min_{a \in \text{Available}(s)} \text{EstimatedHours}(a)$$
  Because $h(n)$ takes the minimum possible duration for each remaining gap and never overestimates the true remaining effort ($h(n) \le h^*(n)$), the heuristic is strictly **admissible**, guaranteeing that the first goal state popped from the priority queue represents an optimal sequence.

#### 5.3.4 Graph Search Execution & Prerequisite Adherence
The algorithm maintains an open set priority queue ordered by evaluation function:
$$f(n) = g(n) + h(n)$$
Transitions from state $S$ to child states are valid only if **all prerequisite activity IDs** of the candidate activity exist in $A_{\text{completed}}$.

#### 5.3.5 A* Search Output & Weekly Schedule Generation
The optimal activity sequence is scheduled into week-by-week milestones based on the student's weekly study budget:
$$\text{StartWeek}(a) = \text{CurrentWeek}, \quad \text{EndWeek}(a) = \text{StartWeek} + \left\lceil \frac{\text{Hours}(a)}{\text{WeeklyHours}} \right\rceil - 1$$
*Example Output (Aqeel Ahamad - 8 hrs/week budget)*:
1. **Weeks 1–2**: Exploratory Data Analysis & Visualization (14 hrs) &bull; Target: *Data Visualization*
2. **Weeks 3–5**: Supervised & Unsupervised Machine Learning (18 hrs) &bull; Target: *Machine Learning*
3. **Weeks 6–8**: End-to-End Predictive Machine Learning Project (22 hrs) &bull; Target: *Machine Learning*  
*Total Effort: 54 Hours across 8 Weeks.*

---

## 6. Screenshots and Sample Outputs of Implementation

### 6.1 Login & Persona Access Interface
The authentication portal provides secure credential sign-in, user registration, and 1-click demo persona buttons:
* **Student Access**: `student_demo` / `student123` (Aqeel Ahamad)
* **Academic Advisor**: `advisor` / `advisor123` (Dr. Nihal Fernando)
* **Programme Coordinator**: `coordinator` / `coordinator123` (Prof. K. Jayasinghe)
* **Administrator**: `admin` / `admin123`

### 6.2 Student Profile, Academic Records & Skill Management
* Centralized header displaying **Aqeel Ahamad (MFA Ahamad)**, Year 2, BSc (Hons) in Data Science & Business Analytics.
* Module records table detailing course codes (`BA2101`, `MA1102`, `CS1120`, `IT1223`, `CS2110`, `IT2133`) with grades and grade points.
* Interactive skill manager with proficiency badges (`None`, `Beginner`, `Intermediate`, `Advanced`).

### 6.3 AI Career Assessment, Radar Comparison & Explainability Output
* Side-by-side display of ranked career track matches with probability progress bars:
  * **Data Science / AI**: 84.5% Match
  * **Software Engineering**: 71.0% Match
* **Plotly Competency Radar Chart**: Visualizes student proficiency polygon (blue) overlaid against the benchmark requirement polygon (dashed red) across 6 domain competencies.
* Diagnostic explanation cards highlighting contributing strengths and growth areas.

### 6.4 Rule-Based Prerequisite Validation & Skill Gap Diagnosis
* Prerequisite audit card displaying green checkmark confirmation:  
  * *“Prerequisite satisfied: Grade A in Mathematics & Statistics meets required minimum B-.”*
* Detailed Skill Gap Matrix displaying Current Level, Required Benchmark, Gap Size, and Priority badges (`High`, `Medium`).

### 6.5 A* Learning Roadmap & Interactive Progress Tracker
* Week-by-week timeline cards displaying activity title, type (`Course`, `Project`, `Practice`), estimated duration, and main competency.
* Interactive status selector (`Planned`, `In-Progress`, `Completed`).
* Marking an activity as "Completed" immediately updates the skill level in SQLite and recalibrates the readiness score from 72.5% to 78.0%.

### 6.6 Academic Advisor Decision Support Dashboard
* Advisee selector enabling Dr. Nihal Fernando to review Aqeel Ahamad's profile.
* Audits of AI classification confidence, prerequisite rule compliance, and radar comparisons.
* Mentorship note submission form with persistent action items logged to the database.

### 6.7 Programme Coordinator Anonymized Cohort Analytics
* Pie chart displaying distribution of target career tracks across the 850-student cohort.
* Horizontal bar chart displaying mean career readiness across computing degree programmes.
* Cohort-wide skill deficiency matrix highlighting common skill gaps.

### 6.8 Machine Learning Model Evaluation & Confusion Matrix Output
* Model explorer view displaying Accuracy (100% K-NN, 98.8% DT), Precision, Recall, and F1-scores.
* Interactive Plotly annotated confusion matrix heatmaps.
* Gini feature importance ranking table.

---

## 7. Problems Faced During Development

### 1. Designing Multi-Dimensional Student Feature Representations
Mapping diverse undergraduate profiles into a standardized numeric vector was challenging due to differing grading scales and subjective skill self-assessments. This was resolved by designing a 36-dimensional feature schema combining normalized grade points (0.0–4.0), domain interest scales (1–5), and integer-mapped skill levels (0–3), coupled with `StandardScaler` normalization.

### 2. Formulating an Admissible Heuristic Function for A* Roadmap Search
Initial heuristic formulations estimated remaining effort by counting missing skills multiplied by average activity hours. However, when an activity required fewer hours than the average, the heuristic occasionally overestimated remaining cost, violating admissibility ($h(n) > h^*(n)$). This was corrected by computing $h(n)$ as the strict minimum duration among uncompleted candidate activities that advance each unsatisfied skill gap.

### 3. Transparent Prerequisite Rule Chaining in the Expert System
Academic curricula involve complex prerequisite dependencies (e.g., prerequisite courses must be passed before advanced electives). The rule engine was designed using a forward-chaining evaluation model where subject areas are dynamically mapped to historical highest grade points, ensuring prerequisite violations are flagged with clear remediation advice.

### 4. Dynamic State Persistence and Reassessment in Streamlit
Streamlit re-executes scripts from top to bottom on user interaction, which initially caused roadmap state and skill updates to reset. This was resolved by decoupling business logic into a dedicated SQLite database manager (`database/db_manager.py`) and managing session state variables across user interactions.

### 5. Multi-Class Evaluation Interpretation and Class Imbalance Mitigation
In multi-class career classification across 5 tracks, evaluating models using simple accuracy can hide class-specific performance disparities. This was addressed by implementing weighted precision, recall, F1-scores, and multiclass confusion matrices to ensure balanced evaluation across all career pathways.

---

## 8. Remaining Work Before Final Submission

| Task No. | Remaining Activity | Description | Target Completion |
|---|---|---|---|
| **1** | **Curriculum Handbook Catalog Expansion** | Integrate the official KDU curriculum handbook module catalog into `data/kdu_curriculum_catalog.json` for automated semester pre-filling. | Week 9 |
| **2** | **Granular Job Title Specializations** | Expand the 5 career tracks into 15+ specialized sub-roles (e.g., BI Analyst, Machine Learning Engineer, Backend API Developer). | Week 9 |
| **3** | **Advanced Document Parsing** | Enhance CV and certificate metadata parsing to extract verified technical keywords automatically. | Week 9 |
| **4** | **Cohort Usability Testing** | Conduct structured usability testing with a representative sample of undergraduate students and academic advisors to gather qualitative feedback. | Week 10 |
| **5** | **Final Report & Demonstration Video** | Finalize the Stage 3 technical report, record the system walkthrough video, and rehearse the presentation using the prepared script. | Week 10 |

---

## 9. Contribution of Group Members

| Member / Index No. | Assigned Role | Work Completed to Date | Remaining Responsibility |
|---|---|---|---|
| **MFA Ahamad (Aqeel Ahamad)** | **Lead AI Engineer & System Architect** | • Designed the multi-tier system architecture and relational SQLite database schema.<br>• Synthesized the 850-record undergraduate benchmark dataset (`scripts/generate_dataset.py`).<br>• Implemented and evaluated K-NN and Decision Tree classification models (`ai_engine/ml_classifier.py`).<br>• Engineered the forward-chaining Rule-Based Expert System and prerequisite knowledge base (`ai_engine/rule_engine.py`).<br>• Implemented the A\* search roadmap optimizer with admissible skill-distance heuristic (`ai_engine/a_star_roadmap.py`).<br>• Developed the multi-role Streamlit web application with Student, Advisor, Coordinator, and Admin dashboards (`app.py`, `modules/`).<br>• Built automated test suite with 100% test pass rate (`tests/test_all.py`).<br>• Implemented the downloadable HTML Career Advisory Report generator (`modules/report_generator.py`). | • Integrate official KDU curriculum module catalog into the student profile workflow.<br>• Expand sub-role specializations under core career tracks.<br>• Conduct usability evaluation sessions with peer students.<br>• Prepare final Stage 3 submission package and presentation video. |

---

*Report prepared and compiled for Stage 2: Progress Review &bull; Faculty of Computing, General Sir John Kotelawala Defence University.*
