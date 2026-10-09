# CareerSense AI — Project Presentation & Demonstration Script
## Essentials of Artificial Intelligence — Group 19
### Faculty of Computing, General Sir John Kotelawala Defence University

---

## Presentation Slide Outline & Speaker Roles

### Slide 1: Title & Presenter Introduction (Speaker: Aqeel Ahamad - 1 min)
- **Slide Title**: CareerSense AI: An Explainable AI-Powered Career Development and Skill-Roadmap Platform for Undergraduate Students
- **Context**: General Sir John Kotelawala Defence University, Faculty of Computing, Intake 41/42.
- **Presenter**:
  - **MFA Ahamad (Aqeel Ahamad)** | Year 2, BSc (Hons) in Data Science & Business Analytics
- **Speaker Notes**:
  > "Good morning, respected lecturers and panel members. Today, I am proud to present CareerSense AI, an intelligent, explainable decision-support and career roadmap platform tailored for undergraduate computing students."


---

### Slide 2: Problem Background & Research Gap (Speaker: KKGAM Bandara - 1.5 mins)
- **Problem**:
  - Proliferation of computing tracks (Software Engineering, Data Science, Cybersecurity, Cloud/DevOps, UI/UX).
  - High advisor-to-student ratios $\rightarrow$ generalized, episodic advice.
  - Students identify critical skill deficiencies too late in their degrees.
- **Research Gap**: Existing systems are either static job lists or opaque black-box matchers. They fail to combine predictive classification with prerequisite reasoning and time-constrained roadmap optimization.
- **Project Aim**: Create a 3-layer AI pipeline that profiles students, predicts career fit, diagnoses prerequisite gaps, and builds an optimal weekly learning roadmap.

---

### Slide 3: Three-Layer AI Architecture (Speaker: SGT Tharumila - 1.5 mins)
- **The Core AI Innovations**:
  1. **AI Layer 1 — Machine Learning Classification**: K-Nearest Neighbors (primary) + Decision Tree (baseline).
  2. **AI Layer 2 — Rule-Based Expert System**: Transparent IF-THEN rules for academic prerequisites and skill gap prioritization.
  3. **AI Layer 3 — A\* Search Optimization**: State-space search over an activity graph to compute the optimal, time-budgeted upskilling pathway.
- **Speaker Notes**:
  > "Rather than using a single algorithm for demonstration, CareerSense AI establishes a pipeline where each AI technique solves a distinct, necessary subproblem: ML predicts alignment, Expert Rules enforce educational standards, and A* optimizes the journey."

---

### Slide 4: AI Layer 1 — ML Modeling & Benchmarks (Speaker: SGT Tharumila - 2 mins)
- **Feature Vector**: 36 normalized inputs (6 academic subject grades, 5 domain interest ratings, 25 technical skill proficiencies).
- **Dataset**: 850 synthetic undergraduate computing profiles across KDU degree programmes.
- **Evaluation Methodology**: Stratified 80/20 train/test split; 5-fold cross-validation with isolated StandardScaler pipelines (zero preprocessing leakage).
- **Evaluation Results**:
  - **K-NN Classifier (Primary Model)**: **88.82% Held-Out Accuracy**, **88.87% Weighted F1**, **89.49% Macro F1** (5-Fold CV Accuracy: **89.56% ± 2.48%**).
  - **Decision Tree (Baseline)**: **60.00% Held-Out Accuracy**, **60.39% Weighted F1** (5-Fold CV: **63.53% ± 4.50%**).
  - **Candidate Benchmarks**: Zero-Rule Dummy (29.41%), Random Forest (82.94%), Multinomial Logistic Regression (88.82%).
- **Explainability**: Decision tree feature importances identify programming grades, mathematical foundation, and interest ratings as prime discriminators. K-NN instance distance metrics reveal Euclidean proximity to historical student profiles.

---

### Slide 5: AI Layer 2 — Rule-Based Expert System (Speaker: MFA Ahamad - 2 mins)
- **Knowledge Representation**: Forward-chaining IF-THEN rules covering:
  1. *Academic Prerequisites*: e.g., Software Engineering requires Programming $\ge$ B- (2.7 GPA points).
  2. *Skill Gap Prioritization*: Benchmark vs current competency level difference mapped to High, Medium, or Satisfied.
  3. *Inter-Skill Dependencies*: e.g., Automated Testing requires OOP foundations; CI/CD requires Docker.
- **Readiness Scoring**: Holistic composite formula: Competency Alignment (50%), Academic Prerequisites (25%), Projects (15%), and Verified Documents (10%).
- **Explainability**: Outputs human-understandable diagnostic messages explaining *why* a gap exists and *what* remedial actions should be taken.

---

### Slide 6: AI Layer 3 — A* Search Roadmap Optimization (Speaker: RMBP Rathnayake - 2 mins)
- **Problem Formulation**:
  - State $S$: (Current skills, Completed activities).
  - Cost $g(n)$: Accumulated study hours.
  - Heuristic $h(n)$: Admissible remaining skill distance (minimum duration of uncompleted activities closing unsatisfied gaps).
  - Graph: Directed Acyclic Graph (DAG) of courses, projects, practices, and certifications with strict prerequisite edges.
- **Weekly Time Constraint**: Allocates activities into week-by-week milestones tailored to the student's declared availability (e.g. 8 hours/week).
- **Speaker Notes**:
  > "Because our heuristic never overestimates the actual hours needed to close remaining gaps, A* guarantees an optimal, prerequisite-consistent learning order within the student's weekly study budget."

---

### Slide 7: Live System Demonstration & End-to-End Scenario (All Members - 3.5 mins)
- **Walkthrough Steps**:
  1. **Student Login**: Log in as Kasun Bandara (Year 2 IT student, 8 hrs/week availability).
  2. **AI Assessment**: Run assessment $\rightarrow$ System predicts Software Engineering (88.5%) and Data Science (74.2%).
  3. **Diagnostic Explanation**: View Plotly radar chart showing competency gaps in Automated Testing, REST APIs, and Docker.
  4. **A\* Roadmap Execution**: View 9-week roadmap. Mark *REST API Fundamentals* as "Completed".
  5. **Dynamic Reassessment**: System updates skill level to Intermediate and automatically recalibrates Career Readiness from 68.5% to 74.0%.
  6. **Report Generation**: Download polished HTML Career Advisory Report.
  7. **Advisor & Coordinator Views**: Show advisor mentorship note submission and coordinator cohort skill deficiency heatmaps.

---

### Slide 8: Ethics, Social Considerations & Conclusion (Speaker: KKGAM Bandara - 1.5 mins)
- **Ethical Safeguards**: Role-based access control, SHA-256 password hashing, anonymized cohort analytics, transparent explainability, and human-in-the-loop decision making.
- **Conclusion**: CareerSense AI provides an end-to-end, technically rigorous, and explainable platform bridging undergraduate academics with industry career readiness.
- **Q&A Session**: Team welcomes questions from the panel.
