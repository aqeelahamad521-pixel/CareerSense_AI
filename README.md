# 🎓 CareerSense AI
### An Explainable AI-Powered Career Development and Skill-Roadmap Platform for Undergraduate Computing Students

**General Sir John Kotelawala Defence University (KDU)**  
**Faculty of Computing — Intake 41/42**  
**Module:** Essentials of Artificial Intelligence (Group 19)

---

## 👤 Author & Academic Attribution

| Field | Details |
|---|---|
| **Student Author** | **MFA Ahamad (Aqeel Ahamad)** |
| **Degree Programme** | **BSc (Hons) in Data Science & Business Analytics** |
| **Academic Level** | **Undergraduate (Intake 41/42)** |
| **Faculty & Department** | **Faculty of Computing, General Sir John Kotelawala Defence University (KDU)** |
| **Module** | **Essentials of Artificial Intelligence (Group 19)** |

---

## 🌟 Key Features & AI Innovation

CareerSense AI solves the problem of generic, non-personalized career guidance by combining **three complementary AI paradigms**:

1. **AI Layer 1 — Machine Learning Classification**:
   - **K-Nearest Neighbors (K-NN)** as primary model with neighbor distance confidence.
   - **Decision Tree Classifier** as interpretable baseline providing decision paths and feature importances.
   - Classifies student profiles across 5 core computing tracks:
     * *Software Engineering*
     * *Data Science / AI*
     * *Cybersecurity*
     * *Cloud / DevOps*
     * *UI/UX Design*
2. **AI Layer 2 — Rule-Based Expert System**:
   - Explicit forward-chaining IF-THEN rules for academic course prerequisites.
   - Computes competency gap sizes (0 to 3 levels) and priority (High, Medium, Satisfied).
   - Validates inter-skill dependencies (e.g. Automated Testing requires OOP).
   - Generates human-readable, transparent diagnostic explanations.
3. **AI Layer 3 — A\* Search Algorithm for Roadmap Optimization**:
   - Formulates upskilling as a state-space graph search over a curated activity DAG (courses, projects, certifications).
   - Cost function $g(n) = \text{accumulated study hours}$.
   - Admissible heuristic $h(n) = \sum_{s \in \text{Gaps}} \min_{a} \text{Hours}(a)$ providing an admissible lower-bound on remaining effort to close all competency gaps.
   - Maps the optimal learning path into week-by-week milestones constrained by the student's declared weekly study budget (e.g., 8 hrs/week), backed by a topological sort fallback if graph activities cannot fully cover the target competencies.
4. **Interactive Multi-Role Web Platform**:
   - **Student Portal**: Profile, academic grades, skill management, document uploads, radar chart comparison, A* roadmap task tracking, and downloadable career reports.
   - **Academic Advisor Portal**: Advisee search, AI explanation review, prerequisite audit, and feedback note submission.
   - **Programme Coordinator Portal**: Anonymized cohort-level interest distribution, skill deficiency heatmaps, and curriculum insights.
   - **Administrator / Model Explorer**: Real-time confusion matrices, benchmark metrics, rule base inspector, and model re-training.

---

## 📂 Project Structure

```
CareerSense_AI/
├── app.py                          # Streamlit application entry point & role dispatcher
├── config.py                       # Global settings, constants, and feature definitions
├── requirements.txt                # Python package dependencies
├── README.md                       # Comprehensive documentation
│
├── data/
│   ├── students_dataset.csv        # 850 synthetic undergraduate computing student records
│   ├── career_definitions.json     # Benchmark competencies and requirements for the 5 tracks
│   ├── rules_knowledge_base.json   # Expert system IF-THEN rules
│   └── learning_graph.json         # Curated learning activity DAG for A* search
│
├── ai_engine/
│   ├── ml_classifier.py            # AI Layer 1: K-NN and Decision Tree models
│   ├── rule_engine.py              # AI Layer 2: Rule-Based Expert System
│   ├── a_star_roadmap.py           # AI Layer 3: A* Roadmap Search Optimizer
│   └── explainability.py           # Multi-model reasoning and radar visualization engine
│
├── database/
│   ├── db_manager.py               # SQLite relational database manager
│   └── seed_data.py                # Pre-populated demo personas and scenario student
│
├── modules/
│   ├── auth.py                     # Session auth and login/registration component
│   ├── student_view.py             # Student dashboard, radar charts, roadmap tracker
│   ├── advisor_view.py             # Academic advisor decision-support portal
│   ├── coordinator_view.py         # Programme coordinator cohort analytics
│   ├── admin_view.py               # Admin portal & AI model benchmarking
│   └── report_generator.py         # Personalized career report generator (HTML/Print)
│
├── scripts/
│   ├── generate_dataset.py         # Generates the 850 student profile dataset
│   └── train_models.py             # Trains models and computes evaluation benchmarks
│
├── tests/
│   └── test_all.py                 # Comprehensive unit and integration test suite
│
└── docs/
    ├── final_report.md             # Complete Academic Final Report (Stage 3 structure)
    └── presentation_script.md      # Presentation slide outline and speaker notes
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure **Python 3.10+** is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train AI Models (Optional - auto-initialized on first run)
```bash
python scripts/generate_dataset.py
python scripts/train_models.py
```

### 4. Run Unit Tests
```bash
python -m unittest tests/test_all.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🔑 Demo Personas & Credentials

> ⚠️ **DEMONSTRATION & SANDBOX CREDENTIALS**: All credentials listed below are provided strictly for local sandbox demonstration of this academic prototype. They connect exclusively to an isolated local SQLite database (`data/careersense.db`) populated with synthetic demonstration records and have no access to external or personal systems. Never commit or use real personal passwords.

You can click any of the **Quick Demo Access** buttons on the login screen or log in using:

| Persona / Role | Username | Password | Notes |
|---|---|---|---|
| **Student Persona** | `student_demo` | `student123` | Demo Student Persona (Undergraduate Computing Student) |
| **Academic Advisor** | `advisor` | `advisor123` | Dr. Nihal Fernando (Review advisees & log feedback) |
| **Programme Coordinator** | `coordinator` | `coordinator123` | Prof. K. Jayasinghe (Anonymized cohort analytics) |
| **System Administrator** | `admin` | `admin123` | Model explorer, confusion matrices, rule inspector |

---

## 📊 AI Model Evaluation Benchmarks

Evaluated on 170 holdout test samples (80/20 train/test split of 850 undergraduate profiles with 5-Fold Stratified Cross-Validation on the training set):

| Metric | K-Nearest Neighbors (Primary Model) | Decision Tree (Baseline Model) |
|---|---|---|
| **5-Fold CV Accuracy (Train Set)** | **89.56% (±2.48%)** *(Pipeline Isolation)* | **63.53% (±4.50%)** |
| **Held-Out Test Accuracy** | **88.82%** | **60.00%** |
| **Precision (Weighted)** | **89.74%** | **64.16%** |
| **Recall (Weighted)** | **88.82%** | **60.00%** |
| **F1-Score (Weighted)** | **88.87%** | **60.39%** |
| **Macro F1-Score (Unweighted)** | **89.49%** | **59.72%** |

#### Multi-Model Benchmark Suite (Held-Out Test Set):
- **Zero-Rule (Dummy Baseline)**: 29.41% Accuracy | 9.09% Macro F1
- **Decision Tree (max depth 6)**: 60.00% Accuracy | 59.72% Macro F1
- **Random Forest (100 trees)**: 82.94% Accuracy | 83.13% Macro F1
- **Multinomial Logistic Regression**: 88.82% Accuracy | 89.45% Macro F1
- **K-Nearest Neighbors ($k=7$, distance)**: **88.82% Accuracy** | **89.49% Macro F1** (Selected Primary Model)

### 🛡️ Evaluation Methodology & Synthetic Dataset Transparency
- **Preprocessing Leakage Remediation**: K-NN cross-validation is evaluated strictly through a scikit-learn `Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(...))])` executed on the training split, ensuring that scaling parameters are computed independently within each fold. The held-out test split (20%, $N=170$) is strictly isolated for final testing.
- **Prediction Score Semantics**: User-facing "Match Scores" represent an uncalibrated composite advisory affinity index (combining academic coursework, technical skills, domain interests, and career aspirations). They are normalized for ranking and should not be interpreted as calibrated Bayesian probabilities of post-graduate employment. The pure ML track prediction (`ml_predicted_track`) evaluates coursework and skills independently from student aspirations.
- **A\* Search Optimality & Bounds**: The heuristic $h(n) = \sum_{s \in \text{Gaps}} \min_{a} \text{Hours}(a)$ provides an admissible lower-bound under single-skill curriculum activities. The search explores state space ordered by $f(n) = g(n) + h(n)$ to minimize total study effort. For sparse or incomplete activity DAGs, a deterministic topological fallback guarantees a valid, prerequisite-consistent sequence.
- **Dataset Context & Ethical Limitations**: The student cohort (`data/students_dataset.csv`, 850 records) is an educational synthetic demonstration reflecting real computing curricula. While engineered with realistic latent skills, prerequisite rules, and overlapping interests, it is explicitly labelled as synthetic and should not be treated as empirical graduate employment data.

---

## 📑 Deliverables Included
- ✅ Complete working Streamlit prototype with multi-role dashboards
- ✅ AI Layer 1 (K-NN & Decision Tree ML Classifier with persistence)
- ✅ AI Layer 2 (Rule-Based Expert System with prerequisite validation)
- ✅ AI Layer 3 (A\* Search Roadmap Optimizer with weekly time constraint)
- ✅ Complete dataset generation script (`scripts/generate_dataset.py`)
- ✅ Automated unit and integration test suite (`tests/test_all.py`)
- ✅ Automated downloadable personalized Career Advisory Reports
- ✅ Complete Academic Final Report (`docs/final_report.md`)
- ✅ Group Presentation & Video Outline Script (`docs/presentation_script.md`)
