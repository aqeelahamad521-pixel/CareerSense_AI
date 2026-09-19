# 🎓 CareerSense AI
### An Explainable AI-Powered Career Development and Skill-Roadmap Platform for Undergraduate Computing Students

**General Sir John Kotelawala Defence University (KDU)**  
**Faculty of Computing — Intake 41/42**  
**Module:** Essentials of Artificial Intelligence (Group 19)

---

## 👤 Student Author & Project Profile

| Name | Registration No. | Degree Programme | Academic Year | Cumulative GPA |
|---|---|---|---|---|
| **MFA Ahamad (Aqeel Ahamad)** | **D/DBA/25/0021** | **BSc (Hons) in Data Science & Business Analytics** | **Year 2** | **3.40** |


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
   - Formulates upskilling as a state-space graph search over an activity DAG (courses, projects, certifications).
   - $g(n) = \text{accumulated study hours}$.
   - Admissible heuristic $h(n) = \sum \min \text{hours for remaining skill gaps}$ (guarantees shortest path).
   - Constrained to student's declared weekly study budget (e.g. 8 hrs/week) to generate week-by-week milestones.
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

You can click any of the **Quick Demo Access** buttons on the login screen or log in using:

| Persona / Role | Username | Password | Notes |
|---|---|---|---|
| **Student Persona** | `student_demo` | `student123` | Aqeel Ahamad / MFA Ahamad (Year 2 Data Science & Business Analytics, GPA 3.40) |

| **Academic Advisor** | `advisor` | `advisor123` | Dr. Nihal Fernando (Review advisees & log feedback) |
| **Programme Coordinator** | `coordinator` | `coordinator123` | Prof. K. Jayasinghe (Anonymized cohort analytics) |
| **System Administrator** | `admin` | `admin123` | Model explorer, confusion matrices, rule inspector |

---

## 📊 AI Model Evaluation Benchmarks

Evaluated on 170 holdout test samples (80/20 train/test split of 850 undergraduate profiles):

| Metric | K-Nearest Neighbors (Primary) | Decision Tree (Baseline) |
|---|---|---|
| **Accuracy** | **89.41%** | **84.71%** |
| **Precision (Weighted)** | **89.65%** | **84.82%** |
| **Recall (Weighted)** | **89.41%** | **84.71%** |
| **F1-Score (Weighted)** | **89.28%** | **84.65%** |

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
