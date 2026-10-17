# CareerSense AI — System Architecture & Integration Guide

> **Document Type:** Project Architectural Analysis & Developer Integration Reference  
> **Repository:** `CareerSense_AI`  
> **Target Framework:** Streamlit (Python 3.11)  
> **Target Audience:** Developers, AI/ML Engineers, and System Architects

---

## Table of Contents
1. [Executive Overview](#1-executive-overview)
2. [Primary Tech Stack & Frameworks](#2-primary-tech-stack--frameworks)
3. [Repository Folder Structure & Main Entry Points](#3-repository-folder-structure--main-entry-points)
4. [Backend Architecture & Data Fetching Logic](#4-backend-architecture--data-fetching-logic)
5. [Integrating an External Python AI Model (FastAPI/Flask)](#5-integrating-an-external-python-ai-model-fastapiflask)
6. [Local Setup & Execution Guide](#6-local-setup--execution-guide)
7. [Default Demo Accounts & Roles](#7-default-demo-accounts--roles)

---

## 1. Executive Overview

**CareerSense AI** is an explainable, multi-tiered career exploration and personalized competency roadmap platform developed for undergraduate computing students (modeled for the Faculty of Computing, General Sir John Kotelawala Defence University - KDU).

Unlike typical black-box career recommendation tools, CareerSense combines three distinct intelligence layers:
1. **Machine Learning Layer:** A non-parametric K-Nearest Neighbors (K-NN) competency model with a Decision Tree diagnostic baseline.
2. **Rule-Based Expert System:** A transparent forward-chaining inference engine enforcing academic prerequisites, inter-skill dependencies, and readiness checks.
3. **Graph Search Optimization:** An A* search algorithm operating on a Directed Acyclic Graph (DAG) of learning activities to construct time-budgeted, prerequisite-aware weekly study roadmaps.

The platform enforces strict role-based access control (RBAC) across four user personas: **Students**, **Advisors**, **Coordinators**, and **Administrators**.

---

## 2. Primary Tech Stack & Frameworks

The application is built entirely as a **monolithic Python full-stack application**. There is no separate JavaScript/TypeScript frontend (e.g., React, Next.js, Vue) or Node.js runtime.

| Layer / Concern | Technology | Version | Purpose & Rationale |
| :--- | :--- | :--- | :--- |
| **Runtime Environment** | **Python** | `3.11.x` | **Strict requirement.** Pinned runtime to ensure binary compatibility with scikit-learn model artifacts (`.pkl`). |
| **Web UI Framework** | **Streamlit** | `1.50.0` | Reactive Python web framework. Handles UI components, routing via tabs, reactive state management via `st.session_state`, and custom CSS design injection (`modules/ui.py`). |
| **Data Visualization** | **Plotly** | `7.1.0` | Generates interactive polar/radar charts (`go.Scatterpolar`) comparing student competency against pathway benchmarks. |
| **Machine Learning** | **scikit-learn** | `1.6.1` | Primary K-NN classifier (`KNeighborsClassifier`), Decision Tree baseline (`DecisionTreeClassifier`), `StandardScaler`, and isolated 5-Fold Stratified Cross-Validation pipelines. |
| **Graph & Planning** | **NetworkX** | `3.4.2` | Graph construction and validation of skill dependency DAGs used by the custom A* learning planner. |
| **Data Processing** | **Pandas** & **NumPy** | `2.3.3` / `2.0.2` | Data manipulation, feature vector extraction, multiclass metrics (accuracy, precision, recall, F1, confusion matrices). |
| **Database & Storage** | **SQLite 3** | Embedded | Embedded relational database (`data/careersense.db`). Foreign keys and busy timeouts enforced (`PRAGMA foreign_keys = ON`). |
| **Security & Auth** | **hashlib / hmac** | Standard Library | Cryptographic PBKDF2-HMAC-SHA256 password hashing with constant-time verification (`hmac.compare_digest`) and legacy migration. |
| **Model Serialization** | **pickle / JSON** | Standard Library | Atomic file writing (`os.replace` on temporary files) with cryptographic SHA-256 provenance tracking. |

---

## 3. Repository Folder Structure & Main Entry Points

### 3.1 Directory Tree

```text
CareerSense_AI/
├── .env.example                       # Template for local environment variables
├── .gitattributes / .gitignore       # Git configurations and artifact ignores
├── .python-version                    # Python version pin (3.11)
├── app.py                             # ⭐ MAIN APPLICATION ENTRY POINT (Streamlit)
├── run_demo.py                        # Programmatic Python CLI wrapper for Streamlit
├── config.py                          # Global schemas, tracks, degree catalogs, weights
├── requirements.txt                   # Exact pinned dependencies
├── Launch_CareerSense.bat             # Shortcut Windows batch launcher
├── Start-CareerSense.bat              # Self-bootstrapping Windows launcher (creates .venv)
├── Start-CareerSense.command          # Self-bootstrapping macOS/Linux launcher
│
├── ai_engine/                         # AI & Algorithmic Core
│   ├── a_star_roadmap.py              # Layer 3: A* search algorithm for roadmap optimization
│   ├── explainability.py              # Explainability & feature attribution engine
│   ├── ml_classifier.py               # Layer 1: K-NN & Decision Tree ML models & metrics
│   ├── rule_engine.py                 # Layer 2: Forward-chaining rule expert system
│   └── saved_models/                  # Serialized ML artifacts & evaluation metrics
│       ├── decision_tree_model.pkl    # Serialized Decision Tree baseline
│       ├── evaluation_metrics.json    # Provenance metadata, 5-fold CV & holdout scores
│       ├── knn_model.pkl              # Serialized K-NN primary model
│       └── scaler.pkl                 # Serialized StandardScaler
│
├── database/                          # Persistence & Data Access Layer
│   ├── db_manager.py                  # DatabaseManager class (all SQLite queries & auth)
│   └── seed_data.py                   # Automatic database seeding script for demo data
│
├── data/                              # Static Knowledge Bases & Dynamic Database
│   ├── career_definitions.json        # 5 Career tracks & required skill benchmarks
│   ├── careersense.db                 # SQLite database file (generated / persisted)
│   ├── learning_graph.json            # Curated learning activities, prerequisites & hours
│   ├── rules_knowledge_base.json      # Prerequisite rules and gap weights
│   └── students_dataset.csv           # 850 synthetic student profiles for ML training
│
├── modules/                           # Role-Based Workspaces & Streamlit Views
│   ├── admin_view.py                  # Administrator workspace (diagnostics, model retrain)
│   ├── advisor_view.py                # Advisor workspace (review assigned students)
│   ├── auth.py                        # Authentication UI, sessions, registration
│   ├── coordinator_view.py            # Coordinator workspace (anonymized cohort analytics)
│   ├── report_generator.py            # Standalone HTML report generator
│   ├── student_view.py                # Student workspace (review, plan, evidence, academics)
│   └── ui.py                          # Design system, CSS theming, badges, headers
│
├── scripts/                           # Developer & Maintenance Utilities
│   ├── generate_dataset.py            # Generates synthetic student dataset (850 records)
│   ├── generate_progress_report_pdf.py# Generates formal audit/progress documentation
│   └── train_models.py                # Trains ML pipelines and exports .pkl artifacts
│
├── tests/                             # Verification & Test Suite
│   └── test_all.py                    # Unit & integration tests for all layers
│
└── docs/                              # Academic project documentation and audit reports
```

### 3.2 Main Application Entry Points

1. **`app.py` (Primary Production Entry Point)**
   - Initialized via `streamlit run app.py`.
   - Sets page configurations (`st.set_page_config`) and applies custom styling (`apply_theme()`).
   - Uses `@st.cache_resource` to instantiate and cache shared singletons:
     - `DatabaseManager`: Initializes SQLite tables and seeds demo data if empty.
     - `RuleBasedExpertSystem`: Loads rule sets from JSON knowledge bases.
     - `AStarRoadmapOptimizer`: Graph-based planner.
     - `CareerClassifier`: Automatically loads or trains the K-NN and Decision Tree models.
     - `ExplainabilityEngine`: Coordinates multi-factor explanations.
   - Dispatches authenticated sessions to role-specific views:
     - `student` $\rightarrow$ `render_student_view()`
     - `advisor` $\rightarrow$ `render_advisor_view()`
     - `coordinator` $\rightarrow$ `render_coordinator_view()`
     - `admin` $\rightarrow$ `render_admin_view()`

2. **`Start-CareerSense.bat` (Windows Production Launcher)**
   - Automated bootstrap script:
     1. Locates Python 3.11 (`py -3.11` or `python`).
     2. Automatically builds a private virtual environment in `.venv-windows`.
     3. Checks the SHA-256 hash of `requirements.txt` against a local stamp file; automatically runs `pip install` only when dependencies change.
     4. Executes `streamlit run app.py` and opens the browser.

3. **`run_demo.py` (Programmatic CLI Runner)**
   - Wraps Streamlit's internal CLI (`streamlit.web.cli.main()`) with predefined flags (`--server.port 8501`, `--server.address 127.0.0.1`, `--server.headless true`).

---

## 4. Backend Architecture & Data Fetching Logic

### 4.1 No Traditional HTTP REST Endpoints
CareerSense AI **does not employ a client-server REST or GraphQL API architecture** (such as FastAPI, Flask, Express, or Django REST Framework). 

Instead, it uses Streamlit's **in-process reactive execution model**:
- Every user interaction (button click, dropdown change, slider drag) triggers a top-to-bottom re-execution of the active view script.
- All "backend" logic, database transactions, and model inferences execute synchronously within the same Python process.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Application                           │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                      UI / View Layer                           │   │
│   │  (modules/student_view.py, advisor_view.py, admin_view.py)     │   │
│   └─────────────────┬──────────────────────────────┬───────────────┘   │
│                     │ In-Memory Python Calls       │                   │
│                     ▼                              ▼                   │
│   ┌─────────────────────────────────┐  ┌───────────────────────────┐   │
│   │     Data Access Layer (DAL)     │  │       AI Engine Core      │   │
│   │    (database/db_manager.py)     │  │    (ai_engine/*.py)       │   │
│   └─────────────────┬───────────────┘  └─────────────┬─────────────┘   │
│                     │ SQL Queries                    │ JSON & Models   │
│                     ▼                              ▼                   │
│   ┌─────────────────────────────────┐  ┌───────────────────────────┐   │
│   │   SQLite Database (Local File)  │  │   JSON Rules & .pkl Files │   │
│   │     (data/careersense.db)       │  │ (data/*.json, models/*.pkl│   │
│   └─────────────────────────────────┘  └───────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Access Layer (`database/db_manager.py`)
All data persistence and retrieval goes through the `DatabaseManager` class. Key methods include:

- **Authentication & User Management:**
  - `authenticate_user(username, password)`: Constant-time PBKDF2 hash verification.
  - `create_user(username, password, role, full_name, reg_no, email)`: User creation with constraints.
- **Student Profile & Academics:**
  - `get_student_profile(user_id)` / `save_student_profile(...)`: Manages GPA, target track, weekly study hours, and CV metadata.
  - `get_academic_records(user_id)` / `set_academic_records(user_id, records)`: Stores module grades and recalculates cumulative GPA.
- **Skills & Interests:**
  - `get_student_skills(user_id)` / `set_student_skill(user_id, skill_name, level, evidence_status)`: Tracks skills and flags evidence as `self_reported` or `advisor_verified`.
  - `get_student_interests(user_id)` / `set_student_interests(user_id, interests_dict)`: Stated domain affinities (1 to 5 scale).
- **Roadmaps & Evidence:**
  - `get_roadmap(user_id, target_career)` / `save_roadmap(...)` / `update_roadmap_activity_status(...)`.
  - `get_student_projects(...)` / `add_student_project(...)` / `verify_student_project(...)`.
  - `get_student_documents(...)` / `add_student_document(...)` (Stores document metadata and SHA-256 fingerprint; files are not stored on disk for privacy).
- **Role-Scoped Queries:**
  - `get_students_for_advisor(advisor_id)`: Enforces assignment boundaries.
  - `get_all_students_summary(actor_role)`: Implements small-cohort data suppression for coordinator privacy.

### 4.3 AI & Planning Pipelines
1. **Pathway Match Scoring (`ai_engine/ml_classifier.py`):**
   - Inputs: 6 academic subject grades, 25 skill proficiency levels, stated domain interests, and degree program.
   - Calculation:
     $$\text{Advisory Affinity} = 0.70 \times \text{K-NN Competency} + 0.20 \times \text{Stated Interests} + 0.10 \times \text{Degree Prior}$$
   - *Note:* The user's target selection is intentionally excluded from the feature vector to prevent bias.
2. **Rule Verification (`ai_engine/rule_engine.py`):**
   - Checks academic prerequisites against configured minimum grade thresholds.
   - Computes competency gaps against benchmark targets in `career_definitions.json`.
3. **Roadmap Generation (`ai_engine/a_star_roadmap.py`):**
   - Runs A* search on `data/learning_graph.json` to minimize total declared activity hours.
   - Schedules activities into sequential weeks based on the student's `weekly_hours` budget.

---

## 5. Integrating an External Python AI Model (FastAPI/Flask)

If you have an existing local Python-based AI service (e.g., a FastAPI or Flask microservice hosting an LLM, deep learning model, or specialized classifier), here are the recommended integration patterns.

### 5.1 Architecture Overview with External Service

```text
┌─────────────────────────────────┐                  ┌──────────────────────────────────┐
│      CareerSense AI System      │                  │   Local AI Microservice          │
│        (Streamlit App)          │                  │     (FastAPI or Flask)           │
│                                 │                  │                                  │
│   ┌──────────────────────────┐  │   HTTP POST      │   ┌───────────────────────────┐  │
│   │ External AI Client       │──┼─────────────────►│   │ POST /api/v1/predict      │  │
│   │ (ai_engine/ext_client.py)│◄─┼──────────────────┤   │ POST /api/v1/parse-cv     │  │
│   └─────────────┬────────────┘  │   JSON Response  │   └─────────────┬─────────────┘  │
│                 ▼               │                  │                 ▼                │
│   ┌──────────────────────────┐  │                  │   ┌───────────────────────────┐  │
│   │ UI Views (student_view)  │  │                  │   │ PyTorch / Transformers /  │  │
│   │ or ML Classifier         │  │                  │   │ LLM / Custom Model        │  │
│   └──────────────────────────┘  │                  │   └───────────────────────────┘  │
└─────────────────────────────────┘                  └──────────────────────────────────┘
```

---

### 5.2 Scenario A: Replacing or Augmenting the Career Prediction Classifier

#### 1. The FastAPI Service (`external_service/main.py`)
Suppose your FastAPI service runs at `http://127.0.0.1:8000`:

```python
# external_service/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI(title="Advanced AI Career Predictor")

class StudentData(BaseModel):
    degree: str
    academic_records: List[Dict]
    skills: Dict[str, str]
    interests: Dict[str, int]
    target_career: Optional[str] = None

class PathwayMatch(BaseModel):
    track: str
    match_score: float
    confidence: float
    explanation: str

@app.post("/api/v1/predict-career", response_model=List[PathwayMatch])
def predict_career(student: StudentData):
    # Process student data through your custom model (PyTorch, TensorFlow, LLM, etc.)
    # Example placeholder response:
    return [
        {
            "track": "Data Science / AI",
            "match_score": 88.5,
            "confidence": 0.91,
            "explanation": "Strong foundation in Mathematics & Statistics with Python proficiency."
        },
        {
            "track": "Software Engineering",
            "match_score": 76.2,
            "confidence": 0.84,
            "explanation": "Solid programming grades, but gaps identified in automated QA."
        }
    ]
```

#### 2. The Client Adapter in CareerSense (`ai_engine/external_ai_client.py`)
Create a new adapter file (or place inside `ai_engine/`):

```python
# ai_engine/external_ai_client.py
import requests
import os
from typing import Dict, List, Optional

API_URL = os.getenv("EXTERNAL_AI_ENDPOINT", "http://127.0.0.1:8000/api/v1/predict-career")
TIMEOUT_SECONDS = 3.0

class ExternalCareerModelClient:
    @staticmethod
    def predict(academic_records: list, skills: dict, interests: dict, degree: str) -> Optional[List[Dict]]:
        payload = {
            "degree": degree,
            "academic_records": academic_records,
            "skills": skills,
            "interests": interests
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            # Fallback gracefully to internal K-NN if external API is unreachable
            print(f"[Warning] External AI service unavailable: {e}")
            return None
```

#### 3. Where to Hook into `ai_engine/ml_classifier.py`
In `ai_engine/ml_classifier.py`, locate `predict_career_matches(...)` (around line 634). You can check if the external model is available, and fall back to the built-in K-NN if it fails:

```python
# Inside CareerClassifier.predict_career_matches(...)
from ai_engine.external_ai_client import ExternalCareerModelClient

# Attempt external prediction:
ext_result = ExternalCareerModelClient.predict(academic_records, skills, interests, degree)
if ext_result:
    return {
        "ranked_matches": ext_result,
        "model_used": "External FastAPI Neural Model",
        "is_actionable": True
    }

# Otherwise, continue with internal K-NN & composite calculation...
```

---

### 5.3 Scenario B: Resume / CV Automated Skill Extraction (NLP / LLM)

Currently, CareerSense only hashes uploaded CV files (`modules/student_view.py`, lines 317–335) for privacy and records `cv_extracted_skills` as a JSON stub in the database. You can connect your Flask/FastAPI service to extract technical skills from uploaded documents.

#### 1. The FastAPI Resume Parser Endpoint
```python
# In your FastAPI server
from fastapi import FastAPI, UploadFile, File
import io

@app.post("/api/v1/extract-skills")
async def extract_skills(file: UploadFile = File(...)):
    content = await file.read()
    # Run NLP / NER / LLM parser on `content`
    extracted = ["Python Data Stack (Pandas/NumPy)", "REST APIs & Web Services", "Containerization (Docker)"]
    return {"extracted_skills": extracted, "inferred_experience_level": "Intermediate"}
```

#### 2. Hooking into `modules/student_view.py`
In `modules/student_view.py`, under Tab 3 (Evidence $\rightarrow$ Documents, line 318):

```python
# When user uploads a file:
uploaded = st.file_uploader("Choose a PDF, DOCX, or text file", type=["pdf", "docx", "txt"])
if st.button("Record document & extract skills"):
    if uploaded:
        # 1. Send file to your local FastAPI endpoint
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        try:
            res = requests.post("http://127.0.0.1:8000/api/v1/extract-skills", files=files, timeout=5)
            if res.status_code == 200:
                skills_found = res.json().get("extracted_skills", [])
                st.success(f"Extracted {len(skills_found)} skills from CV!")
                # Automatically save extracted skills into student profile in SQLite:
                for skill in skills_found:
                    db.set_student_skill(user_id, skill, "Intermediate", evidence_status="self_reported")
        except Exception as err:
            st.warning("CV uploaded, but local skill extraction endpoint was unreachable.")
```

---

## 6. Local Setup & Execution Guide

### 6.1 Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python Version:** **Python 3.11** (Strictly required for scikit-learn binary serialization compatibility).
  - Verify your installation:
    ```powershell
    py -3.11 --version
    # or
    python --version
    ```

---

### 6.2 Setup Commands (Windows PowerShell / Command Prompt)

```powershell
# 1. Navigate to the repository root
cd C:\Users\User\Desktop\Aqeel\CareerSense_AI

# 2. Create a virtual environment using Python 3.11
py -3.11 -m venv .venv

# 3. Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt (cmd.exe):
.venv\Scripts\activate.bat

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install pinned project dependencies
pip install -r requirements.txt

# 6. Train the machine learning models (generates .pkl files and benchmarks)
python scripts/train_models.py

# 7. (Optional) Run the complete automated test suite
python -m unittest discover -s tests -p "test_*.py"

# 8. Start the Streamlit application
streamlit run app.py
```

---

### 6.3 Setup Commands (macOS / Linux)

```bash
# 1. Navigate to repository root
cd CareerSense_AI

# 2. Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Train models
python scripts/train_models.py

# 5. Run tests
python -m unittest discover -s tests -p "test_*.py"

# 6. Start the web app
streamlit run app.py
```

---

### 6.4 One-Click Quick Launchers (No terminal required)
- **Windows:** Double-click `Start-CareerSense.bat` or `Launch_CareerSense.bat`.
- **macOS:** Double-click `Start-CareerSense.command` (right-click and select "Open" on first run).

Once launched, access the application in your browser at:  
👉 **`http://localhost:8501`**

---

## 7. Default Demo Accounts & Roles

The system comes pre-seeded with synthetic test accounts for local development and verification:

| Role | Username | Password | Purpose & Capabilities |
| :--- | :--- | :--- | :--- |
| **Student** | `student_demo` | `student123` | Explore career pathway recommendations, record module grades, assess skills, upload documents, and generate A* study roadmaps. |
| **Advisor** | `advisor` | `advisor123` | View assigned students, inspect evidence (self-reported vs. verified), verify student projects/documents, and leave guidance notes. |
| **Coordinator** | `coordinator` | `coordinator123` | Access aggregated cohort statistics and curriculum demand graphs. Identifiers are masked, and cohorts $< 5$ are suppressed for privacy. |
| **Administrator** | `admin` | `admin123` | View model evaluation metrics, feature importances, cross-validation scores, inspect knowledge-base rules, and retrain models. |

---

## 8. Summary & Key Architectural Takeaways

1. **Monolithic Pure-Python Stack:** No Node.js, React, or Next.js overhead. Built with Streamlit, Plotly, scikit-learn, and SQLite.
2. **Deterministic & Explainable:** Recommendations combine K-NN peer similarity (70%), stated student interests (20%), and degree alignment (10%) with explicit prerequisite checks.
3. **Seamless External AI Integration:** An external FastAPI or Flask model can be integrated in minutes by intercepting `ai_engine/ml_classifier.py` or augmenting document processing in `modules/student_view.py` using standard HTTP calls (`requests`).
4. **Environment Isolation:** Python 3.11 is required to ensure scikit-learn model serialization contracts are maintained.
