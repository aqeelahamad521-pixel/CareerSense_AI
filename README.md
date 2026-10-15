# CareerSense

CareerSense is an academic career-exploration prototype for undergraduate computing students. It combines a competency model, explicit prerequisite rules, and a dependency-aware learning planner. It is designed to support discussion and planning—not to predict employment, certify readiness, or make placement decisions.

## What it does

- Compares five computing pathways using recorded coursework and self-assessed skills.
- Keeps competency signals separate from stated interests.
- Produces an uncalibrated advisory distribution using:
  - 70% K-NN competency alignment;
  - 20% stated domain interests;
  - 10% degree alignment.
- Highlights the selected target without using it as a scoring input.
- Abstains from a top recommendation when fewer than two academic or skill evidence items exist.
- Applies transparent academic-prerequisite and skill-dependency rules.
- Builds a learning sequence with A* search, minimizing declared activity hours before scheduling the result to a weekly study budget.
- Distinguishes self-reported/submitted evidence from advisor-reviewed evidence.
- Gives advisors access only to assigned students and gives coordinators anonymized aggregate data.

## Critical limitations

The 850-profile dataset is synthetic. Its labels and features are generated from hand-authored assumptions, so model metrics measure recovery of those assumptions. They do not establish real-world career suitability, fairness, employability, or outcome accuracy.

The “Prototype readiness index” is a profile and benchmark coverage indicator. It is not a hiring or internship-readiness score. Documents are not retained or parsed: the prototype stores sanitized metadata and a SHA-256 fingerprint only.

See [docs/project_audit_report.md](docs/project_audit_report.md) for the baseline audit and [docs/remediation_report.md](docs/remediation_report.md) for the completed work and remaining deployment prerequisites.

## Architecture

```text
Streamlit workspaces
├── Student: pathway review, plan, evidence, academics, report
├── Advisor: assigned students, evidence review, guidance
├── Coordinator: anonymized aggregates with small-cohort suppression
└── Administrator: model/rule/graph diagnostics and maintenance
        │
        ├── K-NN competency model + Decision Tree diagnostic baseline
        ├── Rule-based prerequisites and competency gaps
        ├── A* minimum-declared-hours learning sequence
        └── SQLite with foreign keys and role-aware access methods
```

The competency model uses 31 features: six academic subject summaries and 25 skill levels. Interests are deliberately excluded from this feature vector and included once in the transparent advisory composite.

## Run locally

Python 3.11 is the supported artifact runtime. Exact dependency versions are pinned so saved scikit-learn artifacts fail closed when the runtime is incompatible.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train_models.py
python -m unittest discover -s tests -p "test_*.py"
streamlit run app.py
```

Open `http://localhost:8501`.

### One-click startup

- **macOS:** double-click `Start-CareerSense.command`. If macOS blocks the first launch, Control-click the file, choose **Open**, and confirm once.
- **Windows:** double-click `Start-CareerSense.bat`.

The launcher creates a private platform-specific environment, installs the exact dependencies on the first run or after `requirements.txt` changes, opens the app in the default browser, and keeps a diagnostic window visible if startup fails. Python 3.11 must already be installed.

## Demo accounts

These local-only accounts use synthetic data:

| Role | Username | Password |
|---|---|---|
| Student | `student_demo` | `student123` |
| Advisor | `advisor` | `advisor123` |
| Coordinator | `coordinator` | `coordinator123` |
| Administrator | `admin` | `admin123` |

Do not reuse these credentials outside the local prototype.

## Verification

The test suite covers metric mathematics, exact composite behavior, target independence, low-evidence abstention, A* coverage and cost properties, readiness edge cases, report escaping, foreign-key enforcement, owner-scoped roadmap updates, advisor assignment boundaries, evidence states, and password migration.

Model evaluation metadata is stored in `ai_engine/saved_models/evaluation_metrics.json`, including the feature schema, exact dependency versions, dataset hash, synthetic-data disclosure, and the held-out evaluation of the production composite.

## Project layout

```text
ai_engine/      Models, rules, explainability, and A* planner
database/       SQLite manager and synthetic demo seed
data/           Synthetic dataset and curated JSON knowledge bases
modules/        Streamlit role workspaces and report generator
scripts/        Dataset generation and model training
tests/          Unit and integration regression suite
docs/           Audit and academic project documentation
```
