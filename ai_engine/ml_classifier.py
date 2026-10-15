"""
AI Layer 1: Machine Learning Career Classifier for CareerSense AI.
Implements:
1. K-Nearest Neighbors (K-NN) as Primary Classifier (with neighbor-based proximity)
2. Decision Tree Classifier as Interpretable Baseline (with Gini feature importances)
3. Standard feature scaling and vector extraction
4. Rigorous multiclass evaluation metrics (Accuracy, Weighted/Macro Precision, Recall, F1, Confusion Matrix)
5. Multi-model benchmarking (Zero-Rule Dummy, Decision Tree, Logistic Regression, Random Forest, K-NN)
6. Isolated 5-Fold Stratified Cross-Validation pipelines to prevent preprocessing data leakage
7. Full provenance tracking (dataset SHA256, timestamps, random seeds, dependency versions)
"""
import os
import sys
import pickle
import json
import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import (
    DATASET_PATH, MODELS_DIR, CAREER_TRACKS,
    ML_FEATURE_COLUMNS, INTEREST_FEATURE_COLUMNS, GRADE_POINTS, SKILL_LEVELS,
    DEGREE_CAREER_ALIGNMENT
)

ARTIFACT_SCHEMA_VERSION = 2

INTEREST_TO_TRACK = {
    "interest_software": "Software Engineering",
    "interest_data": "Data Science / AI",
    "interest_security": "Cybersecurity",
    "interest_cloud": "Cloud / DevOps",
    "interest_design": "UI/UX Design",
}

RUNTIME_INTEREST_TO_TRACK = {
    "Software Development & Systems": "Software Engineering",
    "Data Analysis & AI Research": "Data Science / AI",
    "Cybersecurity & Threat Defense": "Cybersecurity",
    "Cloud Infrastructure & Automation": "Cloud / DevOps",
    "UI/UX Design & User Experience": "UI/UX Design",
}

def compute_multiclass_metrics(y_true, y_pred, labels):
    """
    Computes mathematically rigorous multiclass classification metrics.
    Guarantees that Accuracy, Precision, Recall, and F1 are independently calculated.
    Correctly handles edge cases, zero-division, and missing classes.
    Validates input lengths, non-emptiness, label sets, and uniqueness.
    """
    if y_true is None or y_pred is None:
        raise TypeError("y_true and y_pred cannot be None.")
    
    y_true = list(y_true)
    y_pred = list(y_pred)
    
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays.")
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} does not match len(y_pred)={len(y_pred)}.")
    
    if labels is None or not hasattr(labels, "__iter__"):
        raise ValueError("labels must be an iterable collection of valid class labels.")
        
    labels_list = list(labels)
    if len(labels_list) == 0:
        raise ValueError("labels list must contain at least one class label.")
        
    if len(set(labels_list)) != len(labels_list):
        raise ValueError(f"labels must contain unique, non-duplicated class labels. Given: {labels_list}")
        
    for l in labels_list:
        if l is None or (isinstance(l, str) and not l.strip()):
            raise ValueError(f"Invalid label found in labels: {repr(l)}")
            
    supported_labels = set(labels_list)
    unsupported_true = set(y_true) - supported_labels
    unsupported_pred = set(y_pred) - supported_labels
    
    if unsupported_true:
        raise ValueError(f"Found unsupported labels in y_true not present in configured labels: {sorted(list(unsupported_true))}")
    if unsupported_pred:
        raise ValueError(f"Found unsupported labels in y_pred not present in configured labels: {sorted(list(unsupported_pred))}")
        
    n = len(y_true)
    label_to_idx = {l: i for i, l in enumerate(labels_list)}
    k = len(labels_list)
    cm = [[0 for _ in range(k)] for _ in range(k)]
    correct = 0
    
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_idx[yt]][label_to_idx[yp]] += 1
        if yt == yp:
            correct += 1
            
    acc = correct / n
    
    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0
    
    for i, label in enumerate(labels_list):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(k) if r != i)
        fn = sum(cm[i][c] for c in range(k) if c != i)
        support = sum(cm[i][c] for c in range(k))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per_class[label] = {
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "support": int(support)
        }
        
        macro_p += prec
        macro_r += rec
        macro_f1 += f1
        
        weighted_p += prec * support
        weighted_r += rec * support
        weighted_f1 += f1 * support
        
    macro_p = (macro_p / k) * 100
    macro_r = (macro_r / k) * 100
    macro_f1 = (macro_f1 / k) * 100
    
    total_support = n
    weighted_p = (weighted_p / total_support) * 100 if total_support > 0 else 0.0
    weighted_r = (weighted_r / total_support) * 100 if total_support > 0 else 0.0
    weighted_f1 = (weighted_f1 / total_support) * 100 if total_support > 0 else 0.0
    
    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(weighted_p, 2),
        "recall": round(weighted_r, 2),
        "f1_score": round(weighted_f1, 2),
        "precision_macro": round(macro_p, 2),
        "recall_macro": round(macro_r, 2),
        "f1_score_macro": round(macro_f1, 2),
        "confusion_matrix": cm,
        "per_class": per_class,
        "labels": labels_list
    }

class CareerClassifier:
    def __init__(self):
        self.knn_model = None
        self.dt_model = None
        self.scaler = None
        self.classes_ = CAREER_TRACKS
        self.is_trained = False
        self.metrics = {}
        self.knn_path = MODELS_DIR / "knn_model.pkl"
        self.dt_path = MODELS_DIR / "dt_model.pkl"
        self.scaler_path = MODELS_DIR / "scaler.pkl"
        self.metrics_path = MODELS_DIR / "evaluation_metrics.json"
        self.compatibility_status = "not_checked"
        self.compatibility_reason = ""
        self.load_models()

    def extract_features(self, academic_records: list, interests: dict, skills: dict) -> np.ndarray:
        """Transforms raw student attributes into a feature vector matching ML_FEATURE_COLUMNS."""
        grade_map = {
            "grade_programming": 0.0,
            "grade_math": 0.0,
            "grade_database": 0.0,
            "grade_networking": 0.0,
            "grade_systems": 0.0,
            "grade_design": 0.0
        }
        subj_to_key = {
            "Programming": "grade_programming",
            "Mathematics & Statistics": "grade_math",
            "Databases": "grade_database",
            "Networking": "grade_networking",
            "Operating Systems & Architecture": "grade_systems",
            "Human Computer Interaction & Design": "grade_design"
        }
        for rec in academic_records:
            s_name = rec.get("subject_area")
            if s_name in subj_to_key:
                k = subj_to_key[s_name]
                grade_map[k] = max(grade_map[k], rec.get("grade_points", GRADE_POINTS.get(rec.get("grade", "C"), 2.0)))

        # Harmonize skills with completed academic modules
        combined_skills = dict(skills) if skills else {}
        module_skill_inferences = {
            "Structured Programming": ("Object-Oriented Programming", "Beginner"),
            "Object-Oriented": ("Object-Oriented Programming", "Intermediate"),
            "Algorithms": ("Data Structures & Algorithms", "Intermediate"),
            "Data Structures": ("Data Structures & Algorithms", "Intermediate"),
            "Quality Assurance": ("Automated Testing & QA", "Intermediate"),
            "Testing": ("Automated Testing & QA", "Intermediate"),
            "Architecture": ("REST APIs & Web Services", "Intermediate"),
            "Web Technologies": ("Frontend Framework Awareness", "Intermediate"),
            "Database": ("SQL & Data Querying", "Intermediate"),
            "Big Data": ("SQL & Data Querying", "Intermediate"),
            "Python": ("Python Data Stack (Pandas/NumPy)", "Intermediate"),
            "Statistics": ("Statistical Analysis", "Intermediate"),
            "Probability": ("Statistical Analysis", "Intermediate"),
            "Machine Learning": ("Machine Learning & Modeling", "Intermediate"),
            "Visualization": ("Data Visualization", "Intermediate"),
            "Networks": ("Network Security & Protocols", "Intermediate"),
            "Routing": ("Network Security & Protocols", "Intermediate"),
            "Linux": ("Linux Systems Administration", "Intermediate"),
            "System Administration": ("Linux Systems Administration", "Intermediate"),
            "Cloud": ("Cloud Computing (AWS/GCP/Azure)", "Intermediate"),
            "DevOps": ("Containerization (Docker)", "Intermediate"),
            "Cyber Defense": ("Vulnerability Assessment", "Intermediate"),
            "Hardware Security": ("Cryptography Fundamentals", "Intermediate"),
            "Microprocessor": ("Linux Systems Administration", "Intermediate"),
            "Business Intelligence": ("SQL & Data Querying", "Intermediate"),
            "Business Process": ("User Research & Usability Testing", "Intermediate"),
            "Digital Design": ("Wireframing & Prototyping (Figma)", "Intermediate"),
            "E-Commerce": ("Design Principles & Typography", "Intermediate")
        }
        for rec in academic_records:
            m_title = rec.get("module_name", "")
            g_pt = rec.get("grade_points", GRADE_POINTS.get(rec.get("grade", "C"), 2.0))
            if g_pt >= 2.0:
                for phrase, (sk_name, base_lvl) in module_skill_inferences.items():
                    if phrase.lower() in m_title.lower():
                        cur_lvl = combined_skills.get(sk_name, "None")
                        if SKILL_LEVELS.get(cur_lvl, 0) < SKILL_LEVELS.get(base_lvl, 1):
                            combined_skills[sk_name] = base_lvl

        skill_attr_map = {
            "skill_oop": combined_skills.get("Object-Oriented Programming", "None"),
            "skill_dsa": combined_skills.get("Data Structures & Algorithms", "None"),
            "skill_web_api": combined_skills.get("REST APIs & Web Services", "None"),
            "skill_testing": combined_skills.get("Automated Testing & QA", "None"),
            "skill_git": combined_skills.get("Version Control (Git)", "None"),
            "skill_python_data": combined_skills.get("Python Data Stack (Pandas/NumPy)", "None"),
            "skill_ml": combined_skills.get("Machine Learning & Modeling", "None"),
            "skill_visualization": combined_skills.get("Data Visualization", "None"),
            "skill_sql": combined_skills.get("SQL & Data Querying", "None"),
            "skill_statistics": combined_skills.get("Statistical Analysis", "None"),
            "skill_network_sec": combined_skills.get("Network Security & Protocols", "None"),
            "skill_cryptography": combined_skills.get("Cryptography Fundamentals", "None"),
            "skill_linux": combined_skills.get("Linux Systems Administration", "None"),
            "skill_vuln_assess": combined_skills.get("Vulnerability Assessment", "None"),
            "skill_secure_code": combined_skills.get("Secure Coding Practices", "None"),
            "skill_docker": combined_skills.get("Containerization (Docker)", "None"),
            "skill_cicd": combined_skills.get("CI/CD Pipelines", "None"),
            "skill_cloud_infra": combined_skills.get("Cloud Computing (AWS/GCP/Azure)", "None"),
            "skill_iac": combined_skills.get("Infrastructure as Code", "None"),
            "skill_monitoring": combined_skills.get("System Monitoring & Logging", "None"),
            "skill_user_research": combined_skills.get("User Research & Usability Testing", "None"),
            "skill_figma": combined_skills.get("Wireframing & Prototyping (Figma)", "None"),
            "skill_design_principles": combined_skills.get("Design Principles & Typography", "None"),
            "skill_frontend": combined_skills.get("Frontend Framework Awareness", "None"),
            "skill_design_systems": combined_skills.get("Design Systems & Component Design", "None")
        }

        vec = []
        for col in ML_FEATURE_COLUMNS:
            if col in grade_map:
                vec.append(float(grade_map[col]))
            elif col in skill_attr_map:
                lvl_val = SKILL_LEVELS.get(skill_attr_map[col], 0)
                vec.append(float(lvl_val))
            else:
                vec.append(0.0)

        return np.array(vec, dtype=np.float64).reshape(1, -1)

    def _composite_distribution(self, competency_dist: dict, interest_dist: dict, degree: str = None) -> dict:
        """Exact production advisory composite. The selected target is never an input."""
        degree_dist = DEGREE_CAREER_ALIGNMENT.get(degree, {track: 0.20 for track in self.classes_})
        raw = {
            track: (0.70 * competency_dist.get(track, 0.0))
            + (0.20 * interest_dist.get(track, 0.20))
            + (0.10 * degree_dist.get(track, 0.20))
            for track in self.classes_
        }
        total = sum(raw.values()) or 1.0
        return {track: value / total for track, value in raw.items()}

    def train_models(self, dataset_path=DATASET_PATH) -> dict:
        """
        Trains and evaluates K-NN, Decision Tree, Logistic Regression, Random Forest,
        and Dummy baselines with isolated cross-validation pipelines (no data leakage)
        and strict evaluation on a held-out test set.
        """
        # Scikit-learn is required for rigorous training and cross-validation
        try:
            import sklearn
            from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
            from sklearn.pipeline import Pipeline
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.dummy import DummyClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise RuntimeError(
                "Scikit-learn is required for model training and cross-validation. "
                "Please install dependencies with: pip install -r requirements.txt"
            ) from e

        if not os.path.exists(dataset_path):
            from scripts.generate_dataset import generate_student_dataset
            generate_student_dataset()

        # Compute provenance hash of the dataset
        dataset_bytes = Path(dataset_path).read_bytes()
        dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()

        df = pd.read_csv(dataset_path)
        X = np.array(df[ML_FEATURE_COLUMNS].values, dtype=np.float64)
        y = np.array(df["career_track"].tolist())
        labels = list(self.classes_)

        class_dist = {trk: int(np.sum(y == trk)) for trk in labels}

        # 80/20 Stratified train/test split (Held-out test set used ONLY for final evaluation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # 5-Fold Stratified Cross-Validation on the training split only.
        # Uses Pipeline with StandardScaler so each fold fits scaling independently on its train fold (ZERO LEAKAGE).
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        knn_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=1))
        ])
        knn_cv_scores = cross_val_score(knn_pipe, X_train, y_train, cv=cv, scoring="accuracy")

        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42))
        ])
        lr_cv_scores = cross_val_score(lr_pipe, X_train, y_train, cv=cv, scoring="accuracy")

        dt_clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, random_state=42)
        dt_cv_scores = cross_val_score(dt_clf, X_train, y_train, cv=cv, scoring="accuracy")

        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        rf_cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=cv, scoring="accuracy")

        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_cv_scores = cross_val_score(dummy_clf, X_train, y_train, cv=cv, scoring="accuracy")

        # Fit final models on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.knn_model = KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=1)
        self.knn_model.fit(X_train_scaled, y_train)

        self.dt_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, random_state=42)
        self.dt_model.fit(X_train, y_train)

        final_lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
        final_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(X_train, y_train)
        final_dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

        # Evaluate on the held-out test split (170 samples)
        knn_preds = self.knn_model.predict(X_test_scaled)
        knn_metrics = compute_multiclass_metrics(y_test, knn_preds, labels)
        knn_metrics["name"] = "K-Nearest Neighbors (Primary Model)"
        knn_metrics["cv_accuracy_mean"] = round(float(np.mean(knn_cv_scores)) * 100, 2)
        knn_metrics["cv_accuracy_std"] = round(float(np.std(knn_cv_scores)) * 100, 2)
        knn_metrics["cv_fold_scores"] = [round(float(s) * 100, 2) for s in knn_cv_scores]

        dt_preds = self.dt_model.predict(X_test)
        dt_metrics = compute_multiclass_metrics(y_test, dt_preds, labels)
        dt_metrics["name"] = "Decision Tree Classifier (Baseline Model)"
        dt_metrics["cv_accuracy_mean"] = round(float(np.mean(dt_cv_scores)) * 100, 2)
        dt_metrics["cv_accuracy_std"] = round(float(np.std(dt_cv_scores)) * 100, 2)
        dt_metrics["cv_fold_scores"] = [round(float(s) * 100, 2) for s in dt_cv_scores]

        importances = [round(float(val), 4) for val in self.dt_model.feature_importances_]
        dt_importances = dict(zip(ML_FEATURE_COLUMNS, importances))
        top_features = sorted(dt_importances.items(), key=lambda x: -x[1])[:8]
        dt_metrics["top_features"] = top_features

        # Evaluate the exact composite exposed by the product on the held-out
        # synthetic split. This is still a simulation benchmark, not evidence of
        # employment outcomes, but it prevents the UI from advertising a metric
        # for a different scoring function.
        composite_preds = []
        # train_test_split is repeated on row indices with the same seed so the
        # auxiliary degree/interest fields align exactly with X_test/y_test.
        row_indices = np.arange(len(df))
        _, test_indices = train_test_split(
            row_indices, test_size=0.20, random_state=42, stratify=y
        )
        knn_test_probs = self.knn_model.predict_proba(X_test_scaled)
        for row_idx, prob_row in zip(test_indices, knn_test_probs):
            competency = {
                track: float(prob_row[list(self.knn_model.classes_).index(track)])
                if track in self.knn_model.classes_ else 0.0
                for track in self.classes_
            }
            interest_raw = {
                track: float(df.iloc[row_idx][feature])
                for feature, track in INTEREST_TO_TRACK.items()
            }
            interest_total = sum(interest_raw.values()) or 1.0
            interest_dist = {track: value / interest_total for track, value in interest_raw.items()}
            composite = self._composite_distribution(
                competency, interest_dist, str(df.iloc[row_idx]["degree"])
            )
            composite_preds.append(max(composite, key=composite.get))
        composite_metrics = compute_multiclass_metrics(y_test, composite_preds, labels)
        composite_metrics["name"] = "Production advisory composite (synthetic holdout)"
        composite_metrics["formula"] = "70% K-NN competency + 20% stated interests + 10% degree alignment"
        composite_metrics["limitations"] = (
            "Evaluated only against synthetic generator labels; this does not validate career outcomes, "
            "fairness, employability, or causal recommendations."
        )
        subgroup_diagnostics = {}
        test_degrees = [str(df.iloc[row_idx]["degree"]) for row_idx in test_indices]
        for degree_name in sorted(set(test_degrees)):
            positions = [i for i, value in enumerate(test_degrees) if value == degree_name]
            degree_true = [y_test.iloc[i] if hasattr(y_test, "iloc") else y_test[i] for i in positions]
            degree_pred = [composite_preds[i] for i in positions]
            degree_metrics = compute_multiclass_metrics(degree_true, degree_pred, labels)
            subgroup_diagnostics[degree_name] = {
                "sample_count": len(positions),
                "accuracy": degree_metrics["accuracy"],
                "f1_macro": degree_metrics["f1_score_macro"],
                "accuracy_delta_from_overall": round(
                    degree_metrics["accuracy"] - composite_metrics["accuracy"], 2
                ),
            }
        composite_metrics["synthetic_subgroup_diagnostic"] = {
            "attribute": "degree_programme",
            "groups": subgroup_diagnostics,
            "limitations": (
                "A diagnostic on synthetic labels only. It can reveal simulated performance gaps, "
                "but it is not a real-world fairness audit and excludes protected attributes."
            ),
        }

        dummy_preds = final_dummy.predict(X_test)
        dummy_metrics = compute_multiclass_metrics(y_test, dummy_preds, labels)

        lr_preds = final_lr.predict(X_test_scaled)
        lr_metrics = compute_multiclass_metrics(y_test, lr_preds, labels)

        rf_preds = final_rf.predict(X_test)
        rf_metrics = compute_multiclass_metrics(y_test, rf_preds, labels)

        model_comparison = [
            {
                "model": "Zero-Rule (Dummy Baseline)",
                "type": "Baseline",
                "cv_accuracy": round(float(np.mean(dummy_cv_scores)) * 100, 2),
                "accuracy": dummy_metrics["accuracy"],
                "f1_macro": dummy_metrics["f1_score_macro"],
                "f1_weighted": dummy_metrics["f1_score"],
                "rationale": "Majority-class baseline confirming non-trivial learning"
            },
            {
                "model": "Decision Tree Classifier",
                "type": "Interpretable Baseline",
                "cv_accuracy": dt_metrics["cv_accuracy_mean"],
                "accuracy": dt_metrics["accuracy"],
                "f1_macro": dt_metrics["f1_score_macro"],
                "f1_weighted": dt_metrics["f1_score"],
                "rationale": "White-box rule-interpretable tree benchmark (max depth 6)"
            },
            {
                "model": "Random Forest Classifier",
                "type": "Ensemble Benchmark",
                "cv_accuracy": round(float(np.mean(rf_cv_scores)) * 100, 2),
                "accuracy": rf_metrics["accuracy"],
                "f1_macro": rf_metrics["f1_score_macro"],
                "f1_weighted": rf_metrics["f1_score"],
                "rationale": "Ensemble benchmark evaluating non-linear feature interactions"
            },
            {
                "model": "Multinomial Logistic Regression",
                "type": "Linear Benchmark",
                "cv_accuracy": round(float(np.mean(lr_cv_scores)) * 100, 2),
                "accuracy": lr_metrics["accuracy"],
                "f1_macro": lr_metrics["f1_score_macro"],
                "f1_weighted": lr_metrics["f1_score"],
                "rationale": "L2-regularized linear decision boundary benchmark"
            },
            {
                "model": "K-Nearest Neighbors (k=7, distance)",
                "type": "Primary Selected Model",
                "cv_accuracy": knn_metrics["cv_accuracy_mean"],
                "accuracy": knn_metrics["accuracy"],
                "f1_macro": knn_metrics["f1_score_macro"],
                "f1_weighted": knn_metrics["f1_score"],
                "rationale": "Selected non-parametric model reflecting peer cohort proximity"
            }
        ]

        now_utc = datetime.now(timezone.utc).isoformat()
        provenance = {
            "evaluation_timestamp": now_utc,
            "training_timestamp_utc": now_utc,
            "dataset_hash_sha256": dataset_hash,
            "dataset_sha256": dataset_hash,
            "random_seed": 42,
            "python_version": sys.version.split()[0],
            "dependencies": {
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__
            },
            "environment_versions": {
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "python": sys.version.split()[0]
            },
            "evaluation_protocol": "Stratified 80/20 held-out split with 5-fold CV",
            "preprocessing_isolation": "Pipeline(StandardScaler(), Estimator()) per fold to prevent CV data leakage",
            "protocol_details": {
                "train_test_split": "80% train (680 samples), 20% held-out test (170 samples), stratified by career_track, random_state=42",
                "cross_validation": "5-Fold StratifiedKFold (random_state=42, shuffle=True) on training split only",
                "preprocessing_isolation": "Pipeline(StandardScaler(), Estimator()) per fold to prevent CV data leakage",
                "held_out_test_role": "Strictly held-out; used only for final benchmark comparison, not hyperparameter tuning"
            },
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "feature_columns": list(ML_FEATURE_COLUMNS),
            "data_scope": "Synthetic student profiles generated from hand-authored career rules",
            "external_validation": "Not performed"
        }

        self.metrics = {
            "knn": knn_metrics,
            "decision_tree": dt_metrics,
            "production_composite": composite_metrics,
            "model_comparison": model_comparison,
            "provenance": provenance,
            "dataset_info": {
                "total_samples": len(df),
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "features_count": len(ML_FEATURE_COLUMNS),
                "class_distribution": class_dist,
                "feature_defaults": {
                    column: round(float(df[column].median()), 4) for column in ML_FEATURE_COLUMNS
                }
            }
        }

        # Save artifacts atomically so an interrupted retrain never leaves a
        # partially written pickle that the next process attempts to load.
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for destination, value in (
            (self.knn_path, self.knn_model),
            (self.dt_path, self.dt_model),
            (self.scaler_path, self.scaler),
        ):
            with tempfile.NamedTemporaryFile("wb", dir=MODELS_DIR, delete=False) as tmp:
                pickle.dump(value, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
                temp_path = tmp.name
            os.replace(temp_path, destination)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=MODELS_DIR, delete=False) as tmp:
            json.dump(self.metrics, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_metrics_path = tmp.name
        os.replace(temp_metrics_path, self.metrics_path)

        self.is_trained = True
        return self.metrics

    def load_models(self):
        """Load artifacts only after their provenance matches this runtime."""
        required = (self.knn_path, self.dt_path, self.scaler_path, self.metrics_path)
        if all(path.exists() for path in required):
            try:
                import sklearn
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                provenance = metrics.get("provenance", {})
                saved_versions = provenance.get("dependencies", {})
                checks = {
                    "artifact schema": provenance.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION,
                    "feature schema": provenance.get("feature_columns") == list(ML_FEATURE_COLUMNS),
                    "scikit-learn": saved_versions.get("scikit-learn") == sklearn.__version__,
                    "numpy": saved_versions.get("numpy") == np.__version__,
                    "pandas": saved_versions.get("pandas") == pd.__version__,
                    "Python major/minor": ".".join(str(provenance.get("python_version", "")).split(".")[:2])
                    == ".".join(sys.version.split()[0].split(".")[:2]),
                }
                failed = [name for name, ok in checks.items() if not ok]
                if failed:
                    self.compatibility_status = "incompatible"
                    self.compatibility_reason = "Artifact mismatch: " + ", ".join(failed)
                    self.metrics = metrics
                    self.is_trained = False
                    return False
                with open(self.knn_path, "rb") as f:
                    self.knn_model = pickle.load(f)
                with open(self.dt_path, "rb") as f:
                    self.dt_model = pickle.load(f)
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.metrics = metrics
                self.is_trained = True
                self.compatibility_status = "compatible"
                self.compatibility_reason = "Saved artifacts match the active feature and dependency schema."
                return True
            except Exception as e:
                self.is_trained = False
                self.compatibility_status = "error"
                self.compatibility_reason = f"Artifacts could not be loaded: {type(e).__name__}"
                return False
        else:
            self.is_trained = False
            self.metrics = {}
            self.compatibility_status = "missing"
            self.compatibility_reason = "One or more model artifacts are missing."
            return False

    def predict_career_matches(self, academic_records: list, interests: dict, skills: dict, degree: str = None, target_career: str = None) -> dict:
        """
        Generates advisory pathway match scores for a student.

        Score Semantics & Disambiguation:
        - match_score / affinity_score (0-100%): An uncalibrated composite advisory affinity index
          combining K-NN competency (70%), stated interest (20%), and degree curriculum alignment (10%).
          The selected target is highlighted but never changes the ranking.
          IT IS NOT AN EMPIRICAL OR BAYESIAN PROBABILITY OF POST-GRADUATE EMPLOYMENT.
        - ml_competency_score / competency_prob (0-100%): Pure technical alignment predicted solely from verified
          academic module grades and technical skills (K-NN), completely independent of stated interests and aspirations.
        - knn_neighbor_score / knn_prob (0-100%): Inverse-distance weighted neighborhood class proportions from the K-NN model.
        - dt_leaf_score / dt_prob (0-100%): Training-sample class frequency distribution in the Decision Tree leaf node.
        - probability: Retained strictly for backward compatibility with legacy API callers and UI components; mirrors match_score.
        """
        if not self.is_trained:
            return {"ranked_matches": [], "model_used": "None", "is_trained": False}

        X_raw = self.extract_features(academic_records, interests, skills)
        X_scaled = self.scaler.transform(X_raw)

        knn_probs = self.knn_model.predict_proba(X_scaled)[0]
        knn_classes = list(self.knn_model.classes_)
        
        dt_probs = self.dt_model.predict_proba(X_raw)[0]
        dt_classes = list(self.dt_model.classes_)

        # Degree Alignment Prior
        prior = DEGREE_CAREER_ALIGNMENT.get(degree, {t: 0.20 for t in self.classes_})

        default_int = 1.0 if interests else 2.0
        interest_scores = {}
        for cat, trk in RUNTIME_INTEREST_TO_TRACK.items():
            interest_scores[trk] = float(interests.get(cat, default_int)) if interests else 2.0
        sum_int = sum(interest_scores.values()) or 1.0
        interest_dist = {trk: val / sum_int for trk, val in interest_scores.items()}

        nonzero_skills = sum(1 for lvl in (skills or {}).values() if SKILL_LEVELS.get(str(lvl), 0) > 0)
        evidence_count = len(academic_records) + nonzero_skills
        is_actionable = evidence_count >= 2

        competency_dist = {}
        diagnostics = {}
        for track in self.classes_:
            knn_p = knn_probs[knn_classes.index(track)] if track in knn_classes else 0.0
            dt_p = dt_probs[dt_classes.index(track)] if track in dt_classes else 0.0
            competency_dist[track] = float(knn_p)
            diagnostics[track] = float(dt_p)
        composite_dist = self._composite_distribution(competency_dist, interest_dist, degree)

        ranked = []
        for track in self.classes_:
            knn_p = competency_dist.get(track, 0.0)
            dt_p = diagnostics.get(track, 0.0)
            ml_p = knn_p
            int_p = interest_dist.get(track, 0.20)
            pri_p = prior.get(track, 0.20)
            raw_match_pct = round(float(composite_dist.get(track, 0.0)) * 100, 1)

            ranked.append({
                "track": track,
                "probability": raw_match_pct,
                "match_score": raw_match_pct,
                "affinity_score": raw_match_pct,
                "ml_competency_score": round(float(ml_p) * 100, 1),
                "competency_prob": round(float(ml_p) * 100, 1),
                "knn_neighbor_score": round(float(knn_p) * 100, 1),
                "knn_prob": round(float(knn_p) * 100, 1),
                "dt_leaf_score": round(float(dt_p) * 100, 1),
                "dt_prob": round(float(dt_p) * 100, 1),
                "interest_score": round(float(int_p) * 100, 1),
                "interest_prob": round(float(int_p) * 100, 1),
                "degree_prior_score": round(float(pri_p) * 100, 1),
                "prior_prob": round(float(pri_p) * 100, 1),
                "is_target": (track == target_career)
            })

        # Correct rounding drift so displayed shares remain a coherent distribution.
        total_p = sum(r["match_score"] for r in ranked)
        if total_p > 0:
            for r in ranked:
                norm_val = round((r["match_score"] / total_p) * 100, 1)
                r["probability"] = norm_val
                r["match_score"] = norm_val
                r["affinity_score"] = norm_val

        ranked.sort(key=lambda x: -x["match_score"])
        score_margin = round(
            ranked[0]["match_score"] - ranked[1]["match_score"], 1
        ) if len(ranked) > 1 else 100.0
        recommendation_available = is_actionable and score_margin >= 5.0

        # Determine pure ML top prediction from academic coursework and skills
        ml_sorted = sorted(ranked, key=lambda x: -x["ml_competency_score"])
        ml_predicted_track = ml_sorted[0]["track"] if ml_sorted else "Software Engineering"

        distances, indices = self.knn_model.kneighbors(X_scaled, n_neighbors=5)
        
        return {
            "ranked_matches": ranked,
            "top_track": ranked[0]["track"] if ranked and recommendation_available else None,
            "ml_predicted_track": ml_predicted_track,
            "model_used": "70% K-NN competency + 20% stated interests + 10% degree alignment",
            "nearest_neighbor_distances": [round(float(d), 3) for d in distances[0]],
            "is_actionable": is_actionable,
            "recommendation_available": recommendation_available,
            "score_margin": score_margin,
            "evidence_count": evidence_count,
            "evidence_status": (
                "sufficient_for_exploration" if recommendation_available
                else "ambiguous" if is_actionable
                else "insufficient"
            ),
            "evidence_message": (
                "This exploratory ranking has enough profile evidence and separation to discuss with an advisor."
                if recommendation_available else
                f"The leading pathways are separated by only {score_margin:.1f} points, so no single pathway recommendation is made. Review the full ranking with an advisor."
                if is_actionable else
                "Add at least two academic results or demonstrated skills before using the ranking. "
                "Interests and degree context alone are not enough for a recommendation."
            ),
            "limitations": "Synthetic-rule benchmark only; not a prediction of employment or career success."
        }
