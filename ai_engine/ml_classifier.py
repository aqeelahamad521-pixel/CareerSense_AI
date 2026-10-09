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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import (
    DATASET_PATH, MODELS_DIR, CAREER_TRACKS,
    ML_FEATURE_COLUMNS, GRADE_POINTS, SKILL_LEVELS,
    DEGREE_CAREER_ALIGNMENT
)

def compute_multiclass_metrics(y_true, y_pred, labels):
    """
    Computes mathematically rigorous multiclass classification metrics.
    Guarantees that Accuracy, Precision, Recall, and F1 are independently calculated.
    Correctly handles edge cases, zero-division, and missing classes.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return {}
    
    # 1. Accuracy: correct predictions divided by total evaluated samples
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct / n
    
    # 2. Confusion matrix: rows = actual labels, columns = predicted labels
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cm = [[0 for _ in range(k)] for _ in range(k)]
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt]][label_to_idx[yp]] += 1
            
    # 3. Per-class metrics
    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0
    
    for i, label in enumerate(labels):
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
        "labels": labels
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
        self.load_models()

    def extract_features(self, academic_records: list, interests: dict, skills: dict) -> np.ndarray:
        """Transforms raw student attributes into a feature vector matching ML_FEATURE_COLUMNS."""
        grade_map = {
            "grade_programming": 2.5,
            "grade_math": 2.5,
            "grade_database": 2.5,
            "grade_networking": 2.5,
            "grade_systems": 2.5,
            "grade_design": 2.5
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

        default_int = 1.0 if interests else 2.0
        interest_map = {
            "interest_software": float(interests.get("Software Development & Systems", default_int)),
            "interest_data": float(interests.get("Data Analysis & AI Research", default_int)),
            "interest_security": float(interests.get("Cybersecurity & Threat Defense", default_int)),
            "interest_cloud": float(interests.get("Cloud Infrastructure & Automation", default_int)),
            "interest_design": float(interests.get("UI/UX Design & User Experience", default_int))
        }

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
            elif col in interest_map:
                vec.append(float(interest_map[col]))
            elif col in skill_attr_map:
                lvl_val = SKILL_LEVELS.get(skill_attr_map[col], 0)
                vec.append(float(lvl_val))
            else:
                vec.append(0.0)

        return np.array(vec, dtype=np.float64).reshape(1, -1)

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
            ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))
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

        self.knn_model = KNeighborsClassifier(n_neighbors=7, weights="distance")
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

        provenance = {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_hash_sha256": dataset_hash,
            "random_seed": 42,
            "python_version": sys.version.split()[0],
            "dependencies": {
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__
            },
            "evaluation_protocol": {
                "train_test_split": "80% train (680 samples), 20% held-out test (170 samples), stratified by career_track, random_state=42",
                "cross_validation": "5-Fold StratifiedKFold (random_state=42, shuffle=True) on training split only",
                "preprocessing_isolation": "Pipeline(StandardScaler(), Estimator()) per fold to prevent CV data leakage",
                "held_out_test_role": "Strictly held-out; used only for final benchmark comparison, not hyperparameter tuning"
            }
        }

        self.metrics = {
            "knn": knn_metrics,
            "decision_tree": dt_metrics,
            "model_comparison": model_comparison,
            "provenance": provenance,
            "dataset_info": {
                "total_samples": len(df),
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "features_count": len(ML_FEATURE_COLUMNS),
                "class_distribution": class_dist
            }
        }

        # Save artifacts atomically
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.knn_path, "wb") as f:
            pickle.dump(self.knn_model, f)
        with open(self.dt_path, "wb") as f:
            pickle.dump(self.dt_model, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        self.is_trained = True
        return self.metrics

    def load_models(self):
        """Loads trained models and saved metrics. Returns True if successful, False otherwise."""
        if self.knn_path.exists() and self.dt_path.exists() and self.scaler_path.exists():
            try:
                with open(self.knn_path, "rb") as f:
                    self.knn_model = pickle.load(f)
                with open(self.dt_path, "rb") as f:
                    self.dt_model = pickle.load(f)
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                if self.metrics_path.exists():
                    with open(self.metrics_path, "r", encoding="utf-8") as f:
                        self.metrics = json.load(f)
                self.is_trained = True
                return True
            except Exception as e:
                self.is_trained = False
                self.metrics = {}
                return False
        else:
            self.is_trained = False
            self.metrics = {}
            return False

    def predict_career_matches(self, academic_records: list, interests: dict, skills: dict, degree: str = None, target_career: str = None) -> dict:
        """
        Generates advisory pathway match scores for a student.
        Note: The returned match score is an uncalibrated composite advisory affinity index (0-100%),
        not an empirical probability of employment.
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

        # Map domain interests to career tracks
        category_to_track = {
            "Software Development & Systems": "Software Engineering",
            "Data Analysis & AI Research": "Data Science / AI",
            "Cybersecurity & Threat Defense": "Cybersecurity",
            "Cloud Infrastructure & Automation": "Cloud / DevOps",
            "UI/UX Design & User Experience": "UI/UX Design"
        }
        default_int = 1.0 if interests else 2.0
        interest_scores = {}
        for cat, trk in category_to_track.items():
            interest_scores[trk] = float(interests.get(cat, default_int)) if interests else 2.0
        sum_int = sum(interest_scores.values()) or 1.0
        interest_dist = {trk: val / sum_int for trk, val in interest_scores.items()}

        # Assess profile completeness and balance weights
        has_academic_data = len(academic_records) > 0
        has_skill_data = any(lvl != "None" and lvl != 0 for lvl in skills.values()) if skills else False

        if target_career and target_career in self.classes_:
            if has_academic_data:
                # 40% Current Academic/Skill Competency, 30% Target Aspiration, 15% Domain Interest, 15% Degree Prior
                w_ml, w_target, w_interest, w_prior = 0.40, 0.30, 0.15, 0.15
            elif has_skill_data:
                w_ml, w_target, w_interest, w_prior = 0.35, 0.35, 0.15, 0.15
            else:
                # Cold start: high weight on target aspiration and domain interest
                w_ml, w_target, w_interest, w_prior = 0.15, 0.45, 0.20, 0.20
            target_dist = {t: (1.0 if t == target_career else 0.0) for t in self.classes_}
        else:
            w_target = 0.0
            target_dist = {t: 0.0 for t in self.classes_}
            if has_academic_data:
                w_ml, w_interest, w_prior = 0.65, 0.15, 0.20
            elif has_skill_data:
                w_ml, w_interest, w_prior = 0.55, 0.20, 0.25
            else:
                w_ml, w_interest, w_prior = 0.20, 0.30, 0.50

        ranked = []
        for track in self.classes_:
            knn_p = knn_probs[knn_classes.index(track)] if track in knn_classes else 0.0
            dt_p = dt_probs[dt_classes.index(track)] if track in dt_classes else 0.0
            ml_p = (0.70 * knn_p) + (0.30 * dt_p)
            tgt_p = target_dist.get(track, 0.0)
            int_p = interest_dist.get(track, 0.20)
            pri_p = prior.get(track, 0.20)

            combined_p = (w_ml * ml_p) + (w_target * tgt_p) + (w_interest * int_p) + (w_prior * pri_p)
            
            ranked.append({
                "track": track,
                "probability": round(float(combined_p) * 100, 1),
                "competency_prob": round(float(ml_p) * 100, 1),
                "knn_prob": round(float(knn_p) * 100, 1),
                "dt_prob": round(float(dt_p) * 100, 1),
                "interest_prob": round(float(int_p) * 100, 1),
                "prior_prob": round(float(pri_p) * 100, 1),
                "is_target": (track == target_career)
            })

        # Normalize so affinity percentages sum to 100%
        total_p = sum(r["probability"] for r in ranked)
        if total_p > 0:
            for r in ranked:
                r["probability"] = round((r["probability"] / total_p) * 100, 1)

        ranked.sort(key=lambda x: -x["probability"])

        distances, indices = self.knn_model.kneighbors(X_scaled, n_neighbors=5)
        
        weights_desc = f"ML Competency ({int(w_ml*100)}%)"
        if w_target > 0:
            weights_desc += f" + Target Aspiration ({int(w_target*100)}%)"
        weights_desc += f" + Interests ({int(w_interest*100)}%) + Degree Prior ({int(w_prior*100)}%)"

        return {
            "ranked_matches": ranked,
            "top_track": ranked[0]["track"] if ranked else "Software Engineering",
            "model_used": weights_desc,
            "nearest_neighbor_distances": [round(float(d), 3) for d in distances[0]]
        }
