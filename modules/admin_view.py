"""
System Administrator & AI Model Explorer View for CareerSense AI.
Provides capabilities for:
1. Monitoring AI classification model benchmarks (Accuracy, Precision, Recall, F1, Confusion Matrices, Cross-Validation).
2. Inspecting multi-model comparison benchmarks (Dummy Baseline, Decision Tree, Random Forest, Logistic Regression, K-NN).
3. Triggering on-demand dataset re-synthesis and model re-training.
4. Inspecting the expert system IF-THEN rules knowledge base.
5. Exploring the A* learning activity DAG and graph dependencies.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from config import DATASET_PATH
from scripts.train_models import main as run_train_models

def render_admin_view(db, rule_engine, a_star, ml_classifier):
    st.title("⚙️ System Administrator & AI Model Explorer")
    st.caption("Inspect and manage core AI layers, model performance benchmarks, and rule engines.")

    admin_tabs = st.tabs([
        "🧠 AI Model Benchmarks (ML Layer 1)",
        "📜 Rule Knowledge Base (AI Layer 2)",
        "🕸️ A* Activity Graph (AI Layer 3)",
        "🔄 Pipeline Maintenance"
    ])

    # -------------------------------------------------------------
    # TAB 1: AI Model Benchmarks
    # -------------------------------------------------------------
    with admin_tabs[0]:
        st.markdown("### 🏆 Machine Learning Classification Benchmarks")
        st.caption("Rigorous evaluation on 850 undergraduate student profiles (80% Train, 20% Held-Out Test Split with 5-Fold Cross-Validation).")

        metrics = ml_classifier.metrics
        if not metrics:
            st.warning("Model metrics not loaded. Please train models in the Maintenance tab.")
        else:
            knn = metrics.get("knn", {})
            dt = metrics.get("decision_tree", {})
            info = metrics.get("dataset_info", {})
            comp = metrics.get("model_comparison", [])

            col_ds1, col_ds2, col_ds3, col_ds4 = st.columns(4)
            col_ds1.metric("Total Cohort Profiles", info.get("total_samples", 850))
            col_ds2.metric("Training Samples (80%)", info.get("training_samples", 680))
            col_ds3.metric("Held-Out Test Samples (20%)", info.get("testing_samples", 170))
            col_ds4.metric("Engineered Features", info.get("features_count", 36))

            st.divider()

            # Multi-Model Benchmark Comparison Table
            st.markdown("#### 📊 Candidate Model Comparison & Baseline Justification")
            st.caption("Comparative assessment across five distinct machine learning paradigms on identical held-out test data:")
            
            if comp:
                df_comp = pd.DataFrame(comp)
                col_rename = {
                    "model": "Model Architecture",
                    "type": "Paradigm / Role",
                    "cv_accuracy": "5-Fold CV (%)",
                    "accuracy": "Held-Out Test Acc (%)",
                    "f1_macro": "Macro F1 (%)",
                    "f1_weighted": "Weighted F1 (%)",
                    "rationale": "Evaluation Rationale"
                }
                display_df = df_comp.rename(columns=col_rename)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.divider()

            # Provenance & Audit Verification Card
            prov = metrics.get("provenance", {})
            if prov:
                with st.expander("🛡️ Model Provenance & Audit Metadata", expanded=False):
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.markdown(f"• **Evaluation Timestamp (UTC)**: `{prov.get('training_timestamp_utc', 'N/A')}`")
                        st.markdown(f"• **Dataset SHA-256**: `{prov.get('dataset_sha256', 'N/A')[:24]}...`")
                        st.markdown(f"• **Random Seed**: `{prov.get('random_seed', 'N/A')}` (Deterministic train/test split)")
                    with col_p2:
                        st.markdown(f"• **Evaluation Protocol**: `{prov.get('evaluation_protocol', 'Stratified 80/20 held-out split with 5-fold CV')}`")
                        st.markdown(f"• **Preprocessing Isolation**: `{prov.get('preprocessing_isolation', 'StandardScaler fitted strictly inside training folds')}`")
                        env = prov.get('environment_versions', {})
                        st.markdown(f"• **Runtime Environment**: scikit-learn `{env.get('scikit_learn', 'N/A')}`, numpy `{env.get('numpy', 'N/A')}`, python `{env.get('python', 'N/A')}`")

            # Detailed Side-by-Side Model Diagnostics
            col_m1, col_m2 = st.columns(2)

            # ----------------- K-NN (PRIMARY MODEL) -----------------
            with col_m1:
                st.markdown(f"#### 🔵 Primary: {knn.get('name', 'K-NN (Primary)')}")
                st.caption(f"5-Fold CV Accuracy: **{knn.get('cv_accuracy_mean', 'N/A')}% (±{knn.get('cv_accuracy_std', 'N/A')}%)**")
                
                cm1, cm2 = st.columns(2)
                cm1.metric("Held-Out Accuracy", f"{knn.get('accuracy', 0)}%")
                cm2.metric("Weighted F1-Score", f"{knn.get('f1_score', 0)}%")
                
                cm3, cm4 = st.columns(2)
                cm3.metric("Macro F1-Score", f"{knn.get('f1_score_macro', 'N/A')}%")
                cm4.metric("Weighted Precision", f"{knn.get('precision', 0)}%")

                # Confusion Matrix Heatmap (K-NN)
                st.markdown("##### Confusion Matrix (K-NN)")
                labels = knn.get("labels", [])
                cm_knn = knn.get("confusion_matrix", [])
                if cm_knn and labels:
                    fig_knn = px.imshow(
                        cm_knn,
                        x=labels,
                        y=labels,
                        color_continuous_scale="Blues",
                        text_auto=True,
                        labels=dict(x="Predicted Career Track (Columns)", y="Actual Career Track (Rows)", color="Students")
                    )
                    fig_knn.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340, coloraxis_showscale=False)
                    st.plotly_chart(fig_knn, use_container_width=True)

                # Per-Class Metrics Table (K-NN)
                st.markdown("##### Per-Class Performance Breakdown (K-NN)")
                knn_per_class = knn.get("per_class", {})
                if knn_per_class:
                    rows_pc = []
                    for trk, data in knn_per_class.items():
                        rows_pc.append({
                            "Career Track": trk,
                            "Precision (%)": data["precision"],
                            "Recall (%)": data["recall"],
                            "F1-Score (%)": data["f1_score"],
                            "Test Support": data["support"]
                        })
                    st.dataframe(pd.DataFrame(rows_pc), use_container_width=True, hide_index=True)

            # ------------- DECISION TREE (BASELINE MODEL) -------------
            with col_m2:
                st.markdown(f"#### 🟢 Baseline: {dt.get('name', 'Decision Tree')}")
                st.caption(f"5-Fold CV Accuracy: **{dt.get('cv_accuracy_mean', 'N/A')}% (±{dt.get('cv_accuracy_std', 'N/A')}%)**")
                
                dm1, dm2 = st.columns(2)
                dm1.metric("Held-Out Accuracy", f"{dt.get('accuracy', 0)}%")
                dm2.metric("Weighted F1-Score", f"{dt.get('f1_score', 0)}%")
                
                dm3, dm4 = st.columns(2)
                dm3.metric("Macro F1-Score", f"{dt.get('f1_score_macro', 'N/A')}%")
                dm4.metric("Weighted Precision", f"{dt.get('precision', 0)}%")

                # Confusion Matrix Heatmap (Decision Tree)
                st.markdown("##### Confusion Matrix (Decision Tree)")
                cm_dt = dt.get("confusion_matrix", [])
                if cm_dt and labels:
                    fig_dt = px.imshow(
                        cm_dt,
                        x=labels,
                        y=labels,
                        color_continuous_scale="Greens",
                        text_auto=True,
                        labels=dict(x="Predicted Career Track (Columns)", y="Actual Career Track (Rows)", color="Students")
                    )
                    fig_dt.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340, coloraxis_showscale=False)
                    st.plotly_chart(fig_dt, use_container_width=True)

                # Feature importances
                st.markdown("##### Top Discriminative Features (Gini Importance)")
                top_feats = dt.get("top_features", [])
                if top_feats:
                    df_feats = pd.DataFrame(top_feats, columns=["Feature", "Gini Importance"])
                    st.dataframe(df_feats, use_container_width=True, hide_index=True)

            with st.expander("ℹ️ Methodological Notes: Metric Independence & Target Leakage Prevention"):
                st.markdown(r"""
                - **Synthetic Demonstration Dataset & Limitations**: The dataset is an educational synthetic demonstration representing 850 undergraduate computing students. While engineered with domain realism and latent student archetypes, it is not sampled from empirical longitudinal graduate employment outcomes.
                - **Target Leakage Remediation**: The student dataset generator uses continuous latent aptitudes with overlapping multi-domain interests and realistic grade distributions. No single feature acts as a trivial deterministic shortcut.
                - **Independence of Precision, Recall, and F1**: Accuracy evaluates overall correct classifications ($\sum TP / N$). In multiclass problems with non-diagonal confusion matrices, per-class False Positives ($FP$) and False Negatives ($FN$) diverge, producing distinct Precision ($TP / (TP + FP)$) and Recall ($TP / (TP + FN)$) metrics.
                - **Macro vs. Weighted Metrics**: Macro averages calculate the unweighted arithmetic mean across all 5 career tracks, treating minority tracks equally. Weighted averages weight each class metric by its test support ($N_k / N$).
                - **Model Selection Rationale**: K-NN ($k=7$, distance-weighted) is selected as the primary recommendation engine because student advisory naturally maps to instance-based peer similarity ('students with similar academic and skill profiles to you succeeded in this track'). Decision Tree serves as the white-box interpretable baseline.
                """)

    # -------------------------------------------------------------
    # TAB 2: Rule Knowledge Base
    # -------------------------------------------------------------
    with admin_tabs[1]:
        st.markdown("### 📜 Expert System IF-THEN Rules Knowledge Base")
        st.caption("Declarative prerequisite rules and dependency constraints ensuring explainable advisory reasoning.")

        st.markdown("#### 1. Academic Prerequisite Validation Rules")
        acad_rules = rule_engine.rules.get("academic_prerequisite_rules", [])
        st.dataframe(pd.DataFrame(acad_rules)[["id", "target_track", "subject", "minimum_grade", "severity", "message"]], use_container_width=True, hide_index=True)

        st.markdown("#### 2. Competency Dependency Rules")
        dep_rules = rule_engine.rules.get("competency_dependency_rules", [])
        st.dataframe(pd.DataFrame(dep_rules)[["id", "skill", "prerequisite_skill", "min_prereq_level", "priority", "message"]], use_container_width=True, hide_index=True)

        st.markdown("#### 3. Portfolio & Readiness Rules")
        readiness_rules = rule_engine.rules.get("portfolio_and_readiness_rules", [])
        st.dataframe(pd.DataFrame(readiness_rules), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 3: A* Activity Graph
    # -------------------------------------------------------------
    with admin_tabs[2]:
        st.markdown("### 🕸️ A* Search Activity Graph & Curated Database")
        st.caption("The complete directed graph of courses, projects, and certifications explored by the A* optimizer.")

        acts = a_star.activities
        if acts:
            df_acts = pd.DataFrame(acts)[["id", "title", "track", "type", "estimated_hours", "main_skill", "skill_level_gain", "prerequisites"]]
            st.dataframe(df_acts, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 4: Pipeline Maintenance
    # -------------------------------------------------------------
    with admin_tabs[3]:
        st.markdown("### 🔄 AI Pipeline Re-training & Data Maintenance")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.info("Regenerate synthetic student cohort dataset (850 realistic student profiles) with updated feature schemas.")
            if st.button("Generate Fresh Dataset", use_container_width=True):
                try:
                    from scripts.generate_dataset import generate_student_dataset
                    generate_student_dataset()
                    st.success("Fresh dataset generated in data/students_dataset.csv!")
                except Exception as e:
                    st.error(f"Dataset generation failed: {e}")
        with col_r2:
            st.info("Train K-NN and Decision Tree models, calculate cross-validation benchmarks, and persist model files.")
            if st.button("Retrain All AI Models", use_container_width=True):
                try:
                    run_train_models()
                    loaded = ml_classifier.load_models()
                    if loaded:
                        st.success("All AI models re-trained and metrics reloaded successfully!")
                        st.rerun()
                    else:
                        st.error("Model artifacts could not be loaded after training. Please inspect the logs.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
