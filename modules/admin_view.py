"""
System Administrator & AI Model Explorer View for CareerSense AI.
Provides capabilities for:
1. Monitoring AI classification model benchmarks (Accuracy, Precision, Recall, F1, Confusion Matrix).
2. Triggering on-demand dataset re-synthesis and model re-training.
3. Inspecting the expert system IF-THEN rules knowledge base.
4. Exploring the A* learning activity DAG and graph dependencies.
"""
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
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
        st.caption("Evaluation on 850 undergraduate student profiles (80% Train, 20% Test Split).")

        metrics = ml_classifier.metrics
        if not metrics:
            st.warning("Model metrics not loaded. Please train models in the Maintenance tab.")
        else:
            knn = metrics.get("knn", {})
            dt = metrics.get("decision_tree", {})
            info = metrics.get("dataset_info", {})

            col_ds1, col_ds2, col_ds3, col_ds4 = st.columns(4)
            col_ds1.metric("Total Profiles", info.get("total_samples", 850))
            col_ds2.metric("Train Samples", info.get("training_samples", 680))
            col_ds3.metric("Test Samples", info.get("testing_samples", 170))
            col_ds4.metric("Features Extracted", info.get("features_count", 36))

            st.divider()

            # Side-by-side model comparison
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"#### 🔵 Primary: {knn.get('name', 'K-NN')}")
                cm1, cm2 = st.columns(2)
                cm1.metric("Accuracy", f"{knn.get('accuracy', 0)}%")
                cm2.metric("F1-Score (Weighted)", f"{knn.get('f1_score', 0)}%")
                cm3, cm4 = st.columns(2)
                cm3.metric("Precision", f"{knn.get('precision', 0)}%")
                cm4.metric("Recall", f"{knn.get('recall', 0)}%")

                # Confusion Matrix Heatmap
                st.markdown("##### Confusion Matrix (K-NN)")
                labels = knn.get("labels", [])
                cm_data = knn.get("confusion_matrix", [])
                if cm_data and labels:
                    fig_cm = ff.create_annotated_heatmap(
                        z=cm_data,
                        x=labels,
                        y=labels,
                        colorscale="Blues"
                    )
                    fig_cm.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
                    st.plotly_chart(fig_cm, use_container_width=True)

            with col_m2:
                st.markdown(f"#### 🟢 Baseline: {dt.get('name', 'Decision Tree')}")
                dm1, dm2 = st.columns(2)
                dm1.metric("Accuracy", f"{dt.get('accuracy', 0)}%")
                dm2.metric("F1-Score (Weighted)", f"{dt.get('f1_score', 0)}%")
                dm3, dm4 = st.columns(2)
                dm3.metric("Precision", f"{dt.get('precision', 0)}%")
                dm4.metric("Recall", f"{dt.get('recall', 0)}%")

                # Feature importances
                st.markdown("##### Top Influential Features (Decision Tree)")
                top_feats = dt.get("top_features", [])
                if top_feats:
                    df_feats = pd.DataFrame(top_feats, columns=["Feature", "Gini Importance"])
                    st.dataframe(df_feats, use_container_width=True, hide_index=True)

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
                from scripts.generate_dataset import generate_student_dataset
                generate_student_dataset()
                st.success("Fresh dataset generated in data/students_dataset.csv!")
        with col_r2:
            st.info("Train K-NN and Decision Tree models, calculate cross-validation benchmarks, and persist model files.")
            if st.button("Retrain All AI Models", use_container_width=True):
                run_train_models()
                ml_classifier.load_models()
                st.success("All AI models re-trained and metrics reloaded successfully!")
                st.rerun()
