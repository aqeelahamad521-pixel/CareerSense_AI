"""Administrator workspace for model, rules, and learning-graph diagnostics."""
import pandas as pd
import streamlit as st

from modules.ui import empty_state, page_header, section_header
from scripts.train_models import main as run_train_models


def _metric_table(metric):
    rows = []
    for track, values in metric.get("per_class", {}).items():
        rows.append({
            "Track": track,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1_score"],
            "Support": values["support"],
        })
    return pd.DataFrame(rows)


def render_admin_view(db, rule_engine, a_star, ml_classifier):
    if st.session_state.get("role") != "admin":
        st.error("Administrator access is required.")
        return

    page_header(
        "Administrator workspace",
        "Inspect what the system knows—and what it does not",
        "Review synthetic benchmarks, artifact compatibility, expert rules, and learning-graph coverage before exposing changes to users.",
    )
    tabs = st.tabs(["Model diagnostics", "Expert rules", "Learning graph", "Maintenance"])

    with tabs[0]:
        section_header("Model diagnostics", "All reported classification metrics use synthetic labels generated from hand-authored assumptions.")
        status = ml_classifier.compatibility_status.replace("_", " ").title()
        st.info(f"Artifact status: {status}. {ml_classifier.compatibility_reason}")
        metrics = ml_classifier.metrics
        if not metrics:
            empty_state("No model metrics", "Retrain the models from the maintenance tab.")
        else:
            info = metrics.get("dataset_info", {})
            provenance = metrics.get("provenance", {})
            summary = st.columns(4)
            summary[0].metric("Synthetic profiles", info.get("total_samples", 0))
            summary[1].metric("Held-out profiles", info.get("testing_samples", 0))
            summary[2].metric("Competency features", info.get("features_count", 0))
            summary[3].metric("Artifact schema", provenance.get("artifact_schema_version", "Unknown"))
            st.warning(
                "These results measure recovery of the synthetic generator’s labels. They do not establish fairness, "
                "career suitability, employability, or real-world outcome accuracy."
            )
            composite = metrics.get("production_composite", {})
            st.markdown("#### Exact production composite")
            composite_cols = st.columns(3)
            composite_cols[0].metric("Held-out accuracy", f"{composite.get('accuracy', 0)}%")
            composite_cols[1].metric("Macro F1", f"{composite.get('f1_score_macro', 0)}%")
            composite_cols[2].metric("Weighted F1", f"{composite.get('f1_score', 0)}%")
            st.caption(composite.get("formula", "70% K-NN competency + 20% interests + 10% degree alignment"))
            st.caption(composite.get("limitations", "Synthetic holdout only."))

            subgroup = composite.get("synthetic_subgroup_diagnostic", {})
            groups = subgroup.get("groups", {})
            if groups:
                st.markdown("#### Synthetic subgroup diagnostic")
                subgroup_rows = [
                    {"Degree programme": name, **values}
                    for name, values in groups.items()
                ]
                st.dataframe(pd.DataFrame(subgroup_rows), hide_index=True, width="stretch")
                st.caption(subgroup.get("limitations", "This is not a real-world fairness audit."))

            comparison = metrics.get("model_comparison", [])
            if comparison:
                st.markdown("#### Candidate model comparison")
                st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
            diagnostic_tabs = st.tabs(["K-NN primary", "Decision tree baseline"])
            for tab, key in zip(diagnostic_tabs, ["knn", "decision_tree"]):
                with tab:
                    model_metric = metrics.get(key, {})
                    cols = st.columns(4)
                    cols[0].metric("Accuracy", f"{model_metric.get('accuracy', 0)}%")
                    cols[1].metric("Macro F1", f"{model_metric.get('f1_score_macro', 0)}%")
                    cols[2].metric("Weighted F1", f"{model_metric.get('f1_score', 0)}%")
                    cols[3].metric("CV mean", f"{model_metric.get('cv_accuracy_mean', 0)}%")
                    table = _metric_table(model_metric)
                    if not table.empty:
                        st.dataframe(table, hide_index=True, width="stretch")
            with st.expander("Provenance"):
                st.json(provenance)

    with tabs[1]:
        section_header("Expert rules", "These thresholds are policy assumptions and require academic governance, not only technical review.")
        academic = rule_engine.rules.get("academic_prerequisite_rules", [])
        dependency = rule_engine.rules.get("competency_dependency_rules", [])
        readiness = rule_engine.rules.get("portfolio_and_readiness_rules", [])
        st.markdown("#### Academic prerequisites")
        st.dataframe(pd.DataFrame(academic), hide_index=True, width="stretch")
        st.markdown("#### Competency dependencies")
        st.dataframe(pd.DataFrame(dependency), hide_index=True, width="stretch")
        st.markdown("#### Portfolio rules")
        st.dataframe(pd.DataFrame(readiness), hide_index=True, width="stretch")

    with tabs[2]:
        section_header("Learning graph", "Activities are inputs to a minimum-declared-hours search; completion is not proof of proficiency.")
        activities = pd.DataFrame(a_star.activities)
        columns = ["id", "title", "track", "type", "estimated_hours", "main_skill", "skill_level_gain", "prerequisites"]
        st.dataframe(activities[columns], hide_index=True, width="stretch")
        coverage = []
        for track in a_star.careers:
            result = a_star.generate_optimal_roadmap(track, {}, 8)
            coverage.append({
                "Pathway": track,
                "Status": result["status"],
                "Activities": len(result["roadmap"]),
                "Uncovered gaps": ", ".join(result.get("uncovered_gaps", [])) or "None",
            })
        st.markdown("#### Zero-evidence coverage check")
        st.dataframe(pd.DataFrame(coverage), hide_index=True, width="stretch")

    with tabs[3]:
        section_header("Maintenance", "Regeneration and retraining replace the synthetic dataset and saved artifacts.")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Regenerate synthetic dataset")
            st.write("Rebuild the 850 demonstration profiles using the current generator assumptions.")
            confirm_data = st.checkbox("I understand this replaces the generated CSV", key="confirm_data")
            if st.button("Regenerate dataset", disabled=not confirm_data, width="stretch"):
                try:
                    from scripts.generate_dataset import generate_student_dataset
                    generate_student_dataset()
                    st.success("Synthetic dataset regenerated.")
                except Exception:
                    st.error("Dataset generation failed. Check the application logs.")
        with right:
            st.markdown("#### Retrain model artifacts")
            st.write("Fit the configured models and write new versioned provenance metadata.")
            confirm_model = st.checkbox("I understand this replaces saved model artifacts", key="confirm_model")
            if st.button("Retrain models", type="primary", disabled=not confirm_model, width="stretch"):
                try:
                    run_train_models()
                    if ml_classifier.load_models():
                        st.success("Models retrained and reloaded.")
                        st.rerun()
                    else:
                        st.error(ml_classifier.compatibility_reason)
                except Exception:
                    st.error("Training failed. Check the application logs.")
