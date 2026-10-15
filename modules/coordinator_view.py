"""Privacy-conscious cohort analytics for programme coordinators."""
import pandas as pd
import plotly.express as px
import streamlit as st

from config import CAREER_TRACKS
from modules.ui import empty_state, page_header, section_header


MIN_REPORTING_COHORT = 5


def build_cohort_insights(students, skills_summary, minimum_cohort=MIN_REPORTING_COHORT):
    """Return factual, data-derived observations or suppress small cohorts."""
    if len(students) < minimum_cohort:
        return {
            "suppressed": True,
            "messages": [f"Insights are withheld until at least {minimum_cohort} students are represented."],
        }
    frame = pd.DataFrame(students)
    messages = []
    track_share = frame["target_career"].value_counts(normalize=True)
    if not track_share.empty:
        top_track = track_share.index[0]
        messages.append(f"{top_track} is the most selected pathway at {track_share.iloc[0] * 100:.1f}% of the cohort.")
    if "weekly_hours" in frame:
        messages.append(f"Reported weekly study capacity averages {frame['weekly_hours'].mean():.1f} hours.")
    weak = [row for row in skills_summary if row.get("proficiency_level") in {"None", "Beginner"}]
    if weak:
        counts = {}
        for row in weak:
            counts[row["skill_name"]] = counts.get(row["skill_name"], 0) + row["count"]
        skill, count = max(counts.items(), key=lambda item: item[1])
        messages.append(f"{skill} has the largest recorded beginner-or-lower count ({count} records).")
    return {"suppressed": False, "messages": messages}


def render_coordinator_view(db, rule_engine):
    page_header(
        "Programme workspace",
        "Read the cohort without exposing students",
        "Monitor aggregate pathway demand, recorded readiness, and training needs. Small-group insights are suppressed to reduce re-identification risk.",
    )
    students = db.get_all_students_summary("coordinator")
    if not students:
        empty_state("No cohort data", "Aggregate charts will appear after student profiles are created.")
        return

    frame = pd.DataFrame(students)
    if len(frame) < MIN_REPORTING_COHORT:
        metrics = st.columns(2)
        metrics[0].metric("Students represented", len(frame))
        metrics[1].metric("Minimum reporting cohort", MIN_REPORTING_COHORT)
        section_header("Cohort reporting withheld", "No performance, pathway, or skill aggregate is shown for a potentially identifiable small group.")
        empty_state(
            "More records are required",
            f"Aggregate reporting begins when at least {MIN_REPORTING_COHORT} students are represented.",
        )
        insights = build_cohort_insights(students, [])
        for message in insights["messages"]:
            st.info(message)
        return

    metrics = st.columns(4)
    metrics[0].metric("Students represented", len(frame))
    metrics[1].metric("Mean recorded GPA", f"{frame['gpa'].mean():.2f}")
    metrics[2].metric("Mean readiness index", f"{frame['last_readiness_score'].mean():.1f}/100")
    metrics[3].metric("Pathways configured", len(CAREER_TRACKS))
    st.caption("The readiness index is a profile-completeness and benchmark-planning aid, not an employability measure.")

    section_header("Cohort distribution", "Only aggregate data is available in this workspace.")
    left, right = st.columns(2)
    with left:
        counts = frame["target_career"].value_counts().rename_axis("Pathway").reset_index(name="Students")
        fig = px.bar(counts, x="Students", y="Pathway", orientation="h", color_discrete_sequence=["#131209"])
        fig.update_layout(height=340, margin=dict(l=16, r=16, t=16, b=16), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        by_degree = frame.groupby("degree", as_index=False)["last_readiness_score"].mean()
        by_degree.columns = ["Degree", "Mean readiness index"]
        fig = px.bar(by_degree, x="Mean readiness index", y="Degree", orientation="h", color_discrete_sequence=["#666666"])
        fig.update_layout(height=340, margin=dict(l=16, r=16, t=16, b=16), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    section_header("Recorded skill levels", "Counts reflect submitted profiles, not verified institutional assessments.")
    skills_summary = db.get_all_student_skills_summary("coordinator")
    if skills_summary:
        skills = pd.DataFrame(skills_summary).pivot_table(
            index="skill_name", columns="proficiency_level", values="count", fill_value=0
        ).reset_index()
        st.dataframe(skills, hide_index=True, width="stretch")
    else:
        empty_state("No skill data", "Recorded skill distributions will appear after students complete self-assessments.")

    section_header("Data-derived observations", "These statements are generated from the current cohort rather than fixed demo copy.")
    insights = build_cohort_insights(students, skills_summary)
    for message in insights["messages"]:
        st.info(message)
