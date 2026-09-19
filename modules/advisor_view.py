"""
Academic Advisor Portal View for CareerSense AI.
Allows faculty mentors to:
1. Search and inspect assigned undergraduate advisee profiles.
2. Review AI career classification and explainable evidence.
3. Audit rule-based prerequisite validations and competency deficiencies.
4. Track student milestone progress across their A* roadmap.
5. Record actionable mentorship feedback and guidance notes.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_advisor_view(db, rule_engine, a_star, ml_classifier, explainability):
    advisor_name = st.session_state.full_name or "Academic Advisor"
    st.title("👨‍🏫 Academic Advisor Decision Support Portal")
    st.caption(f"Logged in as: **{advisor_name}** | Faculty of Computing Advising Desk")

    students = db.get_all_students_summary()
    if not students:
        st.warning("No student profiles found in system.")
        return

    # Student selector
    student_options = {f"{s['full_name']} ({s['reg_no']}) - {s['degree']}": s["id"] for s in students}
    selected_label = st.selectbox("Select Student Advisee to Review", list(student_options.keys()))
    selected_student_id = student_options[selected_label]

    # Fetch selected student data
    student_user = db.get_user_by_id(selected_student_id)
    profile = db.get_student_profile(selected_student_id) or {}
    academic_records = db.get_academic_records(selected_student_id)
    student_skills = db.get_student_skills(selected_student_id)
    student_interests = db.get_student_interests(selected_student_id)
    projects = db.get_student_projects(selected_student_id)
    documents = db.get_student_documents(selected_student_id)

    target_track = profile.get("target_career", "Data Science / AI")
    weekly_hours = int(profile.get("weekly_hours", 8))


    # Top summary metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Cumulative GPA", f"{profile.get('gpa', 0.0):.2f}")
    with col_m2:
        st.metric("Academic Standing", f"Year {profile.get('year', 1)}")
    with col_m3:
        st.metric("Selected Pathway", target_track)
    with col_m4:
        st.metric("Readiness Score", f"{profile.get('last_readiness_score', 0.0):.1f}%")

    st.divider()

    # Generate explainability and readiness bundle
    readiness_data = rule_engine.calculate_career_readiness(
        target_track, profile, academic_records, student_skills, projects, documents
    )
    explanation_data = explainability.generate_full_explanation(
        profile, academic_records, student_skills, student_interests, target_track
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 🎯 AI Career Recommendation Diagnostics")
        ranked = explanation_data.get("ranked_matches", [])
        for m in ranked[:3]:
            st.write(f"• **{m['track']}**: {m['probability']}% confidence")
            st.progress(m['probability'] / 100.0)

        st.markdown("##### 📌 Identified Skill Deficiencies")
        gaps = explanation_data.get("skill_gaps", [])
        high_gaps = [g for g in gaps if g["priority"] == "High"]
        if high_gaps:
            for g in high_gaps:
                st.error(f"**{g['skill']}**: Current `{g['current_level']}` vs Benchmark `{g['required_level']}` (Gap: {g['gap_levels']} levels)")
        else:
            st.success("No critical high-priority gaps detected.")

        st.markdown("##### 📜 Academic Prerequisite Rule Checks")
        prereq = explanation_data.get("prerequisite_evaluation", {})
        if prereq.get("overall_academic_met"):
            st.success("✅ All foundational academic course prerequisites met.")
        else:
            for v in prereq.get("violations", []):
                st.warning(f"⚠️ {v['subject']}: {v['message']}")

    with col_right:
        st.markdown("#### 🕸️ Competency Radar (Advisee vs Target Benchmark)")
        radar_info = explanation_data.get("radar_comparison", {})
        if radar_info.get("categories"):
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_info["benchmark_values"] + [radar_info["benchmark_values"][0]],
                theta=radar_info["categories"] + [radar_info["categories"][0]],
                fill='toself',
                name='Career Benchmark',
                line=dict(color='#e53e3e', dash='dash')
            ))
            fig.add_trace(go.Scatterpolar(
                r=radar_info["student_values"] + [radar_info["student_values"][0]],
                theta=radar_info["categories"] + [radar_info["categories"][0]],
                fill='toself',
                name='Advisee Current',
                line=dict(color='#3182ce')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 3])),
                margin=dict(l=20, r=20, t=20, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Roadmap Inspection & Feedback
    col_rm, col_fb = st.columns([1, 1])
    with col_rm:
        st.markdown("#### 🗺️ Student A* Roadmap Progress")
        roadmap_items = db.get_roadmap(selected_student_id, target_track)
        if roadmap_items:
            for act in roadmap_items:
                status_color = "green" if act["status"] == "Completed" else ("orange" if act["status"] == "In-Progress" else "gray")
                st.markdown(f"- **Week {act['week_number']}**: {act['activity_title']} `({act['estimated_hours']} hrs)` &bull; :{status_color}[{act['status']}]")
        else:
            st.info("No active roadmap generated for this student yet.")

    with col_fb:
        st.markdown("#### 📝 Record Mentorship Guidance & Action Items")
        with st.form(f"advisor_form_{selected_student_id}"):
            feedback_text = st.text_area("Advisory Comments & Observations", height=120)
            action_items = st.text_input("Concrete Action Items for Student", placeholder="e.g., Complete automated testing course, push Docker project to GitHub")
            submit_note = st.form_submit_button("Submit Guidance Note", use_container_width=True)
            if submit_note:
                if not feedback_text:
                    st.error("Please provide advisory feedback before submitting.")
                else:
                    db.add_advisor_note(selected_student_id, advisor_name, feedback_text, action_items)
                    st.success("Mentorship note successfully recorded!")
                    st.rerun()

        # Prior notes
        notes = db.get_advisor_notes(selected_student_id)
        if notes:
            st.markdown("##### 📜 Previous Mentorship Records")
            for n in notes[:3]:
                st.caption(f"**{n['advisor_name']}** &bull; {n['created_at']}")
                st.write(n['feedback'])
                if n.get('action_items'):
                    st.info(f"Target: {n['action_items']}")
