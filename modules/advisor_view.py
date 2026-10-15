"""Advisor workspace limited to explicitly assigned students."""
import pandas as pd
import streamlit as st

from modules.ui import empty_state, page_header, section_header, status_badge


def render_advisor_view(db, rule_engine, a_star, ml_classifier, explainability):
    advisor_id = st.session_state.user_id
    advisor_name = st.session_state.full_name or "Academic advisor"
    page_header(
        "Advisor workspace",
        "Review evidence before recommending action",
        "Inspect assigned students, distinguish self-reported records from reviewed evidence, and leave specific guidance.",
    )

    students = db.get_students_for_advisor(advisor_id, st.session_state.role)
    if not students:
        empty_state("No assigned students", "An administrator must assign students before their records appear here.")
        return

    options = {f"{student['full_name']} · {student.get('reg_no') or 'No registration number'}": student["id"] for student in students}
    selected_label = st.selectbox("Assigned student", options)
    student_id = options[selected_label]
    profile = db.get_student_profile(student_id) or {}
    academics = db.get_academic_records(student_id)
    skills = db.get_student_skills(student_id)
    skill_records = db.get_student_skill_records(student_id)
    interests = db.get_student_interests(student_id)
    projects = db.get_student_projects(student_id)
    documents = db.get_student_documents(student_id)
    target = profile.get("target_career", "Software Engineering")

    readiness = rule_engine.calculate_career_readiness(
        target, profile, academics, skills, projects, documents,
        {record["skill_name"]: record["evidence_status"] for record in skill_records},
    )
    explanation = explainability.generate_full_explanation(profile, academics, skills, interests, target)
    metrics = st.columns(4)
    metrics[0].metric("Recorded GPA", f"{profile.get('gpa', 0.0):.2f}")
    metrics[1].metric("Academic year", f"Year {profile.get('year', 1)}")
    short_track = {
        "Software Engineering": "Software Eng.",
        "Data Science / AI": "Data / AI",
        "Cybersecurity": "Cybersecurity",
        "Cloud / DevOps": "Cloud / DevOps",
        "UI/UX Design": "UI / UX",
    }.get(target, target)
    metrics[2].metric("Pathway in focus", short_track)
    metrics[3].metric("Readiness index", f"{readiness['total_readiness_score']:.1f}/100")
    st.caption(readiness["index_limitation"])

    tabs = st.tabs(["Review", "Evidence verification", "Learning plan", "Guidance"])
    with tabs[0]:
        section_header("Review", "Use the index and synthetic-model output as prompts for discussion, not as placement decisions.")
        if not explanation.get("is_actionable") or not explanation.get("recommendation_available", True):
            st.warning(explanation.get("evidence_message"))
        left, right = st.columns(2)
        with left:
            st.markdown("#### Pathway affinity")
            for match in explanation.get("ranked_matches", [])[:5]:
                st.write(f"{match['track']} — **{match.get('match_score', 0):.1f}%**")
                st.progress(match.get("match_score", 0) / 100)
            st.caption(explanation.get("limitations"))
        with right:
            st.markdown("#### Priority competency gaps")
            gaps = [gap for gap in explanation.get("skill_gaps", []) if gap["gap_levels"] > 0]
            if gaps:
                frame = pd.DataFrame(gaps)[["skill", "current_level", "required_level", "priority"]]
                frame.columns = ["Skill", "Recorded", "Benchmark", "Priority"]
                st.dataframe(frame, hide_index=True, width="stretch")
            else:
                empty_state("No configured gaps", "Recorded levels meet the current pathway benchmark.")
        with st.expander("Academic prerequisite checks"):
            prerequisite = explanation.get("prerequisite_evaluation", {})
            if prerequisite.get("overall_academic_met"):
                st.success("Recorded coursework meets the configured thresholds.")
            for violation in prerequisite.get("violations", []):
                st.write(f"**{violation['subject']}** — {violation['message']}")

    with tabs[1]:
        section_header("Evidence verification", "Review individual claims. Verification changes the readiness index but does not certify employability.")
        project_tab, document_tab, skill_tab = st.tabs(["Projects", "Documents", "Skills"])
        with project_tab:
            if not projects:
                empty_state("No projects submitted", "The student has not recorded a portfolio project.")
            for project in projects:
                with st.container(border=True):
                    body, action = st.columns([4, 1])
                    body.markdown(f"#### {project['title']}")
                    body.write(project.get("description") or "No description supplied.")
                    body.caption(project.get("technologies") or "No technologies supplied.")
                    body.markdown(status_badge(project.get("verification_status", "self_reported")), unsafe_allow_html=True)
                    desired = action.selectbox(
                        "Review state", ["self_reported", "advisor_verified"],
                        index=1 if project.get("verification_status") == "advisor_verified" else 0,
                        key=f"project_verify_{project['id']}",
                    )
                    if desired != project.get("verification_status") and action.button("Apply", key=f"project_apply_{project['id']}"):
                        db.verify_student_project(project["id"], student_id, advisor_id, st.session_state.role, desired)
                        st.rerun()
        with document_tab:
            st.info("Only metadata and file fingerprints are available; the uploaded file is not retained or parsed.")
            if not documents:
                empty_state("No documents submitted", "The student has not submitted document metadata.")
            for document in documents:
                with st.container(border=True):
                    body, action = st.columns([4, 1])
                    body.markdown(f"#### {document['doc_type']} · {document['filename']}")
                    body.markdown(status_badge(document.get("verification_status", "submitted")), unsafe_allow_html=True)
                    body.caption(f"SHA-256: {document.get('content_sha256') or 'Legacy record without fingerprint'}")
                    states = ["submitted", "advisor_verified", "rejected"]
                    current = document.get("verification_status", "submitted")
                    desired = action.selectbox("Review state", states, index=states.index(current), key=f"doc_verify_{document['id']}")
                    if desired != current and action.button("Apply", key=f"doc_apply_{document['id']}"):
                        db.verify_student_document(document["id"], student_id, advisor_id, st.session_state.role, desired)
                        st.rerun()
        with skill_tab:
            records = skill_records
            if records:
                for record in records:
                    with st.container(border=True):
                        body, action = st.columns([4, 1])
                        body.markdown(f"#### {record['skill_name']}")
                        body.write(f"Recorded level: **{record['proficiency_level']}**")
                        body.markdown(status_badge(record.get("evidence_status", "self_reported")), unsafe_allow_html=True)
                        states = ["self_reported", "advisor_verified"]
                        current = record.get("evidence_status", "self_reported")
                        desired = action.selectbox(
                            "Evidence state", states, index=states.index(current),
                            key=f"skill_verify_{student_id}_{record['skill_name']}",
                        )
                        if desired != current and action.button("Apply", key=f"skill_apply_{student_id}_{record['skill_name']}"):
                            db.verify_student_skill(student_id, record["skill_name"], advisor_id, st.session_state.role, desired)
                            st.rerun()
            else:
                empty_state("No skills recorded", "The student has not submitted a skill self-assessment.")

    with tabs[2]:
        section_header("Learning plan", "Progress reflects activity status only and never upgrades a skill automatically.")
        roadmap = db.get_roadmap(student_id, target)
        if not roadmap:
            empty_state("No saved learning plan", "The student has not generated a plan for this pathway.")
        for activity in roadmap:
            with st.container(border=True):
                st.markdown(f"#### {activity['activity_title']}")
                st.caption(f"Week {activity['week_number']} · {activity['estimated_hours']} hours · {activity['main_skill']}")
                st.markdown(status_badge(activity["status"]), unsafe_allow_html=True)

    with tabs[3]:
        section_header("Guidance", "Leave observations and concrete next actions visible to the student.")
        with st.form(f"guidance_{student_id}"):
            feedback = st.text_area("Observation and rationale", height=140)
            actions = st.text_area("Next actions", height=100)
            if st.form_submit_button("Save guidance note", type="primary", width="stretch"):
                if not feedback.strip():
                    st.error("Enter an observation before saving.")
                else:
                    db.add_advisor_note(
                        student_id, advisor_name, feedback, actions,
                        actor_user_id=advisor_id, actor_role=st.session_state.role,
                    )
                    st.success("Guidance note saved.")
                    st.rerun()
        notes = db.get_advisor_notes(student_id)
        for note in notes:
            with st.container(border=True):
                st.markdown(f"#### {note['advisor_name']}")
                st.caption(note["created_at"])
                st.write(note["feedback"])
                if note.get("action_items"):
                    st.info(note["action_items"])
