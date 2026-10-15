"""Student workspace for evidence, pathway exploration, and learning plans."""
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    CAREER_TRACKS,
    CORE_SKILLS,
    DEGREE_MODULE_CATALOG,
    DEGREE_PROGRAMMES,
    GRADE_POINTS,
    INTEREST_CATEGORIES,
    SUBJECT_AREAS,
)
from modules.report_generator import CareerReportGenerator
from modules.ui import empty_state, page_header, section_header, status_badge


def _safe_web_url(value: str) -> bool:
    parsed = urlparse(value or "")
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _render_radar(radar_info):
    if not radar_info.get("categories"):
        empty_state("No competency comparison yet", "Add skills to compare your profile with the selected pathway benchmark.")
        return
    categories = radar_info["categories"]
    student_values = radar_info["student_values"]
    benchmark_values = radar_info["benchmark_values"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values + [benchmark_values[0]],
        theta=categories + [categories[0]],
        name="Pathway benchmark",
        line=dict(color="#666666", dash="dot"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=student_values + [student_values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Recorded level",
        line=dict(color="#131209"),
        fillcolor="rgba(19,18,9,.10)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 3], tickvals=[0, 1, 2, 3], ticktext=["None", "Beginner", "Intermediate", "Advanced"])),
        height=390,
        margin=dict(l=24, r=24, t=24, b=24),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")


def render_student_view(db, rule_engine, a_star, ml_classifier, explainability):
    user_id = st.session_state.user_id
    user_info = db.get_user_by_id(user_id) or {}
    profile = db.get_student_profile(user_id) or {}

    page_header(
        "Student workspace",
        f"Welcome back, {user_info.get('full_name', 'student')}",
        "Use your recorded coursework and evidence to compare pathways, understand gaps, and build a manageable learning plan.",
    )

    academic_records = db.get_academic_records(user_id)
    student_skills = db.get_student_skills(user_id)
    skill_records = db.get_student_skill_records(user_id)
    student_interests = db.get_student_interests(user_id)
    projects = db.get_student_projects(user_id)
    documents = db.get_student_documents(user_id)
    current_target = profile.get("target_career", "Software Engineering")
    weekly_hours = int(profile.get("weekly_hours", 8))

    readiness_data = rule_engine.calculate_career_readiness(
        current_target, profile, academic_records, student_skills, projects, documents,
        {record["skill_name"]: record["evidence_status"] for record in skill_records},
    )
    if readiness_data["total_readiness_score"] != profile.get("last_readiness_score"):
        db.save_student_profile(
            user_id,
            profile.get("degree", DEGREE_PROGRAMMES[0]),
            profile.get("year", 1),
            profile.get("gpa", 0.0),
            current_target,
            weekly_hours,
            readiness_score=readiness_data["total_readiness_score"],
        )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Academic year", f"Year {profile.get('year', 1)}")
    summary_cols[1].metric("Recorded GPA", f"{profile.get('gpa', 0.0):.2f}")
    short_track = {
        "Software Engineering": "Software Eng.",
        "Data Science / AI": "Data / AI",
        "Cybersecurity": "Cybersecurity",
        "Cloud / DevOps": "Cloud / DevOps",
        "UI/UX Design": "UI / UX",
    }.get(current_target, current_target)
    summary_cols[2].metric("Pathway in focus", short_track)
    summary_cols[3].metric("Readiness index", f"{readiness_data['total_readiness_score']:.1f}/100")
    st.caption(readiness_data["index_limitation"])

    with st.expander("Profile and study constraints"):
        c1, c2, c3 = st.columns(3)
        target_value = c1.selectbox(
            "Pathway to explore",
            CAREER_TRACKS,
            index=CAREER_TRACKS.index(current_target) if current_target in CAREER_TRACKS else 0,
        )
        hours_value = c2.slider("Study hours each week", 4, 30, weekly_hours)
        year_value = c3.selectbox("Academic year", [1, 2, 3, 4], index=max(0, min(3, int(profile.get("year", 1)) - 1)))
        if st.button("Save profile settings", type="primary"):
            db.save_student_profile(
                user_id,
                profile.get("degree", DEGREE_PROGRAMMES[0]),
                year_value,
                profile.get("gpa", 0.0),
                target_value,
                hours_value,
            )
            st.success("Profile settings saved.")
            st.rerun()

    tabs = st.tabs(["Pathway review", "Learning plan", "Evidence", "Academic record", "Report"])

    explanation_data = explainability.generate_full_explanation(
        profile, academic_records, student_skills, student_interests, current_target
    )

    with tabs[0]:
        section_header("Pathway review", "A transparent planning view—not a probability of employment or career success.")
        if not explanation_data.get("is_actionable"):
            st.warning(explanation_data.get("evidence_message", "Add more evidence before using this ranking."))
        else:
            if explanation_data.get("recommendation_available"):
                st.info(explanation_data.get("evidence_message", "Use this ranking as a discussion prompt."))
            else:
                st.warning(explanation_data.get("evidence_message", "The result is too close to select one pathway."))
            left, right = st.columns([1, 1])
            with left:
                st.markdown("#### Exploratory affinity")
                for match in explanation_data.get("ranked_matches", []):
                    label = f"{match['track']} · selected" if match.get("is_target") else match["track"]
                    score = float(match.get("match_score", 0.0))
                    name_col, value_col = st.columns([4, 1])
                    name_col.write(label)
                    value_col.markdown(f"**{score:.1f}%**")
                    st.progress(min(1.0, max(0.0, score / 100.0)))
                with st.expander("How this index is calculated"):
                    st.write(explanation_data.get("model_type"))
                    st.caption(explanation_data.get("limitations"))
                    st.write("The selected pathway is highlighted only. Changing it does not boost its score.")
            with right:
                st.markdown("#### Competency comparison")
                _render_radar(explanation_data.get("radar_comparison", {}))

        st.markdown("#### Evidence and gaps for the selected pathway")
        left, right = st.columns(2)
        with left:
            st.markdown("##### Recorded strengths")
            for item in explanation_data.get("influential_factors", {}).get("strengths", []):
                st.success(item)
        with right:
            st.markdown("##### Priority gaps")
            for item in explanation_data.get("influential_factors", {}).get("limitations", []):
                st.warning(item)
        prerequisite = explanation_data.get("prerequisite_evaluation", {})
        with st.expander("Academic prerequisite checks"):
            if prerequisite.get("overall_academic_met"):
                st.success("Recorded coursework meets the configured prerequisite thresholds.")
            else:
                for violation in prerequisite.get("violations", []):
                    st.write(f"**{violation['subject']}** — {violation['message']}")

    with tabs[1]:
        section_header("Learning plan", "A dependency-aware sequence optimized for minimum declared activity hours, then scheduled to your weekly budget.")
        button_col, note_col = st.columns([1, 3])
        if button_col.button("Regenerate plan", type="primary", width="stretch"):
            generated = a_star.generate_optimal_roadmap(current_target, student_skills, weekly_hours)
            db.save_roadmap(user_id, current_target, generated["roadmap"])
            st.success("Learning plan regenerated.")
            st.rerun()
        note_col.caption("Regenerating replaces the saved plan for this pathway. Your recorded skills are not changed automatically.")

        roadmap_items = db.get_roadmap(user_id, current_target)
        if not roadmap_items:
            generated = a_star.generate_optimal_roadmap(current_target, student_skills, weekly_hours)
            db.save_roadmap(user_id, current_target, generated["roadmap"])
            roadmap_items = db.get_roadmap(user_id, current_target)
        plan_info = a_star.generate_optimal_roadmap(current_target, student_skills, weekly_hours)
        if plan_info.get("status") == "partial_coverage":
            st.warning(plan_info["message"])
        else:
            st.info(plan_info.get("message", "Plan ready."))
        st.caption(plan_info.get("optimization_scope", ""))

        total_hours = sum(float(item["estimated_hours"]) for item in roadmap_items)
        completed = sum(item["status"] == "Completed" for item in roadmap_items)
        plan_metrics = st.columns(3)
        plan_metrics[0].metric("Declared effort", f"{total_hours:g} hours")
        plan_metrics[1].metric("Activities", len(roadmap_items))
        plan_metrics[2].metric("Completed", f"{completed}/{len(roadmap_items)}")

        if not roadmap_items:
            empty_state("No activities needed", "The recorded skills already meet the configured benchmark for this pathway.")
        for item in roadmap_items:
            with st.container(border=True):
                body, control = st.columns([4, 1.4])
                end_week = item.get("week_end") or item["week_number"]
                timing = f"Week {item['week_number']}" if end_week == item["week_number"] else f"Weeks {item['week_number']}–{end_week}"
                body.markdown(f"#### {item['activity_title']}")
                body.caption(f"{timing} · {item['estimated_hours']} hours · {item['main_skill']}")
                if item.get("description"):
                    body.write(item["description"])
                body.markdown(status_badge(item["status"]), unsafe_allow_html=True)
                if _safe_web_url(item.get("resource_url", "")):
                    body.link_button("Open learning resource", item["resource_url"])
                status_options = ["Planned", "In-Progress", "Completed"]
                new_status = control.selectbox(
                    "Activity status",
                    status_options,
                    index=status_options.index(item["status"]),
                    key=f"roadmap_status_{item['id']}",
                )
                if new_status != item["status"]:
                    updated = db.update_roadmap_activity_status(item["id"], user_id, new_status)
                    if updated:
                        st.toast("Activity status updated. Skills remain self-reported until you edit or verify them.")
                        st.rerun()

    with tabs[2]:
        section_header("Evidence", "Separate what you report from what an advisor has reviewed.")
        evidence_tabs = st.tabs(["Skills and interests", "Projects", "Documents", "Advisor notes"])
        with evidence_tabs[0]:
            left, right = st.columns([1.4, 1])
            with left:
                st.markdown("#### Skills")
                if skill_records:
                    frame = pd.DataFrame(skill_records).rename(columns={
                        "skill_name": "Skill", "proficiency_level": "Level",
                        "evidence_status": "Evidence", "last_updated": "Updated",
                    })
                    st.dataframe(frame, hide_index=True, width="stretch")
                else:
                    empty_state("No skills recorded", "Add a skill and level to begin a competency comparison.")
            with right:
                st.markdown("#### Add or update a skill")
                with st.form("skill_form"):
                    skill_name = st.selectbox("Technical skill", CORE_SKILLS)
                    skill_level = st.selectbox("Self-assessed level", ["None", "Beginner", "Intermediate", "Advanced"])
                    if st.form_submit_button("Save self-assessment", type="primary", width="stretch"):
                        db.set_student_skill(user_id, skill_name, skill_level, evidence_status="self_reported")
                        st.success("Skill saved as self-reported evidence.")
                        st.rerun()
            st.markdown("#### Domain interests")
            interest_cols = st.columns(2)
            updated_interests = {}
            for index, category in enumerate(INTEREST_CATEGORIES):
                updated_interests[category] = interest_cols[index % 2].slider(
                    category, 1, 5, int(student_interests.get(category, 3)), key=f"interest_{index}"
                )
            if st.button("Save interests"):
                db.set_student_interests(user_id, updated_interests)
                st.success("Interests saved.")
                st.rerun()

        with evidence_tabs[1]:
            left, right = st.columns([1.4, 1])
            with left:
                st.markdown("#### Portfolio projects")
                if not projects:
                    empty_state("No projects recorded", "Add coursework or personal projects and ask an advisor to review the evidence.")
                for project in projects:
                    with st.container(border=True):
                        st.markdown(f"#### {project['title']}")
                        st.markdown(status_badge(project.get("verification_status", "self_reported")), unsafe_allow_html=True)
                        if project.get("description"):
                            st.write(project["description"])
                        st.caption(project.get("technologies", ""))
                        if _safe_web_url(project.get("github_url", "")):
                            st.link_button("Open project link", project["github_url"])
            with right:
                st.markdown("#### Add a project")
                with st.form("project_form"):
                    title = st.text_input("Project title")
                    description = st.text_area("What you built and contributed")
                    technologies = st.text_input("Technologies")
                    project_url = st.text_input("Repository or demo URL")
                    demonstrated = st.text_input("Demonstrated skills")
                    if st.form_submit_button("Add self-reported project", type="primary", width="stretch"):
                        if not title.strip():
                            st.error("Enter a project title.")
                        elif project_url and not _safe_web_url(project_url):
                            st.error("Use a complete http or https URL.")
                        else:
                            db.add_student_project(user_id, title, description, technologies, project_url, demonstrated)
                            st.success("Project added as self-reported evidence.")
                            st.rerun()

        with evidence_tabs[2]:
            st.info("For privacy, CareerSense stores document metadata and a SHA-256 fingerprint only. The uploaded file itself is not retained or parsed.")
            left, right = st.columns([1.4, 1])
            with left:
                if not documents:
                    empty_state("No document metadata recorded", "Submit a CV, transcript, or certificate for advisor review.")
                for document in documents:
                    with st.container(border=True):
                        st.markdown(f"#### {document['doc_type']} · {document['filename']}")
                        st.markdown(status_badge(document.get("verification_status", "submitted")), unsafe_allow_html=True)
                        st.caption(f"Submitted {document['upload_date']}")
            with right:
                document_type = st.selectbox("Document type", ["CV", "Transcript", "Certificate", "Other"])
                uploaded = st.file_uploader("Choose a PDF, DOCX, or text file", type=["pdf", "docx", "txt"])
                if st.button("Record document metadata", type="primary", width="stretch"):
                    if uploaded is None:
                        st.error("Choose a file first.")
                    elif uploaded.size > 10 * 1024 * 1024:
                        st.error("Choose a file smaller than 10 MB.")
                    else:
                        clean_name = Path(uploaded.name).name
                        payload = uploaded.getvalue()
                        digest = hashlib.sha256(payload).hexdigest()
                        db.add_student_document(
                            user_id, document_type, clean_name,
                            extracted_metadata={"file_size": uploaded.size},
                            content_sha256=digest, mime_type=uploaded.type,
                        )
                        st.success("Document metadata recorded for advisor review.")
                        st.rerun()

        with evidence_tabs[3]:
            notes = db.get_advisor_notes(user_id)
            if not notes:
                empty_state("No advisor notes yet", "Guidance from your assigned advisor will appear here.")
            for note in notes:
                with st.container(border=True):
                    st.markdown(f"#### {note['advisor_name']}")
                    st.caption(note["created_at"])
                    st.write(note["feedback"])
                    if note.get("action_items"):
                        st.info(note["action_items"])

    with tabs[3]:
        section_header("Academic record", "Save completed modules only; the recorded GPA is recalculated from these grades.")
        degree = profile.get("degree", DEGREE_PROGRAMMES[0])
        if degree not in DEGREE_MODULE_CATALOG:
            degree = DEGREE_PROGRAMMES[0]
        selected_degree = st.selectbox("Degree programme", DEGREE_PROGRAMMES, index=DEGREE_PROGRAMMES.index(degree))
        if selected_degree != degree:
            st.warning("Changing degree changes the module catalogue. Existing custom modules remain in the record after you save.")
        modules = DEGREE_MODULE_CATALOG.get(selected_degree, [])
        grade_options = ["Not taken yet", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
        existing = {record.get("module_code"): record.get("grade") for record in academic_records}
        with st.form("academic_record_form"):
            columns = st.columns(2)
            selections = {}
            for index, module in enumerate(modules):
                current = existing.get(module["code"], "Not taken yet")
                selections[module["code"]] = columns[index % 2].selectbox(
                    f"{module['code']} · {module['name']}", grade_options,
                    index=grade_options.index(current) if current in grade_options else 0,
                )
            if st.form_submit_button("Save completed modules", type="primary", width="stretch"):
                records = []
                for module in modules:
                    grade = selections[module["code"]]
                    if grade != "Not taken yet":
                        records.append({
                            "subject_area": module["subject"], "module_code": module["code"],
                            "module_name": module["name"], "grade": grade,
                        })
                catalogue_codes = {module["code"] for module in modules}
                records.extend(record for record in academic_records if record.get("module_code") not in catalogue_codes)
                db.save_student_profile(user_id, selected_degree, profile.get("year", 1), profile.get("gpa", 0.0), current_target, weekly_hours)
                db.set_academic_records(user_id, records)
                st.success("Academic record saved and GPA recalculated.")
                st.rerun()
        with st.expander("Add a custom or elective module"):
            with st.form("elective_form"):
                code = st.text_input("Module code")
                name = st.text_input("Module name")
                subject = st.selectbox("Subject area", SUBJECT_AREAS)
                grade = st.selectbox("Grade", list(GRADE_POINTS))
                if st.form_submit_button("Add elective"):
                    if not name.strip():
                        st.error("Enter a module name.")
                    else:
                        updated = [row for row in academic_records if row.get("module_code") != code]
                        updated.append({"subject_area": subject, "module_code": code, "module_name": name, "grade": grade})
                        db.set_academic_records(user_id, updated)
                        st.success("Elective added.")
                        st.rerun()
        if academic_records:
            display = pd.DataFrame(academic_records)[["module_code", "module_name", "subject_area", "grade", "grade_points"]]
            st.dataframe(display, hide_index=True, width="stretch")
        else:
            empty_state("No completed modules recorded", "Select grades above and save your completed modules.")

    with tabs[4]:
        section_header("Career planning report", "Download a portable summary of the current evidence, pathway review, gaps, and learning plan.")
        roadmap_items = db.get_roadmap(user_id, current_target)
        report_html = CareerReportGenerator.generate_html_report(
            profile, user_info, explanation_data, readiness_data, roadmap_items
        )
        st.download_button(
            "Download HTML report", report_html,
            file_name=f"CareerSense_Report_{user_info.get('username', 'student')}.html",
            mime="text/html", type="primary", width="stretch",
        )
        with st.expander("Preview report"):
            st.components.v1.html(report_html, height=700, scrolling=True)
