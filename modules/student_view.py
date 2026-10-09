"""
Student Portal View for CareerSense AI.
Provides complete interactive functionality for:
- Profile & Academic Records Management
- Skills, Interests & Project Portfolio
- Document Upload & Metadata Extraction
- AI Career Assessment (K-NN & Decision Tree) with Explainability
- Plotly Competency Radar Charts & Readiness Breakdown
- A* Search Learning Roadmap with Weekly Time Budget
- Interactive Activity Progress Tracking & Live Reassessment
- Personalized Career Report Generation & Download
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from config import (
    CAREER_TRACKS, SKILL_LEVELS, SUBJECT_AREAS, CORE_SKILLS,
    INTEREST_CATEGORIES, GRADE_POINTS, DEGREE_PROGRAMMES, DEGREE_MODULE_CATALOG
)
from modules.report_generator import CareerReportGenerator

def render_student_view(db, rule_engine, a_star, ml_classifier, explainability):
    user_id = st.session_state.user_id
    user_info = db.get_user_by_id(user_id) or {}
    profile = db.get_student_profile(user_id) or {}
    
    # Header & Profile Bar
    st.title("🎓 Student Career Development Dashboard")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns([3, 2, 2, 2])
    with col_p1:
        st.subheader(user_info.get("full_name", "Student"))
        st.caption(f"Reg No: **{user_info.get('reg_no', 'N/A')}** | {profile.get('degree', 'Computing')}")
    with col_p2:
        st.metric("Academic Year", f"Year {profile.get('year', 1)}")
    with col_p3:
        st.metric("Cumulative GPA", f"{profile.get('gpa', 0.0):.2f}")
    with col_p4:
        readiness_val = profile.get("last_readiness_score", 0.0)
        st.metric("Career Readiness", f"{readiness_val:.1f}%")

    # Settings expander for quick preference tuning
    with st.expander("⚙️ Profile Settings & Study Constraints", expanded=False):
        c_set1, c_set2, c_set3 = st.columns(3)
        with c_set1:
            target_career = st.selectbox("Target Career Track", CAREER_TRACKS, index=CAREER_TRACKS.index(profile.get("target_career", "Software Engineering")) if profile.get("target_career") in CAREER_TRACKS else 0)
        with c_set2:
            weekly_hours = st.slider("Weekly Available Study Hours", 4, 30, int(profile.get("weekly_hours", 8)), step=1)
        with c_set3:
            academic_year = st.selectbox("Academic Year Level", [1, 2, 3, 4], index=int(profile.get("year", 2)) - 1)
        
        if st.button("Save Profile Preferences", use_container_width=True):
            db.save_student_profile(
                user_id=user_id,
                degree=profile.get("degree", "BSc (Hons) in Data Science & Business Analytics"),
                year=academic_year,
                gpa=profile.get("gpa", 3.40),
                target_career=target_career,
                weekly_hours=weekly_hours
            )
            st.success("Preferences updated!")
            st.rerun()

    # Main Navigation Tabs
    tabs = st.tabs([
        "📊 AI Career Assessment",
        "🧭 A* Learning Roadmap",
        "🎯 Competencies & Skills",
        "📚 Academic Records",
        "💼 Projects & Portfolio",
        "📄 Documents & CV",
        "📝 Advisor Feedback",
        "📥 Download Report"
    ])

    # Fetch fresh student state
    academic_records = db.get_academic_records(user_id)
    student_skills = db.get_student_skills(user_id)
    student_interests = db.get_student_interests(user_id)
    projects = db.get_student_projects(user_id)
    documents = db.get_student_documents(user_id)

    current_target = profile.get("target_career", "Data Science / AI")
    current_weekly_hours = int(profile.get("weekly_hours", 8))

    # Calculate live readiness
    readiness_data = rule_engine.calculate_career_readiness(
        current_target, profile, academic_records, student_skills, projects, documents
    )
    # Update readiness score in profile
    if readiness_data["total_readiness_score"] != profile.get("last_readiness_score"):
        db.save_student_profile(
            user_id=user_id,
            degree=profile.get("degree", "BSc (Hons) in Data Science & Business Analytics"),
            year=profile.get("year", 2),
            gpa=profile.get("gpa", 3.40),
            target_career=current_target,
            weekly_hours=current_weekly_hours,
            readiness_score=readiness_data["total_readiness_score"]
        )


    # -------------------------------------------------------------
    # TAB 1: AI Career Assessment & Explainability
    # -------------------------------------------------------------
    with tabs[0]:
        st.markdown(f"### 🤖 AI Career Track Prediction & Explainability Analysis")
        st.caption("Combines K-Nearest Neighbors (K-NN) proximity modeling, Decision Tree feature attribution, and Rule-Based reasoning.")
        
        if not academic_records:
            st.warning(
                f"📝 **No completed coursework recorded yet for {profile.get('degree', 'your degree')}**!\n\n"
                f"CareerSense AI never assumes your grades. Please go to **Tab 4 ('📚 Academic Records')** to select the grades you have achieved so far in your degree's curriculum. "
                f"The AI engine will then evaluate your actual course performance, compute your cumulative GPA, and generate your career match!"
            )

        explanation_data = explainability.generate_full_explanation(
            profile, academic_records, student_skills, student_interests, current_target
        )

        col_assess_left, col_assess_right = st.columns([1, 1])

        with col_assess_left:
            st.markdown("#### 🏆 Advisory Pathway Match Scores")
            st.caption("Composite advisory affinity index combining ML competency, domain interest, degree curriculum alignment, and target career aspiration (uncalibrated multi-criteria index, not an empirical probability of employment).")
            ranked_matches = explanation_data.get("ranked_matches", [])
            for m in ranked_matches:
                trk = m["track"]
                prob = m["probability"]
                is_selected = (trk == current_target)
                badge = " ⭐ (Target)" if is_selected else ""
                
                col_m1, col_m2 = st.columns([3, 1])
                col_m1.markdown(f"**{trk}{badge}**")
                col_m2.markdown(f"**{prob}%**")
                st.progress(prob / 100.0)

            # Pathway Synergy Callout when Top Match differs from Target Career
            top_track = ranked_matches[0]["track"] if ranked_matches else ""
            if top_track and current_target and top_track != current_target:
                target_prob = next((m["probability"] for m in ranked_matches if m["track"] == current_target), 0.0)
                st.info(
                    f"🎯 **Career Pathway Alignment Insight**:\n\n"
                    f"• **Current Academic Foundation**: Your completed coursework gives you an immediate technical baseline in **{top_track}** ({ranked_matches[0]['probability']}%).\n\n"
                    f"• **Aspirational Goal Track**: You selected **{current_target}** ({target_prob}%) as your target career with high domain affinity.\n\n"
                    f"• **The Bridging Strategy**: Strong programming and database fundamentals are the core prerequisite for modern {current_target}. Your **A* Learning Roadmap** (Tab 2) focuses on closing the specific technical gaps to transition into your dream track!"
                )

            st.caption(f"💡 **Model Pipeline**: {explanation_data.get('model_type', 'K-NN & Decision Tree Pipeline')}")

            # Nearest neighbor distance metric
            nn_distances = explanation_data.get("nearest_neighbor_distances", [])
            if nn_distances:
                avg_dist = round(sum(nn_distances) / len(nn_distances), 2)
                st.markdown("##### 📍 K-NN Instance Proximity")
                st.caption("Student feature vector proximity to historical cohort profiles:")
                st.write(f"Average nearest neighbor distance: **{avg_dist} Euclidean units** (calculated across the 5 closest student profiles in the standardized feature space).")

        with col_assess_right:
            st.markdown("#### 🕸️ Competency Radar Analysis")
            radar_info = explanation_data.get("radar_comparison", {})
            if radar_info.get("categories"):
                categories = radar_info["categories"]
                student_vals = radar_info["student_values"]
                benchmark_vals = radar_info["benchmark_values"]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=benchmark_vals + [benchmark_vals[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Career Benchmark',
                    line=dict(color='#e53e3e', dash='dash')
                ))
                fig.add_trace(go.Scatterpolar(
                    r=student_vals + [student_vals[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Your Current Level',
                    line=dict(color='#3182ce')
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 3],
                            tickvals=[0, 1, 2, 3],
                            ticktext=["None", "Beginner", "Intermediate", "Advanced"]
                        )
                    ),
                    showlegend=True,
                    margin=dict(l=30, r=30, t=20, b=20),
                    height=320
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### 🔍 Explainability Breakdown & Diagnostic Reasoning")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown("##### ✅ Contributing Strengths & Positive Drivers")
            for strength in explanation_data["influential_factors"]["strengths"]:
                st.success(f"• {strength}")
        with col_exp2:
            st.markdown("##### ⚠️ Growth Areas & Priority Gaps")
            for limitation in explanation_data["influential_factors"]["limitations"]:
                st.warning(f"• {limitation}")

        # Prerequisite validation alert
        prereq_eval = explanation_data["prerequisite_evaluation"]
        if not prereq_eval.get("overall_academic_met"):
            st.error("🚨 **Academic Prerequisite Notice:**")
            for viol in prereq_eval.get("violations", []):
                st.write(f"• **{viol['subject']}**: {viol['message']} *Remedy: {viol['remedy']}*")
        else:
            st.success("✅ **Academic Prerequisites Satisfied:** Foundational course requirements for this career track are fully met.")

    # -------------------------------------------------------------
    # TAB 2: A* Learning Roadmap & Progress Tracking
    # -------------------------------------------------------------
    with tabs[1]:
        st.markdown("### 🧭 A* Optimized Skill Development Roadmap")
        st.caption(f"Personalized path generated using **A* Graph Search** over learning activities, constrained to **{current_weekly_hours} study hours/week**.")

        col_act_top1, col_act_top2 = st.columns([3, 1])
        with col_act_top1:
            st.markdown(f"Target Role: **{current_target}** | Time Allocation: **{current_weekly_hours} hrs/week**")
        with col_act_top2:
            if st.button("🔄 Regenerate A* Roadmap", use_container_width=True):
                result = a_star.generate_optimal_roadmap(current_target, student_skills, current_weekly_hours)
                db.save_roadmap(user_id, current_target, result["roadmap"])
                st.success("Optimized roadmap regenerated!")
                st.rerun()

        # Fetch roadmap from DB
        roadmap_items = db.get_roadmap(user_id, current_target)
        if not roadmap_items:
            # Auto-generate initial roadmap if none saved
            result = a_star.generate_optimal_roadmap(current_target, student_skills, current_weekly_hours)
            db.save_roadmap(user_id, current_target, result["roadmap"])
            roadmap_items = db.get_roadmap(user_id, current_target)

        # Roadmap metrics
        total_hours = sum(item["estimated_hours"] for item in roadmap_items)
        max_week = max([item["week_number"] for item in roadmap_items], default=1)
        completed_count = sum(1 for item in roadmap_items if item["status"] == "Completed")
        progress_pct = (completed_count / len(roadmap_items) * 100) if roadmap_items else 100.0

        c_rm1, c_rm2, c_rm3, c_rm4 = st.columns(4)
        c_rm1.metric("Total Planned Effort", f"{total_hours} Hours")
        c_rm2.metric("Target Horizon", f"{max_week} Weeks")
        c_rm3.metric("Completed Milestones", f"{completed_count} / {len(roadmap_items)}")
        c_rm4.metric("Roadmap Progress", f"{progress_pct:.0f}%")

        st.progress(progress_pct / 100.0)
        st.write("")

        # Display sequence of activities
        st.markdown("#### 📋 Milestone Sequence & Activity Tracking")
        for act in roadmap_items:
            act_id = act["id"]
            with st.container(border=True):
                c_item1, c_item2, c_item3 = st.columns([4, 2, 2])
                with c_item1:
                    status_emoji = "✅" if act["status"] == "Completed" else ("⏳" if act["status"] == "In-Progress" else "📌")
                    st.markdown(f"**{status_emoji} Week {act['week_number']}: {act['activity_title']}**")
                    st.caption(f"Main Competency: **{act['main_skill']}** | Effort: **{act['estimated_hours']} hrs**")
                with c_item2:
                    st.badge = act["activity_type"]
                    st.markdown(f"Type: `{act['activity_type']}`")
                with c_item3:
                    current_status = act["status"]
                    new_status = st.selectbox(
                        "Status",
                        ["Planned", "In-Progress", "Completed"],
                        index=["Planned", "In-Progress", "Completed"].index(current_status),
                        key=f"status_select_{act_id}"
                    )
                    if new_status != current_status:
                        db.update_roadmap_activity_status(act_id, new_status)
                        # If marked completed, automatically update student skill in DB!
                        if new_status == "Completed":
                            target_skill = act["main_skill"]
                            current_level = student_skills.get(target_skill, "None")
                            # Advance level
                            new_lvl = "Intermediate" if current_level in ("None", "Beginner") else "Advanced"
                            db.set_student_skill(user_id, target_skill, new_lvl)
                            st.toast(f"🎉 Skill updated! {target_skill} advanced to {new_lvl}!")
                        st.rerun()

    # -------------------------------------------------------------
    # TAB 3: Competencies & Skills
    # -------------------------------------------------------------
    with tabs[2]:
        st.markdown("### 🎯 Skill Management & Competency Self-Assessment")
        st.caption("Record and update your verified proficiency levels across essential software engineering and computing disciplines.")

        col_sk_left, col_sk_right = st.columns([3, 2])
        with col_sk_left:
            st.markdown("#### 🛠️ Current Technical Skills")
            skills_df_data = []
            for sk, lvl in student_skills.items():
                skills_df_data.append({"Skill Name": sk, "Proficiency Level": lvl})
            if skills_df_data:
                st.dataframe(pd.DataFrame(skills_df_data), use_container_width=True, hide_index=True)
            else:
                st.info("No skills recorded yet. Add your skills using the form on the right.")

            # Domain Interests
            st.markdown("#### 🌟 Domain Career Interests (1–5)")
            col_int1, col_int2 = st.columns(2)
            updated_interests = {}
            for i, cat in enumerate(INTEREST_CATEGORIES):
                curr_rate = student_interests.get(cat, 3)
                with (col_int1 if i % 2 == 0 else col_int2):
                    updated_interests[cat] = st.slider(cat, 1, 5, curr_rate, key=f"interest_slider_{i}")
            if st.button("Save Interest Ratings"):
                db.set_student_interests(user_id, updated_interests)
                st.success("Interests saved successfully!")
                st.rerun()

        with col_sk_right:
            st.markdown("#### ➕ Add or Update Skill")
            with st.form("skill_add_form"):
                new_skill_name = st.selectbox("Select Technical Skill", CORE_SKILLS)
                new_skill_level = st.selectbox("Proficiency Level", ["None", "Beginner", "Intermediate", "Advanced"], index=2)
                skill_submit = st.form_submit_button("Save Skill", use_container_width=True)
                if skill_submit:
                    db.set_student_skill(user_id, new_skill_name, new_skill_level)
                    st.success(f"Updated {new_skill_name} to {new_skill_level}!")
                    st.rerun()

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # TAB 4: Academic Records
    # -------------------------------------------------------------
    with tabs[3]:
        student_degree = profile.get("degree", "BSc (Hons) in Data Science & Business Analytics")
        if student_degree not in DEGREE_MODULE_CATALOG:
            student_degree = "BSc (Hons) in Data Science & Business Analytics"

        st.markdown(f"### 📚 Academic Curriculum & Module Records")
        st.caption("CareerSense AI evaluates your actual academic performance. Select the grades you achieved in your degree's curriculum.")

        col_top_act1, col_top_act2 = st.columns([3, 2])
        with col_top_act1:
            st.info(f"🎓 **Enrolled Degree:** {student_degree} &nbsp;|&nbsp; Cumulative GPA: **{profile.get('gpa', 0.0):.2f}**")
        with col_top_act2:
            deg_idx = DEGREE_PROGRAMMES.index(student_degree) if student_degree in DEGREE_PROGRAMMES else 0
            change_deg = st.selectbox("Change Degree Programme", DEGREE_PROGRAMMES, index=deg_idx, key="change_deg_sel")
            if change_deg != student_degree:
                db.save_student_profile(
                    user_id=user_id,
                    degree=change_deg,
                    year=profile.get("year", 2),
                    gpa=profile.get("gpa", 0.0),
                    target_career=current_target,
                    weekly_hours=current_weekly_hours
                )
                st.toast(f"Degree updated to {change_deg}!")
                st.rerun()

        # Curriculum Grade Sheet
        st.markdown(f"#### 📝 Degree Curriculum Grade Sheet: **{student_degree}**")
        st.caption("Select your achieved letter grade for each module. If you have not completed a module yet, leave it as **Not Taken Yet**.")

        curr_modules = DEGREE_MODULE_CATALOG.get(student_degree, [])
        grade_options = ["Not Taken Yet", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]

        # Map existing records by module_code and module_name
        existing_grades_map = {}
        for r in academic_records:
            if r.get("module_code"):
                existing_grades_map[r["module_code"]] = r["grade"]
            if r.get("module_name"):
                existing_grades_map[r["module_name"]] = r["grade"]

        form_grades = {}
        col_sheet_l, col_sheet_r = st.columns(2)
        for i, mod in enumerate(curr_modules):
            target_col = col_sheet_l if i % 2 == 0 else col_sheet_r
            cur_g = existing_grades_map.get(mod["code"], existing_grades_map.get(mod["name"], "Not Taken Yet"))
            sel_idx = grade_options.index(cur_g) if cur_g in grade_options else 0
            with target_col:
                form_grades[mod["code"]] = st.selectbox(
                    f"**[{mod['code']}]** {mod['name']} *({mod['credits']} cr - {mod['subject']})*",
                    options=grade_options,
                    index=sel_idx,
                    key=f"sheet_mod_{student_degree}_{mod['code']}"
                )

        col_save_btn, col_clear_btn = st.columns([3, 1])
        with col_save_btn:
            if st.button("💾 Save Academic Transcript & Recalculate GPA", type="primary", use_container_width=True):
                # Build new records from curriculum
                new_records = []
                for mod in curr_modules:
                    g = form_grades.get(mod["code"], "Not Taken Yet")
                    if g != "Not Taken Yet":
                        new_records.append({
                            "subject_area": mod["subject"],
                            "module_code": mod["code"],
                            "module_name": mod["name"],
                            "grade": g,
                            "grade_points": GRADE_POINTS.get(g, 2.0)
                        })

                # Preserve any custom elective modules not in curr_modules
                curr_codes = {m["code"] for m in curr_modules}
                curr_names = {m["name"] for m in curr_modules}
                for r in academic_records:
                    if r.get("module_code") not in curr_codes and r.get("module_name") not in curr_names:
                        new_records.append(r)

                db.set_academic_records(user_id, new_records)
                st.success("🎉 Academic records updated! GPA recalculated and AI Career Assessment updated.")
                st.rerun()

        with col_clear_btn:
            if st.button("🗑️ Reset All Modules", use_container_width=True, help="Clear all saved module records"):
                db.set_academic_records(user_id, [])
                st.toast("Academic records cleared.")
                st.rerun()

        st.divider()
        col_summary_l, col_summary_r = st.columns([3, 2])
        with col_summary_l:
            st.markdown("#### 📋 Current Completed Transcript")
            if academic_records:
                df_acad = pd.DataFrame(academic_records)[["subject_area", "module_code", "module_name", "grade", "grade_points"]]
                df_acad.columns = ["Subject Domain", "Module Code", "Module Name", "Grade", "Grade Points"]
                st.dataframe(df_acad, use_container_width=True, hide_index=True)
            else:
                st.warning("No completed modules currently saved. Select your grades in the Curriculum Grade Sheet above and click Save.")

        with col_summary_r:
            st.markdown("#### ➕ Add Custom / Elective Module")
            st.caption("If you completed an elective outside your primary degree catalog, record it here.")
            with st.form("custom_elective_form"):
                e_code = st.text_input("Module Code (e.g. ELEC201)")
                e_name = st.text_input("Module Title")
                e_subj = st.selectbox("Subject Domain", SUBJECT_AREAS)
                e_grade = st.selectbox("Grade", list(GRADE_POINTS.keys()), index=1)
                e_sub = st.form_submit_button("Add Elective Module", use_container_width=True)
                if e_sub:
                    if not e_name:
                        st.error("Please enter a valid module title.")
                    else:
                        updated = [r for r in academic_records if r.get("module_code") != e_code and r.get("module_name") != e_name]
                        updated.append({
                            "subject_area": e_subj,
                            "module_code": e_code,
                            "module_name": e_name,
                            "grade": e_grade,
                            "grade_points": GRADE_POINTS.get(e_grade, 2.0)
                        })
                        db.set_academic_records(user_id, updated)
                        st.success(f"Added elective {e_name} ({e_grade})!")
                        st.rerun()

    # -------------------------------------------------------------
    # TAB 5: Projects & Portfolio
    # -------------------------------------------------------------
    with tabs[4]:
        st.markdown("### 💼 Practical Project Portfolio")
        st.caption("Concrete evidence of implemented systems, repositories, and demonstrated technical proficiencies.")

        col_proj_l, col_proj_r = st.columns([3, 2])
        with col_proj_l:
            if projects:
                for p in projects:
                    with st.container(border=True):
                        st.markdown(f"**{p['title']}**")
                        st.write(p['description'])
                        st.caption(f"🛠️ Technologies: `{p['technologies']}`")
                        if p.get('github_url'):
                            st.markdown(f"🔗 [Project Repository / Demo Link]({p['github_url']})")
            else:
                st.info("No projects recorded. Add your university course projects or personal GitHub projects.")

        with col_proj_r:
            st.markdown("#### ➕ Add New Portfolio Project")
            with st.form("project_form"):
                p_title = st.text_input("Project Title")
                p_desc = st.text_area("Summary & Responsibilities")
                p_tech = st.text_input("Technologies (e.g. Python, Docker, React)")
                p_url = st.text_input("GitHub / Demo URL")
                p_skills = st.text_input("Demonstrated Skills (comma separated)")
                p_sub = st.form_submit_button("Add Project to Portfolio", use_container_width=True)
                if p_sub:
                    if not p_title:
                        st.error("Project title is required.")
                    else:
                        db.add_student_project(user_id, p_title, p_desc, p_tech, p_url, p_skills)
                        st.success("Project added to portfolio!")
                        st.rerun()

    # -------------------------------------------------------------
    # TAB 6: Documents & CV
    # -------------------------------------------------------------
    with tabs[5]:
        st.markdown("### 📄 Career Documents & Verification Metadata")
        st.caption("Upload CVs, semester transcripts, and industry certifications.")

        col_doc_l, col_doc_r = st.columns([3, 2])
        with col_doc_l:
            st.markdown("#### 🗄️ Uploaded Documents")
            if documents:
                for doc in documents:
                    with st.container(border=True):
                        st.markdown(f"**{doc['doc_type']}**: `{doc['filename']}`")
                        st.caption(f"Uploaded: {doc['upload_date']}")
            else:
                st.info("No documents uploaded yet.")

        with col_doc_r:
            st.markdown("#### 📤 Upload Document")
            doc_type = st.selectbox("Document Classification", ["CV", "Transcript", "Certificate", "Other"])
            uploaded_file = st.file_uploader("Select PDF or Document", type=["pdf", "docx", "txt"])
            if st.button("Process & Upload Document", use_container_width=True):
                if uploaded_file is not None:
                    db.add_student_document(
                        user_id=user_id,
                        doc_type=doc_type,
                        filename=uploaded_file.name,
                        extracted_metadata={"file_size": uploaded_file.size, "verified": True}
                    )
                    st.success(f"{doc_type} '{uploaded_file.name}' successfully uploaded and indexed!")
                    st.rerun()
                else:
                    st.error("Please choose a file to upload.")

    # -------------------------------------------------------------
    # TAB 7: Advisor Feedback
    # -------------------------------------------------------------
    with tabs[6]:
        st.markdown("### 📝 Academic Advisor Mentorship Feedback")
        advisor_notes = db.get_advisor_notes(user_id)
        if advisor_notes:
            for note in advisor_notes:
                with st.container(border=True):
                    st.markdown(f"**Advisor:** {note['advisor_name']} &bull; <small>{note['created_at']}</small>", unsafe_allow_html=True)
                    st.write(note['feedback'])
                    if note.get('action_items'):
                        st.info(f"📌 **Action Items:** {note['action_items']}")
        else:
            st.info("No advisor feedback logged yet. Your assigned academic advisor will leave guidance notes during mentoring sessions.")

    # -------------------------------------------------------------
    # TAB 8: Download Report
    # -------------------------------------------------------------
    with tabs[7]:
        st.markdown("### 📥 Download Personalized Career Advisory Report")
        st.write("Generate a comprehensive, printable career advisory document summarizing your academic standing, AI predictions, competency gaps, and A* roadmap.")

        report_html = CareerReportGenerator.generate_html_report(
            student_profile=profile,
            user_info=user_info,
            explanation_data=explanation_data,
            readiness_data=readiness_data,
            roadmap_data=roadmap_items
        )

        st.download_button(
            label="📄 Download Full Career Advisory Report (HTML)",
            data=report_html,
            file_name=f"CareerSense_Report_{user_info.get('username', 'student')}.html",
            mime="text/html",
            use_container_width=True
        )
        
        with st.expander("👁️ Preview Report Inline", expanded=True):
            st.components.v1.html(report_html, height=600, scrolling=True)
