"""
Authentication & Role-Based Access Control Module for CareerSense AI.
Handles user sign-in, registration, session persistence, and role guards.
"""
import streamlit as st
import sqlite3
from database.db_manager import DatabaseManager
from config import CAREER_TRACKS, DEGREE_PROGRAMMES, DEGREE_MODULE_CATALOG
from modules.ui import page_header

def init_auth_session():
    """Ensures authentication keys are present in st.session_state."""
    defaults = {
        "authenticated": False,
        "user_id": None,
        "username": None,
        "role": None,
        "full_name": None,
        "reg_no": None,
        "active_tab": "Overview",
        "is_demo": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def login_user(user: dict, is_demo: bool = False):
    st.session_state.authenticated = True
    st.session_state.user_id = user["id"]
    st.session_state.username = user["username"]
    st.session_state.role = user["role"]
    st.session_state.full_name = user["full_name"]
    st.session_state.reg_no = user.get("reg_no", "")
    st.session_state.is_demo = is_demo or (user.get("username") in ["student_demo", "advisor", "coordinator", "admin"])

def logout_user():
    st.session_state.clear()
    init_auth_session()
    st.rerun()

def render_login_and_registration(db: DatabaseManager):
    """Renders the combined Sign-in / Sign-up dialog."""
    page_header(
        "KDU Faculty of Computing",
        "Build a career plan from the evidence you have",
        "Compare computing pathways, inspect the reasoning, and turn skill gaps into a practical weekly plan. This academic prototype does not predict employment outcomes.",
    )

    tab_login, tab_register, tab_demo = st.tabs(["Sign in", "Create account", "Explore demo"])
    
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign in", type="primary", width="stretch")
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = db.authenticate_user(username, password)
                    if user:
                        login_user(user)
                        st.success(f"Welcome back, {user['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

    with tab_register:
        st.markdown("#### Create a student workspace")
        st.caption("Record only completed coursework. You can add interests, skills, and evidence after registration.")

        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            r_fullname = st.text_input("Full name", key="reg_fn")
            r_username = st.text_input("Username", key="reg_un")
            r_password = st.text_input("Password", type="password", help="Use at least 8 characters.", key="reg_pw")
            r_email = st.text_input("University email", key="reg_em")
        with col_reg2:
            r_regno = st.text_input("Registration number", placeholder="D/BCS/27/0052", key="reg_rn")
            r_degree = st.selectbox("Degree programme", DEGREE_PROGRAMMES, key="reg_deg")
            r_year = st.selectbox("Academic year", [1, 2, 3, 4], index=1, key="reg_yr")
            r_target = st.selectbox("Pathway to explore first", CAREER_TRACKS, key="reg_tgt")
            r_hours = st.slider("Study hours available each week", 4, 30, 8, key="reg_hrs")

        st.divider()
        st.markdown(f"#### Completed coursework for {r_degree}")
        st.caption("Leave modules you have not completed as ‘Not taken yet’.")

        curriculum = DEGREE_MODULE_CATALOG.get(r_degree, [])
        grade_options = ["Not taken yet", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
        
        selected_grades = {}
        col_m_left, col_m_right = st.columns(2)
        for i, mod in enumerate(curriculum):
            target_col = col_m_left if i % 2 == 0 else col_m_right
            with target_col:
                selected_grades[mod["code"]] = st.selectbox(
                    f"{mod['code']} · {mod['name']} ({mod['credits']} credits)",
                    options=grade_options,
                    index=0,
                    key=f"reg_mod_{r_degree}_{mod['code']}"
                )

        if st.button("Create student workspace", type="primary", width="stretch"):
            if not r_fullname or not r_username or not r_password:
                st.error("Enter your full name, username, and password.")
            elif len(r_password) < 8:
                st.error("Use a password with at least 8 characters.")
            else:
                try:
                    uid = db.create_user(
                        username=r_username,
                        password=r_password,
                        role="student",
                        full_name=r_fullname,
                        reg_no=r_regno,
                        email=r_email
                    )
                    
                    # Collect only the modules the student actually completed
                    completed_records = []
                    from config import GRADE_POINTS
                    for mod in curriculum:
                        grade_val = selected_grades.get(mod["code"], "Not taken yet")
                        if grade_val != "Not taken yet":
                            completed_records.append({
                                "subject_area": mod["subject"],
                                "module_code": mod["code"],
                                "module_name": mod["name"],
                                "grade": grade_val,
                                "grade_points": GRADE_POINTS.get(grade_val, 2.0)
                            })
                    
                    # Compute initial GPA based on actual completed modules
                    initial_gpa = 0.0
                    if completed_records:
                        total_pts = sum(r["grade_points"] for r in completed_records)
                        initial_gpa = round(total_pts / len(completed_records), 2)

                    db.save_student_profile(
                        user_id=uid,
                        degree=r_degree,
                        year=r_year,
                        gpa=initial_gpa,
                        target_career=r_target,
                        weekly_hours=r_hours
                    )
                    
                    if completed_records:
                        db.set_academic_records(uid, completed_records)

                    # Initialize domain interests
                    initial_interests = {
                        "Software Development & Systems": 3,
                        "Data Analysis & AI Research": 3,
                        "Cybersecurity & Threat Defense": 3,
                        "Cloud Infrastructure & Automation": 3,
                        "UI/UX Design & User Experience": 3
                    }
                    db.set_student_interests(uid, initial_interests)

                    # Auto login newly registered student
                    user_dict = db.get_user_by_id(uid)
                    if user_dict:
                        login_user(user_dict)
                        st.success(f"Welcome, {r_fullname}. Your workspace is ready.")
                        st.rerun()
                    else:
                        st.success("Account created successfully! Please sign in using your credentials.")
                except sqlite3.IntegrityError:
                    st.error("That username is already in use. Choose another one.")
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("The account could not be created. Try again or contact the administrator.")

    with tab_demo:
        st.info("Use a synthetic persona to inspect each workspace. Demo data must not be treated as a real student record.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Open student demo", width="stretch"):
                user = db.authenticate_user("student_demo", "student123")
                if user:
                    login_user(user, is_demo=True)
                    st.rerun()

        with col2:
            if st.button("Open advisor demo", width="stretch"):
                user = db.authenticate_user("advisor", "advisor123")
                if user:
                    login_user(user, is_demo=True)
                    st.rerun()
        col3, col4 = st.columns(2)
        with col3:
            if st.button("Open coordinator demo", width="stretch"):
                user = db.authenticate_user("coordinator", "coordinator123")
                if user:
                    login_user(user, is_demo=True)
                    st.rerun()
        with col4:
            if st.button("Open administrator demo", width="stretch"):
                user = db.authenticate_user("admin", "admin123")
                if user:
                    login_user(user, is_demo=True)
                    st.rerun()
