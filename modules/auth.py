"""
Authentication & Role-Based Access Control Module for CareerSense AI.
Handles user sign-in, registration, session persistence, and role guards.
"""
import streamlit as st
from database.db_manager import DatabaseManager
from config import CAREER_TRACKS

def init_auth_session():
    """Ensures authentication keys are present in st.session_state."""
    defaults = {
        "authenticated": False,
        "user_id": None,
        "username": None,
        "role": None,
        "full_name": None,
        "reg_no": None,
        "active_tab": "Overview"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def login_user(user: dict):
    st.session_state.authenticated = True
    st.session_state.user_id = user["id"]
    st.session_state.username = user["username"]
    st.session_state.role = user["role"]
    st.session_state.full_name = user["full_name"]
    st.session_state.reg_no = user.get("reg_no", "")

def logout_user():
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.full_name = None
    st.session_state.reg_no = None
    st.rerun()

def render_login_and_registration(db: DatabaseManager):
    """Renders the combined Sign-in / Sign-up dialog."""
    st.markdown("### 🎓 Welcome to CareerSense AI")
    st.markdown(
        "An Explainable AI-Powered Career Development & Skill-Roadmap Platform for Undergraduate Students."
    )
    
    tab_login, tab_register, tab_demo = st.tabs(["🔐 Sign In", "📝 Create Account", "⚡ Quick Demo Access"])
    
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
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
        with st.form("register_form"):
            r_fullname = st.text_input("Full Name")
            r_username = st.text_input("Desired Username")
            r_email = st.text_input("University Email")
            r_regno = st.text_input("Registration Number (e.g. D/BIT/24/0041)")
            r_degree = st.selectbox("Undergraduate Degree Programme", [
                "BSc (Hons) in Software Engineering",
                "BSc (Hons) in Computer Science",
                "BSc (Hons) in Information Technology",
                "BSc (Hons) in Information Systems",
                "BSc (Hons) in Computer Engineering",
                "BSc (Hons) in Data Science & Business Analytics"
            ])
            r_year = st.selectbox("Academic Year", [1, 2, 3, 4], index=1)
            r_target = st.selectbox("Initial Career Aspiration", CAREER_TRACKS)
            r_hours = st.slider("Available Study Hours per Week", 4, 30, 8)
            r_password = st.text_input("Password", type="password")
            r_submit = st.form_submit_button("Create Student Account", use_container_width=True)
            
            if r_submit:
                if not r_fullname or not r_username or not r_password:
                    st.error("Please fill in all mandatory fields.")
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
                        db.save_student_profile(
                            user_id=uid,
                            degree=r_degree,
                            year=r_year,
                            gpa=3.0,
                            target_career=r_target,
                            weekly_hours=r_hours
                        )
                        st.success("Account created successfully! Please sign in using your credentials.")
                    except Exception as e:
                        st.error(f"Error registering account: {str(e)}")

    with tab_demo:
        st.info("Select a persona to sign in immediately (Default Student: **Aqeel Ahamad / MFA Ahamad** - Year 2 Data Science & Business Analytics, GPA 3.40):")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🧑‍🎓 Aqeel Ahamad (Student)", use_container_width=True):
                user = db.authenticate_user("student_demo", "student123")
                if user:
                    login_user(user)
                    st.rerun()

        with col2:
            if st.button("👨‍🏫 Academic Advisor", use_container_width=True):
                user = db.authenticate_user("advisor", "advisor123")
                if user:
                    login_user(user)
                    st.rerun()
        with col3:
            if st.button("📊 Coordinator", use_container_width=True):
                user = db.authenticate_user("coordinator", "coordinator123")
                if user:
                    login_user(user)
                    st.rerun()
        with col4:
            if st.button("⚙️ Administrator", use_container_width=True):
                user = db.authenticate_user("admin", "admin123")
                if user:
                    login_user(user)
                    st.rerun()
