"""
CareerSense AI - Main Application Entrypoint
An Explainable AI-Powered Career Development and Skill-Roadmap Platform
Faculty of Computing, General Sir John Kotelawala Defence University (KDU)
"""
import streamlit as st
from pathlib import Path
import sys

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from database.db_manager import DatabaseManager
from database.seed_data import seed_database
from ai_engine.rule_engine import RuleBasedExpertSystem
from ai_engine.a_star_roadmap import AStarRoadmapOptimizer
from ai_engine.ml_classifier import CareerClassifier
from ai_engine.explainability import ExplainabilityEngine
from modules.auth import init_auth_session, render_login_and_registration, logout_user
from modules.student_view import render_student_view
from modules.advisor_view import render_advisor_view
from modules.coordinator_view import render_coordinator_view
from modules.admin_view import render_admin_view

# Streamlit Page Config
st.set_page_config(
    page_title="CareerSense AI | KDU Computing",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    /* Button style */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }
    /* Sidebar header */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_system_components():
    """Initializes and caches database, AI models, rule engine, and A* optimizer."""
    db = DatabaseManager()
    
    # Auto-seed if database is empty
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM users")
        if cursor.fetchone()["count"] == 0:
            seed_database()

    rule_engine = RuleBasedExpertSystem()
    a_star = AStarRoadmapOptimizer()
    ml_classifier = CareerClassifier()
    
    # Auto-train models if not already trained
    if not ml_classifier.is_trained:
        try:
            ml_classifier.train_models()
        except Exception as e:
            print(f"Model initialization notice: {e}")

    explainability = ExplainabilityEngine(rule_engine, ml_classifier)
    return db, rule_engine, a_star, ml_classifier, explainability

def main():
    init_auth_session()
    db, rule_engine, a_star, ml_classifier, explainability = get_system_components()

    # If user is not authenticated, show sign-in/registration screen
    if not st.session_state.authenticated:
        col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
        with col_c2:
            render_login_and_registration(db)
        return

    # Sidebar Navigation & Role Information
    with st.sidebar:
        st.markdown("### 🎓 CareerSense AI")
        st.caption("KDU Faculty of Computing &bull; Intake 41/42")
        st.write(f"Logged in as: **{st.session_state.full_name}**")
        st.write(f"Role: `{st.session_state.role.capitalize()}`")
        if st.session_state.reg_no:
            st.caption(f"Reg No: {st.session_state.reg_no}")

        st.divider()

        # Role switching for demonstration sandbox / administrator only
        if st.session_state.get("is_demo", False) or st.session_state.get("role") == "admin":
            st.markdown("##### 🔀 Quick Persona Switch (Demo)")
            roles_available = ["student", "advisor", "coordinator", "admin"]
            selected_role = st.selectbox(
                "Active Role View",
                roles_available,
                index=roles_available.index(st.session_state.role)
            )
            if selected_role != st.session_state.role:
                st.session_state.role = selected_role
                st.rerun()

        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True):
            logout_user()

        st.markdown("---")
        st.caption("**Demonstration Sandbox**")
        st.caption("• Environment: Local Sandbox")
        st.caption("• Primary Persona: Aqeel Ahamad (Demo Student)")
        st.caption("• Degree: BSc (Hons) in Data Science & Business Analytics")
        st.caption("• Status: Academic Prototype")


    # Dispatch to appropriate role dashboard with server-side authorization check
    current_role = st.session_state.role
    is_demo = st.session_state.get("is_demo", False)

    if current_role == "student":
        render_student_view(db, rule_engine, a_star, ml_classifier, explainability)
    elif current_role == "advisor":
        if st.session_state.role not in ["advisor", "admin"] and not is_demo:
            st.error("⛔ Access Denied: Academic Advisor role required.")
        else:
            render_advisor_view(db, rule_engine, a_star, ml_classifier, explainability)
    elif current_role == "coordinator":
        if st.session_state.role not in ["coordinator", "admin"] and not is_demo:
            st.error("⛔ Access Denied: Programme Coordinator role required.")
        else:
            render_coordinator_view(db, rule_engine)
    elif current_role == "admin":
        if st.session_state.role != "admin" and not is_demo:
            st.error("⛔ Access Denied: Administrator role required.")
        else:
            render_admin_view(db, rule_engine, a_star, ml_classifier)
    else:
        st.error("Unknown user role.")

if __name__ == "__main__":
    main()
