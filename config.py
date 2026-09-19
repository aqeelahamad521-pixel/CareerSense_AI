"""
CareerSense AI - Configuration & Constants
Defines global configurations, career tracks, grading scales, and schemas.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "careersense.db"
DATASET_PATH = DATA_DIR / "students_dataset.csv"
MODELS_DIR = BASE_DIR / "ai_engine" / "saved_models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 5 Core Career Tracks (as defined in Project Proposal Appendix A)
CAREER_TRACKS = [
    "Software Engineering",
    "Data Science / AI",
    "Cybersecurity",
    "Cloud / DevOps",
    "UI/UX Design"
]

# Skill Proficiency Levels & Integer Weights
SKILL_LEVELS = {
    "None": 0,
    "Beginner": 1,
    "Intermediate": 2,
    "Advanced": 3
}

LEVEL_TO_NAME = {v: k for k, v in SKILL_LEVELS.items()}

# Academic Grade to Grade Points Mapping (KDU standard GPA scale)
GRADE_POINTS = {
    "A+": 4.0,
    "A": 4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C": 2.0,
    "C-": 1.7,
    "D+": 1.3,
    "D": 1.0,
    "E": 0.0,
    "F": 0.0
}

# Academic Subject Areas evaluated across Computing degrees
SUBJECT_AREAS = [
    "Programming",
    "Mathematics & Statistics",
    "Databases",
    "Networking",
    "Operating Systems & Architecture",
    "Human Computer Interaction & Design"
]

# Core Technical Skills analyzed by the platform
CORE_SKILLS = [
    # Software Engineering skills
    "Object-Oriented Programming",
    "Data Structures & Algorithms",
    "REST APIs & Web Services",
    "Automated Testing & QA",
    "Version Control (Git)",
    
    # Data Science / AI skills
    "Python Data Stack (Pandas/NumPy)",
    "Machine Learning & Modeling",
    "Data Visualization",
    "SQL & Data Querying",
    "Statistical Analysis",
    
    # Cybersecurity skills
    "Network Security & Protocols",
    "Cryptography Fundamentals",
    "Linux Systems Administration",
    "Vulnerability Assessment",
    "Secure Coding Practices",
    
    # Cloud / DevOps skills
    "Containerization (Docker)",
    "CI/CD Pipelines",
    "Cloud Computing (AWS/GCP/Azure)",
    "Infrastructure as Code",
    "System Monitoring & Logging",
    
    # UI/UX skills
    "User Research & Usability Testing",
    "Wireframing & Prototyping (Figma)",
    "Design Principles & Typography",
    "Frontend Framework Awareness",
    "Design Systems & Component Design"
]

# Domain Interest Categories (Rated 1 to 5)
INTEREST_CATEGORIES = [
    "Software Development & Systems",
    "Data Analysis & AI Research",
    "Cybersecurity & Threat Defense",
    "Cloud Infrastructure & Automation",
    "UI/UX Design & User Experience"
]

# ML Feature Columns definition
ML_FEATURE_COLUMNS = [
    # Grades (0.0 to 4.0)
    "grade_programming",
    "grade_math",
    "grade_database",
    "grade_networking",
    "grade_systems",
    "grade_design",
    # Interests (1 to 5)
    "interest_software",
    "interest_data",
    "interest_security",
    "interest_cloud",
    "interest_design",
    # Skill ratings (0 to 3)
    "skill_oop",
    "skill_dsa",
    "skill_web_api",
    "skill_testing",
    "skill_git",
    "skill_python_data",
    "skill_ml",
    "skill_visualization",
    "skill_sql",
    "skill_statistics",
    "skill_network_sec",
    "skill_cryptography",
    "skill_linux",
    "skill_vuln_assess",
    "skill_secure_code",
    "skill_docker",
    "skill_cicd",
    "skill_cloud_infra",
    "skill_iac",
    "skill_monitoring",
    "skill_user_research",
    "skill_figma",
    "skill_design_principles",
    "skill_frontend",
    "skill_design_systems"
]

# User Roles
USER_ROLES = ["student", "advisor", "coordinator", "admin"]
