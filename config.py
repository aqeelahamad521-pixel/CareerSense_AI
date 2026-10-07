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

# 6 Recognized Degree Programmes in KDU Faculty of Computing
DEGREE_PROGRAMMES = [
    "BSc (Hons) in Software Engineering",
    "BSc (Hons) in Computer Science",
    "BSc (Hons) in Information Technology",
    "BSc (Hons) in Information Systems",
    "BSc (Hons) in Computer Engineering",
    "BSc (Hons) in Data Science & Business Analytics"
]

# Field-Specific Module Catalogs for all degree options
DEGREE_MODULE_CATALOG = {
    "BSc (Hons) in Software Engineering": [
        {"code": "SE1103", "name": "Structured Programming & Logic", "subject": "Programming", "credits": 3},
        {"code": "SE1112", "name": "Discrete Mathematics for Computing", "subject": "Mathematics & Statistics", "credits": 2},
        {"code": "SE1203", "name": "Object-Oriented Programming (Java/C++)", "subject": "Programming", "credits": 3},
        {"code": "SE1213", "name": "Database Management Systems", "subject": "Databases", "credits": 3},
        {"code": "SE2103", "name": "Data Structures & Algorithms", "subject": "Programming", "credits": 3},
        {"code": "SE2113", "name": "Software Requirements Engineering", "subject": "Human Computer Interaction & Design", "credits": 3},
        {"code": "SE2122", "name": "Computer Networks & Protocols", "subject": "Networking", "credits": 2},
        {"code": "SE2203", "name": "Software Architecture & Design Patterns", "subject": "Programming", "credits": 3},
        {"code": "SE2213", "name": "Software Quality Assurance & Testing", "subject": "Programming", "credits": 3},
        {"code": "SE2223", "name": "Operating Systems & Systems Programming", "subject": "Operating Systems & Architecture", "credits": 3}
    ],
    "BSc (Hons) in Computer Science": [
        {"code": "CS1103", "name": "Principles of Programming", "subject": "Programming", "credits": 3},
        {"code": "CS1113", "name": "Calculus & Linear Algebra", "subject": "Mathematics & Statistics", "credits": 3},
        {"code": "CS1203", "name": "Object-Oriented Programming", "subject": "Programming", "credits": 3},
        {"code": "CS1212", "name": "Probability & Statistics for CS", "subject": "Mathematics & Statistics", "credits": 2},
        {"code": "CS2103", "name": "Data Structures & Algorithms", "subject": "Programming", "credits": 3},
        {"code": "CS2113", "name": "Computer Organization & Architecture", "subject": "Operating Systems & Architecture", "credits": 3},
        {"code": "CS2123", "name": "Database Systems & Design", "subject": "Databases", "credits": 3},
        {"code": "CS2203", "name": "Theory of Computation & Automata", "subject": "Mathematics & Statistics", "credits": 3},
        {"code": "CS2213", "name": "Operating Systems Principles", "subject": "Operating Systems & Architecture", "credits": 3},
        {"code": "CS2223", "name": "Artificial Intelligence Fundamentals", "subject": "Programming", "credits": 3}
    ],
    "BSc (Hons) in Information Technology": [
        {"code": "IT1103", "name": "Introduction to Programming & Scripting", "subject": "Programming", "credits": 3},
        {"code": "IT1112", "name": "Mathematics for Information Technology", "subject": "Mathematics & Statistics", "credits": 2},
        {"code": "IT1203", "name": "Web Technologies & Interactive Media", "subject": "Human Computer Interaction & Design", "credits": 3},
        {"code": "IT1213", "name": "Database Systems & Data Modeling", "subject": "Databases", "credits": 3},
        {"code": "IT2103", "name": "Data Communication & Computer Networks", "subject": "Networking", "credits": 3},
        {"code": "IT2113", "name": "Object-Oriented Application Development", "subject": "Programming", "credits": 3},
        {"code": "IT2123", "name": "System Administration & Linux", "subject": "Operating Systems & Architecture", "credits": 3},
        {"code": "IT2203", "name": "Network Routing & Switching", "subject": "Networking", "credits": 3},
        {"code": "IT2213", "name": "Cloud Infrastructure Services & DevOps", "subject": "Operating Systems & Architecture", "credits": 3},
        {"code": "IT2223", "name": "Information Assurance & Cyber Defense", "subject": "Networking", "credits": 3}
    ],
    "BSc (Hons) in Information Systems": [
        {"code": "IS1103", "name": "Foundations of Information Systems", "subject": "Databases", "credits": 3},
        {"code": "IS1112", "name": "Business Mathematics & Quantitative Methods", "subject": "Mathematics & Statistics", "credits": 2},
        {"code": "IS1203", "name": "Business Application Programming", "subject": "Programming", "credits": 3},
        {"code": "IS1213", "name": "Database Design & Enterprise Management", "subject": "Databases", "credits": 3},
        {"code": "IS2103", "name": "Systems Analysis & Business Process Modeling", "subject": "Human Computer Interaction & Design", "credits": 3},
        {"code": "IS2113", "name": "Enterprise Architecture & ERP Systems", "subject": "Databases", "credits": 3},
        {"code": "IS2123", "name": "Business Intelligence & Data Mining", "subject": "Databases", "credits": 3},
        {"code": "IS2203", "name": "E-Commerce Platforms & Digital Design", "subject": "Human Computer Interaction & Design", "credits": 3},
        {"code": "IS2213", "name": "Information Systems Security & Audit", "subject": "Networking", "credits": 3},
        {"code": "IS2223", "name": "IT Project Management & Strategy", "subject": "Operating Systems & Architecture", "credits": 3}
    ],
    "BSc (Hons) in Computer Engineering": [
        {"code": "CE1103", "name": "Structured Programming for Engineers", "subject": "Programming", "credits": 3},
        {"code": "CE1113", "name": "Engineering Mathematics & Differential Equations", "subject": "Mathematics & Statistics", "credits": 3},
        {"code": "CE1203", "name": "Digital Electronics & Logic Design", "subject": "Operating Systems & Architecture", "credits": 3},
        {"code": "CE1213", "name": "Data Structures & Algorithms", "subject": "Programming", "credits": 3},
        {"code": "CE2103", "name": "Microprocessor & Embedded Systems", "subject": "Operating Systems & Architecture", "credits": 3},
        {"code": "CE2113", "name": "Computer Architecture & Organization", "subject": "Operating Systems & Architecture", "credits": 3},
        {"code": "CE2123", "name": "Computer Communication & Network Protocols", "subject": "Networking", "credits": 3},
        {"code": "CE2203", "name": "Signals & Systems Analysis", "subject": "Mathematics & Statistics", "credits": 3},
        {"code": "CE2213", "name": "Real-Time Embedded Operating Systems", "subject": "Operating Systems & Architecture", "credits": 3},
        {"code": "CE2223", "name": "Hardware Security & Cryptographic Hardware", "subject": "Networking", "credits": 3}
    ],
    "BSc (Hons) in Data Science & Business Analytics": [
        {"code": "BA1102", "name": "Introduction to Data Science & Analytics", "subject": "Databases", "credits": 2},
        {"code": "BA1113", "name": "Linear Algebra & Optimization for Analytics", "subject": "Mathematics & Statistics", "credits": 3},
        {"code": "BA1203", "name": "Python for Data Science & Computing", "subject": "Programming", "credits": 3},
        {"code": "BA1213", "name": "Applied Statistics & Probability Theory", "subject": "Mathematics & Statistics", "credits": 3},
        {"code": "BA2103", "name": "Database Systems & Big Data Querying", "subject": "Databases", "credits": 3},
        {"code": "BA2113", "name": "Statistical Machine Learning & Modeling", "subject": "Programming", "credits": 3},
        {"code": "BA2123", "name": "Data Visualization & Business Dashboards", "subject": "Human Computer Interaction & Design", "credits": 3},
        {"code": "BA2203", "name": "Time Series Analysis & Business Forecasting", "subject": "Mathematics & Statistics", "credits": 3},
        {"code": "BA2213", "name": "Optimization & Decision Modeling", "subject": "Mathematics & Statistics", "credits": 3},
        {"code": "BA2223", "name": "Big Data Technologies & Cloud Warehousing", "subject": "Operating Systems & Architecture", "credits": 3}
    ]
}

# Degree-to-Career Pathway Natural Alignment Priors
DEGREE_CAREER_ALIGNMENT = {
    "BSc (Hons) in Software Engineering": {
        "Software Engineering": 0.45,
        "Cloud / DevOps": 0.25,
        "UI/UX Design": 0.15,
        "Cybersecurity": 0.10,
        "Data Science / AI": 0.05
    },
    "BSc (Hons) in Computer Science": {
        "Software Engineering": 0.35,
        "Data Science / AI": 0.30,
        "Cybersecurity": 0.15,
        "Cloud / DevOps": 0.15,
        "UI/UX Design": 0.05
    },
    "BSc (Hons) in Information Technology": {
        "Cloud / DevOps": 0.35,
        "Cybersecurity": 0.30,
        "Software Engineering": 0.20,
        "UI/UX Design": 0.10,
        "Data Science / AI": 0.05
    },
    "BSc (Hons) in Information Systems": {
        "UI/UX Design": 0.35,
        "Data Science / AI": 0.25,
        "Software Engineering": 0.20,
        "Cloud / DevOps": 0.10,
        "Cybersecurity": 0.10
    },
    "BSc (Hons) in Computer Engineering": {
        "Cloud / DevOps": 0.35,
        "Cybersecurity": 0.30,
        "Software Engineering": 0.25,
        "Data Science / AI": 0.05,
        "UI/UX Design": 0.05
    },
    "BSc (Hons) in Data Science & Business Analytics": {
        "Data Science / AI": 0.50,
        "Software Engineering": 0.20,
        "Cloud / DevOps": 0.15,
        "UI/UX Design": 0.10,
        "Cybersecurity": 0.05
    }
}

