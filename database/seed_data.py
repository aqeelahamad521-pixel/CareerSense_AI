"""
Seed Data Script for CareerSense AI.
Populates the database with system users and the primary student profile:
Aqeel Ahamad (MFA Ahamad), Year 2, BSc (Hons) in Data Science & Business Analytics, GPA 3.40.
"""
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from database.db_manager import DatabaseManager
from config import DATABASE_PATH

def seed_database():
    # If DB exists, re-initialize cleanly
    db = DatabaseManager()
    
    print("Seeding database with Aqeel Ahamad (MFA Ahamad) profile...")
    
    # 1. System Administrative Users
    admin_id = db.create_user(
        username="admin",
        password="admin123",
        role="admin",
        full_name="System Administrator",
        email="admin@kdu.ac.lk"
    )
    
    advisor_id = db.create_user(
        username="advisor",
        password="advisor123",
        role="advisor",
        full_name="Dr. Nihal Fernando",
        email="n.fernando@kdu.ac.lk"
    )
    
    coord_id = db.create_user(
        username="coordinator",
        password="coordinator123",
        role="coordinator",
        full_name="Prof. K. Jayasinghe",
        email="k.jayasinghe@kdu.ac.lk"
    )
    
    # 2. Primary Student Persona: Aqeel Ahamad (MFA Ahamad)
    # Year 2, Data Science and Business Analytics, GPA 3.40
    student1_id = db.create_user(
        username="student_demo",
        password="student123",
        role="student",
        full_name="Aqeel Ahamad (MFA Ahamad)",
        reg_no="STU-DEMO-01",
        email="demo.student@kdu.ac.lk"
    )
    
    db.save_student_profile(
        user_id=student1_id,
        degree="BSc (Hons) in Data Science & Business Analytics",
        year=2,
        gpa=3.40,
        target_career="Data Science / AI",
        weekly_hours=8,
        cv_filename="Aqeel_Ahamad_CV.pdf",
        cv_extracted_skills=["Python", "Pandas", "NumPy", "SQL", "Machine Learning", "Statistics", "Data Visualization"],
        readiness_score=72.5
    )
    
    # Academic records for Aqeel Ahamad yielding exactly 3.40 GPA
    db.set_academic_records(student1_id, [
        {"subject_area": "Mathematics & Statistics", "module_code": "BA2101", "module_name": "Applied Statistics & Probability", "grade": "A"},
        {"subject_area": "Mathematics & Statistics", "module_code": "MA1102", "module_name": "Linear Algebra & Optimization", "grade": "A-"},
        {"subject_area": "Programming", "module_code": "CS1120", "module_name": "Python for Data Science", "grade": "A-"},
        {"subject_area": "Databases", "module_code": "IT1223", "module_name": "Database Management & Big Data Querying", "grade": "B+"},
        {"subject_area": "Programming", "module_code": "CS2110", "module_name": "Algorithms & Computational Thinking", "grade": "B"},
        {"subject_area": "Operating Systems & Architecture", "module_code": "IT2133", "module_name": "Operating Systems & Cloud Architecture", "grade": "B-"}
    ])
    
    # Interests for Aqeel Ahamad (Data Science & AI highest)
    db.set_student_interests(student1_id, {
        "Data Analysis & AI Research": 5,
        "Software Development & Systems": 4,
        "Cloud Infrastructure & Automation": 3,
        "Cybersecurity & Threat Defense": 2,
        "UI/UX Design & User Experience": 2
    })
    
    # Skills for Aqeel Ahamad
    initial_skills = {
        "Python Data Stack (Pandas/NumPy)": "Intermediate",
        "Statistical Analysis": "Intermediate",
        "SQL & Data Querying": "Intermediate",
        "Machine Learning & Modeling": "Beginner",        # Gap: needs Intermediate
        "Data Visualization": "Beginner",                # Gap: needs Intermediate
        "Object-Oriented Programming": "Beginner",
        "Version Control (Git)": "Intermediate",
        "Cloud Computing (AWS/GCP/Azure)": "Beginner",
        "Linux Systems Administration": "Beginner"
    }
    for sk, lvl in initial_skills.items():
        db.set_student_skill(student1_id, sk, lvl)
        
    # Projects for Aqeel Ahamad
    db.add_student_project(
        user_id=student1_id,
        title="Customer Churn & Predictive Lifetime Value Engine",
        description="Engineered an end-to-end predictive customer retention model utilizing Scikit-learn, Pandas, and interactive Plotly dashboards.",
        technologies="Python, Scikit-learn, Pandas, NumPy, Plotly",
        github_url="https://github.com/aqeel-ahamad/customer-churn-ml",
        demonstrated_skills="Python Data Stack (Pandas/NumPy), Statistical Analysis, Machine Learning & Modeling"
    )
    db.add_student_project(
        user_id=student1_id,
        title="Retail Sales Analytics & Time-Series Forecasting Dashboard",
        description="Developed interactive business analytics portal for forecasting multi-store inventory demand using time-series decomposition.",
        technologies="Python, Streamlit, Statsmodels, SQLite, Seaborn",
        github_url="https://github.com/aqeel-ahamad/retail-sales-analytics",
        demonstrated_skills="SQL & Data Querying, Data Visualization, Python Data Stack (Pandas/NumPy)"
    )
    
    # Documents for Aqeel Ahamad
    db.add_student_document(student1_id, "CV", "Aqeel_Ahamad_CV.pdf", {"pages": 2, "verified_skills": ["Python", "SQL", "Pandas", "Scikit-learn", "Git"]})
    db.add_student_document(student1_id, "Transcript", "Academic_Transcript_Year2.pdf", {"gpa": 3.40, "credits_earned": 64})
    db.add_student_document(student1_id, "Certificate", "IBM_Data_Science_Professional_Certificate.pdf", {"issuer": "IBM Coursera", "date": "2025-10-14"})
    db.add_student_document(student1_id, "Certificate", "DeepLearning_AI_Machine_Learning_Specialization.pdf", {"issuer": "DeepLearning.AI", "date": "2025-07-22"})
    
    # A* Optimized Roadmap for Aqeel Ahamad (Data Science / AI)
    example_roadmap = [
        {"activity_id": "ACT_DS_02", "activity_title": "Exploratory Data Analysis & Visualization", "activity_type": "Project", "week_number": 1, "estimated_hours": 14, "main_skill": "Data Visualization", "status": "In-Progress"},
        {"activity_id": "ACT_DS_04", "activity_title": "Supervised & Unsupervised Machine Learning", "activity_type": "Course", "week_number": 3, "estimated_hours": 18, "main_skill": "Machine Learning & Modeling", "status": "Planned"},
        {"activity_id": "ACT_DS_05", "activity_title": "End-to-End Predictive Machine Learning Project", "activity_type": "Project", "week_number": 5, "estimated_hours": 22, "main_skill": "Machine Learning & Modeling", "status": "Planned"}
    ]
    db.save_roadmap(student1_id, "Data Science / AI", example_roadmap)
    
    # Advisor note for Aqeel Ahamad
    db.add_advisor_note(
        student_id=student1_id,
        advisor_name="Dr. Nihal Fernando",
        feedback="Aqeel demonstrates exceptional quantitative reasoning and strong foundations in Data Science and Business Analytics. Advised to complete an end-to-end ML deployment pipeline to achieve full industry readiness.",
        action_items="Complete Machine Learning project integration; containerize predictive service by end of Semester 2."
    )
    
    print("Database seeding completed successfully for Aqeel Ahamad (MFA Ahamad)!")

if __name__ == "__main__":
    seed_database()
