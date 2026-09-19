"""
Dataset Generation Script for CareerSense AI.
Synthesizes a realistic dataset of 850 undergraduate computing students
from KDU Faculty of Computing (CS, SE, COE, DBA, IT, IS) with correlated
grades, interest profiles, technical skill proficiencies, and career labels.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import CAREER_TRACKS, DATASET_PATH, ML_FEATURE_COLUMNS

def generate_student_dataset(n_samples=850, random_state=42):
    np.random.seed(random_state)
    
    first_names = [
        "Kasun", "Nimal", "Tharindu", "Chamari", "Dilshan", "Sanduni", "Rashmi",
        "Praveen", "Kavinda", "Methmi", "Ishara", "Nuwan", "Sachini", "Anushka",
        "Dulan", "Shanika", "Hasitha", "Maleesha", "Buddhika", "Supun", "Chathuri",
        "Amal", "Sewwandi", "Akila", "Dinithi", "Sahan", "Kavindi", "Ravindu"
    ]
    last_names = [
        "Bandara", "Perera", "Silva", "Fernando", "Rathnayake", "Ahamad", "Jayasinghe",
        "Gunawardena", "Dissanayake", "Herath", "Karunaratne", "Wickramasinghe",
        "Tharumila", "Fonseka", "Mendis", "Alwis", "Senaratne", "Rajapaksha"
    ]
    degrees = ["BSc (Hons) in Software Engineering", "BSc (Hons) in Computer Science",
               "BSc (Hons) in Information Technology", "BSc (Hons) in Information Systems",
               "BSc (Hons) in Computer Engineering", "BSc (Hons) in Data Science & Business Analytics"]
    
    students = []
    
    # We assign students across the 5 tracks proportionally
    tracks_distribution = [
        ("Software Engineering", int(n_samples * 0.28)),
        ("Data Science / AI", int(n_samples * 0.22)),
        ("Cybersecurity", int(n_samples * 0.18)),
        ("Cloud / DevOps", int(n_samples * 0.17)),
        ("UI/UX Design", n_samples - int(n_samples * 0.28) - int(n_samples * 0.22) - int(n_samples * 0.18) - int(n_samples * 0.17))
    ]
    
    student_id_counter = 1001
    
    for track_name, count in tracks_distribution:
        for _ in range(count):
            first = np.random.choice(first_names)
            last = np.random.choice(last_names)
            name = f"{first} {last}"
            reg_no = f"D/IT/24/{student_id_counter:04d}"
            degree = np.random.choice(degrees)
            year = np.random.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.35, 0.15])
            
            # Base grades (0.0 to 4.0 scale)
            # Default moderate grades across subjects
            g_prog = np.clip(np.random.normal(2.8, 0.6), 1.0, 4.0)
            g_math = np.clip(np.random.normal(2.6, 0.6), 1.0, 4.0)
            g_db   = np.clip(np.random.normal(2.7, 0.5), 1.0, 4.0)
            g_net  = np.clip(np.random.normal(2.5, 0.6), 1.0, 4.0)
            g_sys  = np.clip(np.random.normal(2.5, 0.6), 1.0, 4.0)
            g_des  = np.clip(np.random.normal(2.6, 0.6), 1.0, 4.0)
            
            # Base interests (1 to 5)
            int_se  = np.random.randint(1, 4)
            int_ds  = np.random.randint(1, 4)
            int_sec = np.random.randint(1, 4)
            int_cld = np.random.randint(1, 4)
            int_ui  = np.random.randint(1, 4)
            
            # Base skills (0 to 3: 0=None, 1=Beginner, 2=Intermediate, 3=Advanced)
            # Default low-moderate
            skills = {col: np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15]) for col in ML_FEATURE_COLUMNS if col.startswith("skill_")}
            
            # Inject strong realistic signals according to the ground truth track
            if track_name == "Software Engineering":
                g_prog = np.clip(np.random.normal(3.5, 0.4), 2.7, 4.0)
                g_db   = np.clip(np.random.normal(3.2, 0.5), 2.3, 4.0)
                int_se = np.random.choice([4, 5], p=[0.3, 0.7])
                skills["skill_oop"] = np.random.choice([2, 3], p=[0.4, 0.6])
                skills["skill_dsa"] = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                skills["skill_web_api"] = np.random.choice([1, 2, 3], p=[0.25, 0.5, 0.25])
                skills["skill_testing"] = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                skills["skill_git"] = np.random.choice([2, 3], p=[0.4, 0.6])
                
            elif track_name == "Data Science / AI":
                g_math = np.clip(np.random.normal(3.5, 0.4), 2.7, 4.0)
                g_prog = np.clip(np.random.normal(3.2, 0.5), 2.3, 4.0)
                int_ds = np.random.choice([4, 5], p=[0.25, 0.75])
                skills["skill_python_data"] = np.random.choice([2, 3], p=[0.3, 0.7])
                skills["skill_ml"] = np.random.choice([1, 2, 3], p=[0.2, 0.55, 0.25])
                skills["skill_statistics"] = np.random.choice([2, 3], p=[0.4, 0.6])
                skills["skill_sql"] = np.random.choice([2, 3], p=[0.35, 0.65])
                skills["skill_visualization"] = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                
            elif track_name == "Cybersecurity":
                g_net = np.clip(np.random.normal(3.5, 0.4), 2.7, 4.0)
                g_sys = np.clip(np.random.normal(3.3, 0.5), 2.3, 4.0)
                int_sec = np.random.choice([4, 5], p=[0.25, 0.75])
                skills["skill_network_sec"] = np.random.choice([2, 3], p=[0.3, 0.7])
                skills["skill_cryptography"] = np.random.choice([1, 2, 3], p=[0.25, 0.5, 0.25])
                skills["skill_linux"] = np.random.choice([2, 3], p=[0.35, 0.65])
                skills["skill_vuln_assess"] = np.random.choice([1, 2, 3], p=[0.3, 0.45, 0.25])
                skills["skill_secure_code"] = np.random.choice([1, 2], p=[0.6, 0.4])
                
            elif track_name == "Cloud / DevOps":
                g_sys = np.clip(np.random.normal(3.4, 0.4), 2.7, 4.0)
                g_net = np.clip(np.random.normal(3.1, 0.5), 2.3, 4.0)
                int_cld = np.random.choice([4, 5], p=[0.2, 0.8])
                skills["skill_docker"] = np.random.choice([2, 3], p=[0.35, 0.65])
                skills["skill_cicd"] = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                skills["skill_cloud_infra"] = np.random.choice([2, 3], p=[0.4, 0.6])
                skills["skill_linux"] = np.random.choice([2, 3], p=[0.3, 0.7])
                skills["skill_monitoring"] = np.random.choice([1, 2], p=[0.55, 0.45])
                
            elif track_name == "UI/UX Design":
                g_des = np.clip(np.random.normal(3.6, 0.35), 3.0, 4.0)
                int_ui = np.random.choice([4, 5], p=[0.2, 0.8])
                skills["skill_figma"] = np.random.choice([2, 3], p=[0.25, 0.75])
                skills["skill_user_research"] = np.random.choice([2, 3], p=[0.35, 0.65])
                skills["skill_design_principles"] = np.random.choice([2, 3], p=[0.3, 0.7])
                skills["skill_design_systems"] = np.random.choice([1, 2, 3], p=[0.3, 0.45, 0.25])
                skills["skill_frontend"] = np.random.choice([1, 2], p=[0.5, 0.5])
            
            # Calculate GPA as mean of grades
            gpa = np.round(np.mean([g_prog, g_math, g_db, g_net, g_sys, g_des]), 2)
            
            row = {
                "student_id": f"STU_{student_id_counter}",
                "name": name,
                "reg_no": reg_no,
                "degree": degree,
                "year": year,
                "gpa": gpa,
                "grade_programming": np.round(g_prog, 2),
                "grade_math": np.round(g_math, 2),
                "grade_database": np.round(g_db, 2),
                "grade_networking": np.round(g_net, 2),
                "grade_systems": np.round(g_sys, 2),
                "grade_design": np.round(g_des, 2),
                "interest_software": int_se,
                "interest_data": int_ds,
                "interest_security": int_sec,
                "interest_cloud": int_cld,
                "interest_design": int_ui,
                "career_track": track_name
            }
            # Append all skills
            row.update(skills)
            students.append(row)
            student_id_counter += 1
            
    df = pd.DataFrame(students)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Generated {len(df)} student profiles saved to {DATASET_PATH}")
    print("Class distribution:")
    print(df["career_track"].value_counts())
    return df

if __name__ == "__main__":
    generate_student_dataset()
