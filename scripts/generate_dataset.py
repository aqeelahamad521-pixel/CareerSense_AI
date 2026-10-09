"""
Dataset Generation Script for CareerSense AI.
Synthesizes a realistic dataset of 850 undergraduate computing students
from KDU Faculty of Computing (SE, CS, IT, IS, CE, DBA) with correlated
grades, multi-domain interest profiles, technical skill proficiencies,
and ground-truth career labels.

Designed with authentic probabilistic correlations and latent aptitudes,
preventing target leakage and reflecting realistic educational variance.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import (
    CAREER_TRACKS,
    DATASET_PATH,
    ML_FEATURE_COLUMNS,
    DEGREE_PROGRAMMES,
    DEGREE_CAREER_ALIGNMENT
)

DEGREE_CODES = {
    "BSc (Hons) in Software Engineering": "SE",
    "BSc (Hons) in Computer Science": "CS",
    "BSc (Hons) in Information Technology": "IT",
    "BSc (Hons) in Information Systems": "IS",
    "BSc (Hons) in Computer Engineering": "CE",
    "BSc (Hons) in Data Science & Business Analytics": "DBA"
}

def generate_student_dataset(n_samples=850, random_state=42):
    """
    Generates a realistic student cohort dataset.
    
    Features:
    - 6 academic subject area grades (1.0 to 4.0 scale)
    - 5 domain interest ratings (1 to 5 Likert scale)
    - 25 technical skill proficiencies (0: None, 1: Beginner, 2: Intermediate, 3: Advanced)
    
    Ground-truth career tracks are derived from realistic multi-criteria composite
    evaluations with natural boundary ambiguity, avoiding single-feature target leakage.
    """
    np.random.seed(random_state)
    
    first_names = [
        "Kasun", "Nimal", "Tharindu", "Chamari", "Dilshan", "Sanduni", "Rashmi",
        "Praveen", "Kavinda", "Methmi", "Ishara", "Nuwan", "Sachini", "Anushka",
        "Dulan", "Shanika", "Hasitha", "Maleesha", "Buddhika", "Supun", "Chathuri",
        "Amal", "Sewwandi", "Akila", "Dinithi", "Sahan", "Kavindi", "Ravindu",
        "Aqeel", "Fathima", "Kavindu", "Oshada", "Senuri", "Vihan", "Minoli"
    ]
    last_names = [
        "Bandara", "Perera", "Silva", "Fernando", "Rathnayake", "Ahamad", "Jayasinghe",
        "Gunawardena", "Dissanayake", "Herath", "Karunaratne", "Wickramasinghe",
        "Tharumila", "Fonseka", "Mendis", "Alwis", "Senaratne", "Rajapaksha",
        "Peiris", "Abeysekara", "Liyanage", "Cooray", "Gamage", "Wijesinghe"
    ]
    
    degree_probs = [0.22, 0.20, 0.18, 0.14, 0.12, 0.14]
    
    students = []
    student_id_counter = 1001
    
    for _ in range(n_samples):
        first = np.random.choice(first_names)
        last = np.random.choice(last_names)
        name = f"{first} {last}"
        
        degree = np.random.choice(DEGREE_PROGRAMMES, p=degree_probs)
        deg_code = DEGREE_CODES.get(degree, "IT")
        
        year = int(np.random.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.35, 0.15]))
        batch_year = 25 - year
        reg_no = f"D/{deg_code}/{batch_year:02d}/{student_id_counter:04d}"
        
        # Determine latent primary orientation based on degree prior probabilities
        track_priors = DEGREE_CAREER_ALIGNMENT[degree]
        tracks = list(track_priors.keys())
        priors = list(track_priors.values())
        true_track = np.random.choice(tracks, p=priors)
        
        # Base student academic caliber (latent ability)
        base_gpa = np.clip(np.random.normal(3.0, 0.42), 1.8, 3.9)
        
        # Subject grades: correlated with base academic caliber and track orientation
        g_prog = np.clip(np.random.normal(base_gpa + (0.35 if true_track in ["Software Engineering", "Data Science / AI"] else -0.15), 0.40), 1.0, 4.0)
        g_math = np.clip(np.random.normal(base_gpa + (0.35 if true_track == "Data Science / AI" else (0.10 if true_track == "Software Engineering" else -0.20)), 0.45), 1.0, 4.0)
        g_db   = np.clip(np.random.normal(base_gpa + (0.25 if true_track in ["Software Engineering", "Data Science / AI"] else 0.0), 0.40), 1.0, 4.0)
        g_net  = np.clip(np.random.normal(base_gpa + (0.40 if true_track in ["Cybersecurity", "Cloud / DevOps"] else -0.20), 0.40), 1.0, 4.0)
        g_sys  = np.clip(np.random.normal(base_gpa + (0.35 if true_track in ["Cloud / DevOps", "Cybersecurity"] else -0.15), 0.40), 1.0, 4.0)
        g_des  = np.clip(np.random.normal(base_gpa + (0.45 if true_track == "UI/UX Design" else -0.25), 0.40), 1.0, 4.0)
        
        # Multi-domain interests (1 to 5 scale).
        # Students have multi-faceted interests: primary track has higher mean,
        # but secondary related fields are also explored, avoiding target leakage.
        int_se  = int(np.clip(np.round(np.random.normal(4.0 if true_track == "Software Engineering" else (3.1 if true_track in ["Cloud / DevOps", "Data Science / AI"] else 2.2), 0.95)), 1, 5))
        int_ds  = int(np.clip(np.round(np.random.normal(4.1 if true_track == "Data Science / AI" else (2.9 if true_track == "Software Engineering" else 2.1), 0.95)), 1, 5))
        int_sec = int(np.clip(np.round(np.random.normal(4.1 if true_track == "Cybersecurity" else (3.1 if true_track == "Cloud / DevOps" else 2.1), 0.95)), 1, 5))
        int_cld = int(np.clip(np.round(np.random.normal(4.1 if true_track == "Cloud / DevOps" else (3.1 if true_track in ["Software Engineering", "Cybersecurity"] else 2.1), 0.95)), 1, 5))
        int_ui  = int(np.clip(np.round(np.random.normal(4.2 if true_track == "UI/UX Design" else (2.5 if true_track == "Software Engineering" else 1.9), 0.90)), 1, 5))
        
        # Technical skills (0 to 3 scale)
        def sample_skill(domain_match, general=False):
            if domain_match:
                p = [0.08, 0.24, 0.42, 0.26] if year >= 3 else [0.20, 0.45, 0.25, 0.10]
            elif general:
                p = [0.25, 0.40, 0.25, 0.10]
            else:
                p = [0.52, 0.32, 0.13, 0.03]
            return int(np.random.choice([0, 1, 2, 3], p=p))
        
        skills = {
            "skill_oop": sample_skill(true_track in ["Software Engineering", "Data Science / AI"], general=True),
            "skill_dsa": sample_skill(true_track == "Software Engineering"),
            "skill_web_api": sample_skill(true_track in ["Software Engineering", "Cloud / DevOps"]),
            "skill_testing": sample_skill(true_track == "Software Engineering"),
            "skill_git": sample_skill(True, general=True),
            
            "skill_python_data": sample_skill(true_track == "Data Science / AI"),
            "skill_ml": sample_skill(true_track == "Data Science / AI"),
            "skill_visualization": sample_skill(true_track in ["Data Science / AI", "UI/UX Design"]),
            "skill_sql": sample_skill(true_track in ["Data Science / AI", "Software Engineering"], general=True),
            "skill_statistics": sample_skill(true_track == "Data Science / AI"),
            
            "skill_network_sec": sample_skill(true_track == "Cybersecurity"),
            "skill_cryptography": sample_skill(true_track == "Cybersecurity"),
            "skill_linux": sample_skill(true_track in ["Cybersecurity", "Cloud / DevOps"], general=True),
            "skill_vuln_assess": sample_skill(true_track == "Cybersecurity"),
            "skill_secure_code": sample_skill(true_track in ["Cybersecurity", "Software Engineering"]),
            
            "skill_docker": sample_skill(true_track == "Cloud / DevOps"),
            "skill_cicd": sample_skill(true_track == "Cloud / DevOps"),
            "skill_cloud_infra": sample_skill(true_track == "Cloud / DevOps"),
            "skill_iac": sample_skill(true_track == "Cloud / DevOps"),
            "skill_monitoring": sample_skill(true_track == "Cloud / DevOps"),
            
            "skill_user_research": sample_skill(true_track == "UI/UX Design"),
            "skill_figma": sample_skill(true_track == "UI/UX Design"),
            "skill_design_principles": sample_skill(true_track == "UI/UX Design"),
            "skill_frontend": sample_skill(true_track in ["UI/UX Design", "Software Engineering"]),
            "skill_design_systems": sample_skill(true_track == "UI/UX Design"),
        }
        
        gpa = round(float(np.mean([g_prog, g_math, g_db, g_net, g_sys, g_des])), 2)
        
        row = {
            "student_id": f"STU_{student_id_counter}",
            "name": name,
            "reg_no": reg_no,
            "degree": degree,
            "year": year,
            "gpa": gpa,
            "grade_programming": round(float(g_prog), 2),
            "grade_math": round(float(g_math), 2),
            "grade_database": round(float(g_db), 2),
            "grade_networking": round(float(g_net), 2),
            "grade_systems": round(float(g_sys), 2),
            "grade_design": round(float(g_des), 2),
            "interest_software": int_se,
            "interest_data": int_ds,
            "interest_security": int_sec,
            "interest_cloud": int_cld,
            "interest_design": int_ui,
            "career_track": true_track
        }
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
