"""
Dataset Generation Script for CareerSense AI.
Synthesizes a realistic dataset of 850 undergraduate computing students
from KDU Faculty of Computing (SE, CS, IT, IS, CE, DBA) with correlated
grades, multi-domain interest profiles, technical skill proficiencies,
and simulated pathway labels.

The generator is for software testing only. Independent latent aptitudes drive
both noisy observations and a noisy label; the label is never chosen first and
then copied into every feature.
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
    
    Simulated career-track labels are derived from latent aptitudes plus
    independent noise. They are not observed outcomes or ground truth.
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
        
        # Create continuous latent aptitudes before any label. Degree alignment
        # contributes a modest shift rather than choosing a class directly.
        track_priors = DEGREE_CAREER_ALIGNMENT[degree]
        latent = {
            track: float(np.random.normal(0.0, 0.85) + (track_priors[track] - 0.20) * 1.5)
            for track in CAREER_TRACKS
        }
        
        # Base student academic caliber (latent ability)
        base_gpa = np.clip(np.random.normal(3.0, 0.42), 1.8, 3.9)
        
        # Subject grades are noisy manifestations of several latent aptitudes.
        g_prog = np.clip(np.random.normal(base_gpa + .16 * latent["Software Engineering"] + .10 * latent["Data Science / AI"], .42), 1.0, 4.0)
        g_math = np.clip(np.random.normal(base_gpa + .20 * latent["Data Science / AI"] + .05 * latent["Software Engineering"], .46), 1.0, 4.0)
        g_db = np.clip(np.random.normal(base_gpa + .12 * latent["Software Engineering"] + .12 * latent["Data Science / AI"], .42), 1.0, 4.0)
        g_net = np.clip(np.random.normal(base_gpa + .16 * latent["Cybersecurity"] + .14 * latent["Cloud / DevOps"], .42), 1.0, 4.0)
        g_sys = np.clip(np.random.normal(base_gpa + .16 * latent["Cloud / DevOps"] + .12 * latent["Cybersecurity"], .42), 1.0, 4.0)
        g_des = np.clip(np.random.normal(base_gpa + .22 * latent["UI/UX Design"], .42), 1.0, 4.0)
        
        # Multi-domain interests (1 to 5 scale).
        # Students have multi-faceted interests: primary track has higher mean,
        # but secondary related fields are also explored, avoiding target leakage.
        def sample_interest(track):
            return int(np.clip(np.round(np.random.normal(3.0 + .72 * latent[track], .85)), 1, 5))

        int_se = sample_interest("Software Engineering")
        int_ds = sample_interest("Data Science / AI")
        int_sec = sample_interest("Cybersecurity")
        int_cld = sample_interest("Cloud / DevOps")
        int_ui = sample_interest("UI/UX Design")
        
        # Technical skills (0 to 3 scale)
        def sample_skill(track, general=False):
            expected = .35 + .30 * (year - 1) + .55 * latent[track] + (.35 if general else 0.0)
            return int(np.clip(np.round(np.random.normal(expected, .70)), 0, 3))
        
        skills = {
            "skill_oop": sample_skill("Software Engineering", general=True),
            "skill_dsa": sample_skill("Software Engineering"),
            "skill_web_api": sample_skill("Software Engineering"),
            "skill_testing": sample_skill("Software Engineering"),
            "skill_git": sample_skill("Software Engineering", general=True),
            
            "skill_python_data": sample_skill("Data Science / AI"),
            "skill_ml": sample_skill("Data Science / AI"),
            "skill_visualization": sample_skill("Data Science / AI"),
            "skill_sql": sample_skill("Data Science / AI", general=True),
            "skill_statistics": sample_skill("Data Science / AI"),
            
            "skill_network_sec": sample_skill("Cybersecurity"),
            "skill_cryptography": sample_skill("Cybersecurity"),
            "skill_linux": sample_skill("Cybersecurity", general=True),
            "skill_vuln_assess": sample_skill("Cybersecurity"),
            "skill_secure_code": sample_skill("Cybersecurity"),
            
            "skill_docker": sample_skill("Cloud / DevOps"),
            "skill_cicd": sample_skill("Cloud / DevOps"),
            "skill_cloud_infra": sample_skill("Cloud / DevOps"),
            "skill_iac": sample_skill("Cloud / DevOps"),
            "skill_monitoring": sample_skill("Cloud / DevOps"),
            
            "skill_user_research": sample_skill("UI/UX Design"),
            "skill_figma": sample_skill("UI/UX Design"),
            "skill_design_principles": sample_skill("UI/UX Design"),
            "skill_frontend": sample_skill("UI/UX Design"),
            "skill_design_systems": sample_skill("UI/UX Design"),
        }

        # Simulated label uses the latent aptitudes plus independent outcome
        # noise. It is deliberately generated after the evidence and should
        # never be described as an observed career outcome.
        label_scores = {track: latent[track] + float(np.random.normal(0.0, .50)) for track in CAREER_TRACKS}
        true_track = max(label_scores, key=label_scores.get)
        
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
