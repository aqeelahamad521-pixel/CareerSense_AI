"""
Database Management Module for CareerSense AI.
Uses SQLite to store authentication, profiles, academic records, skills,
projects, documents metadata, roadmaps, and advisor notes.
"""
import sqlite3
import hashlib
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import DATABASE_PATH, GRADE_POINTS, SKILL_LEVELS, SUBJECT_AREAS, CORE_SKILLS, INTEREST_CATEGORIES

class DatabaseManager:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = str(db_path)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def init_db(self):
        """Initializes all database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Users Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('student', 'advisor', 'coordinator', 'admin')),
                full_name TEXT NOT NULL,
                reg_no TEXT,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 2. Student Profiles Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_profiles (
                user_id INTEGER PRIMARY KEY,
                degree TEXT NOT NULL,
                year INTEGER DEFAULT 1,
                gpa REAL DEFAULT 0.0,
                target_career TEXT DEFAULT 'Software Engineering',
                weekly_hours INTEGER DEFAULT 8,
                cv_filename TEXT,
                cv_extracted_skills TEXT,
                last_readiness_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """)

            # 3. Academic Records Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS academic_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                subject_area TEXT NOT NULL,
                module_code TEXT,
                module_name TEXT NOT NULL,
                grade TEXT NOT NULL,
                grade_points REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """)

            # 4. Student Skills Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                skill_name TEXT NOT NULL,
                proficiency_level TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, skill_name),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """)

            # 5. Student Interests Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_interests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                interest_category TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                UNIQUE (user_id, interest_category),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """)

            # 6. Student Projects Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                technologies TEXT,
                github_url TEXT,
                demonstrated_skills TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """)

            # 7. Student Documents Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                doc_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                extracted_metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """)

            # 8. Roadmaps Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS roadmaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                target_career TEXT NOT NULL,
                activity_id TEXT NOT NULL,
                activity_title TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                week_number INTEGER NOT NULL,
                estimated_hours INTEGER NOT NULL,
                main_skill TEXT NOT NULL,
                status TEXT DEFAULT 'Planned' CHECK (status IN ('Planned', 'In-Progress', 'Completed')),
                completion_date TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """)

            # 9. Advisor Notes Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS advisor_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                advisor_name TEXT NOT NULL,
                feedback TEXT NOT NULL,
                action_items TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """)
            conn.commit()

    # --- Authentication Methods ---
    def create_user(self, username, password, role, full_name, reg_no=None, email=None):
        pwd_hash = self.hash_password(password)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO users (username, password_hash, role, full_name, reg_no, email)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (username, pwd_hash, role, full_name, reg_no, email))
            user_id = cursor.lastrowid
            conn.commit()
            return user_id

    def authenticate_user(self, username, password):
        pwd_hash = self.hash_password(password)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, pwd_hash))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_user_by_id(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- Profile Methods ---
    def get_student_profile(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                profile = dict(row)
                if profile.get("cv_extracted_skills"):
                    try:
                        profile["cv_extracted_skills"] = json.loads(profile["cv_extracted_skills"])
                    except Exception:
                        profile["cv_extracted_skills"] = []
                return profile
            return None

    def save_student_profile(self, user_id, degree, year, gpa, target_career, weekly_hours, cv_filename=None, cv_extracted_skills=None, readiness_score=None):
        skills_json = json.dumps(cv_extracted_skills) if cv_extracted_skills else None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO student_profiles (user_id, degree, year, gpa, target_career, weekly_hours, cv_filename, cv_extracted_skills, last_readiness_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, 0.0), CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                degree=excluded.degree,
                year=excluded.year,
                gpa=excluded.gpa,
                target_career=excluded.target_career,
                weekly_hours=excluded.weekly_hours,
                cv_filename=COALESCE(excluded.cv_filename, student_profiles.cv_filename),
                cv_extracted_skills=COALESCE(excluded.cv_extracted_skills, student_profiles.cv_extracted_skills),
                last_readiness_score=COALESCE(excluded.last_readiness_score, student_profiles.last_readiness_score),
                updated_at=CURRENT_TIMESTAMP
            """, (user_id, degree, year, gpa, target_career, weekly_hours, cv_filename, skills_json, readiness_score))
            conn.commit()

    # --- Academic Records ---
    def get_academic_records(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM academic_records WHERE user_id = ?", (user_id,))
            return [dict(r) for r in cursor.fetchall()]

    def set_academic_records(self, user_id, records_list):
        """Replaces or sets module records for student."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM academic_records WHERE user_id = ?", (user_id,))
            for rec in records_list:
                pts = GRADE_POINTS.get(rec.get("grade", "C"), 2.0)
                cursor.execute("""
                INSERT INTO academic_records (user_id, subject_area, module_code, module_name, grade, grade_points)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, rec["subject_area"], rec.get("module_code", ""), rec["module_name"], rec["grade"], pts))
            
            # Recalculate GPA
            cursor.execute("SELECT AVG(grade_points) as avg_gpa FROM academic_records WHERE user_id = ?", (user_id,))
            avg_gpa = cursor.fetchone()["avg_gpa"] or 0.0
            cursor.execute("UPDATE student_profiles SET gpa = ? WHERE user_id = ?", (round(avg_gpa, 2), user_id))
            conn.commit()

    # --- Skills & Interests ---
    def get_student_skills(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT skill_name, proficiency_level FROM student_skills WHERE user_id = ?", (user_id,))
            return {r["skill_name"]: r["proficiency_level"] for r in cursor.fetchall()}

    def set_student_skill(self, user_id, skill_name, level):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO student_skills (user_id, skill_name, proficiency_level)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, skill_name) DO UPDATE SET
                proficiency_level=excluded.proficiency_level,
                last_updated=CURRENT_TIMESTAMP
            """, (user_id, skill_name, level))
            conn.commit()

    def get_student_interests(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT interest_category, rating FROM student_interests WHERE user_id = ?", (user_id,))
            return {r["interest_category"]: r["rating"] for r in cursor.fetchall()}

    def set_student_interests(self, user_id, interests_dict):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for cat, rating in interests_dict.items():
                cursor.execute("""
                INSERT INTO student_interests (user_id, interest_category, rating)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, interest_category) DO UPDATE SET
                    rating=excluded.rating
                """, (user_id, cat, rating))
            conn.commit()

    # --- Projects & Documents ---
    def get_student_projects(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student_projects WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
            return [dict(r) for r in cursor.fetchall()]

    def add_student_project(self, user_id, title, description, technologies, github_url, demonstrated_skills):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO student_projects (user_id, title, description, technologies, github_url, demonstrated_skills)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, title, description, technologies, github_url, demonstrated_skills))
            conn.commit()

    def get_student_documents(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student_documents WHERE user_id = ? ORDER BY upload_date DESC", (user_id,))
            return [dict(r) for r in cursor.fetchall()]

    def add_student_document(self, user_id, doc_type, filename, extracted_metadata=None):
        meta_str = json.dumps(extracted_metadata) if extracted_metadata else None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO student_documents (user_id, doc_type, filename, extracted_metadata)
            VALUES (?, ?, ?, ?)
            """, (user_id, doc_type, filename, meta_str))
            conn.commit()

    # --- Roadmaps & Progress Tracking ---
    def save_roadmap(self, user_id, target_career, activities_list):
        """Saves generated roadmap activities."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM roadmaps WHERE user_id = ? AND target_career = ?", (user_id, target_career))
            for item in activities_list:
                cursor.execute("""
                INSERT INTO roadmaps (user_id, target_career, activity_id, activity_title, activity_type, week_number, estimated_hours, main_skill, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, target_career, item["activity_id"], item["activity_title"], item["activity_type"],
                      item["week_number"], item["estimated_hours"], item["main_skill"], item.get("status", "Planned")))
            conn.commit()

    def get_roadmap(self, user_id, target_career=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if target_career:
                cursor.execute("SELECT * FROM roadmaps WHERE user_id = ? AND target_career = ? ORDER BY week_number ASC, id ASC", (user_id, target_career))
            else:
                cursor.execute("SELECT * FROM roadmaps WHERE user_id = ? ORDER BY week_number ASC, id ASC", (user_id,))
            return [dict(r) for r in cursor.fetchall()]

    def update_roadmap_activity_status(self, roadmap_id, new_status):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE roadmaps
            SET status = ?, completion_date = CASE WHEN ? = 'Completed' THEN CURRENT_TIMESTAMP ELSE NULL END
            WHERE id = ?
            """, (new_status, new_status, roadmap_id))
            conn.commit()

    # --- Advisor Notes ---
    def add_advisor_note(self, student_id, advisor_name, feedback, action_items=""):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO advisor_notes (student_id, advisor_name, feedback, action_items)
            VALUES (?, ?, ?, ?)
            """, (student_id, advisor_name, feedback, action_items))
            conn.commit()

    def get_advisor_notes(self, student_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM advisor_notes WHERE student_id = ? ORDER BY created_at DESC", (student_id,))
            return [dict(r) for r in cursor.fetchall()]

    # --- Analytics for Coordinator ---
    def get_all_students_summary(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT u.id, u.full_name, u.reg_no, p.degree, p.year, p.gpa, p.target_career, p.last_readiness_score, p.weekly_hours
            FROM users u
            JOIN student_profiles p ON u.id = p.user_id
            WHERE u.role = 'student'
            """)
            return [dict(r) for r in cursor.fetchall()]

    def get_all_student_skills_summary(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT skill_name, proficiency_level, COUNT(*) as count
            FROM student_skills
            GROUP BY skill_name, proficiency_level
            """)
            return [dict(r) for r in cursor.fetchall()]
