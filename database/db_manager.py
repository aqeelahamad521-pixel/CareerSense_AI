"""
Database Management Module for CareerSense AI.
Uses SQLite to store authentication, profiles, academic records, skills,
projects, documents metadata, roadmaps, and advisor notes.
"""
import sqlite3
import hashlib
import hmac
import secrets
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import DATABASE_PATH, GRADE_POINTS, SKILL_LEVELS, SUBJECT_AREAS, CORE_SKILLS, INTEREST_CATEGORIES

class DatabaseManager:
    PBKDF2_ALGORITHM = "sha256"
    PBKDF2_ITERATIONS = 100_000
    SALT_BYTES = 16

    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = str(db_path)
        self._active_connections = []
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        if not hasattr(self, "_active_connections"):
            self._active_connections = []
        self._active_connections.append(conn)
        return conn

    def close(self):
        """Explicitly closes all active database connections opened by this manager."""
        if hasattr(self, "_active_connections"):
            while self._active_connections:
                conn = self._active_connections.pop()
                try:
                    conn.close()
                except Exception:
                    pass

    @staticmethod
    def _ensure_column(cursor, table: str, column: str, definition: str):
        existing = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @classmethod
    def hash_password(cls, password: str, iterations: Optional[int] = None) -> str:
        """
        Cryptographically hashes password using PBKDF2-HMAC-SHA256 with a unique random salt.
        Format: pbkdf2:sha256:<iterations>$<salt_hex>$<hash_hex>
        """
        if iterations is None:
            iterations = cls.PBKDF2_ITERATIONS
        salt = secrets.token_hex(cls.SALT_BYTES)
        salt_bytes = bytes.fromhex(salt)
        key = hashlib.pbkdf2_hmac(cls.PBKDF2_ALGORITHM, password.encode("utf-8"), salt_bytes, iterations)
        return f"pbkdf2:{cls.PBKDF2_ALGORITHM}:{iterations}${salt}${key.hex()}"

    @classmethod
    def verify_password(cls, password: str, stored_hash: str) -> Tuple[bool, bool]:
        """
        Verifies password against stored_hash in constant time using hmac.compare_digest.
        Returns:
            (is_valid: bool, needs_rehash: bool)

        Handles:
        1. Modern salted format: pbkdf2:<algo>:<iterations>$<salt_hex>$<hash_hex>
        2. Legacy unsalted SHA-256 (64-char hex string) with needs_rehash=True for transparent migration.
        """
        if not stored_hash or not password or not isinstance(stored_hash, str) or not isinstance(password, str):
            return False, False

        # Modern format: pbkdf2:<algo>:<iterations>$<salt_hex>$<hash_hex>
        if stored_hash.startswith("pbkdf2:"):
            try:
                parts = stored_hash.split("$")
                if len(parts) == 3:
                    header, salt_hex, key_hex = parts
                    header_parts = header.split(":")
                    if len(header_parts) == 3 and header_parts[0] == "pbkdf2":
                        algo = header_parts[1].lower()
                        # Strictly whitelist approved secure hash algorithms
                        if algo not in ("sha256", "sha512"):
                            return False, False
                        iterations = int(header_parts[2])
                        # Bounds check iterations: min 10k, max 2M to prevent DoS or weak configurations
                        if iterations < 10_000 or iterations > 2_000_000:
                            return False, False
                        # Validate salt and key hex format and lengths
                        if len(salt_hex) < 16 or len(salt_hex) > 128:
                            return False, False
                        salt_bytes = bytes.fromhex(salt_hex)
                        if len(key_hex) not in (64, 128):
                            return False, False
                        computed_key = hashlib.pbkdf2_hmac(algo, password.encode("utf-8"), salt_bytes, iterations).hex()
                        if hmac.compare_digest(computed_key.lower(), key_hex.lower()):
                            needs_rehash = (iterations < cls.PBKDF2_ITERATIONS or algo != cls.PBKDF2_ALGORITHM)
                            return True, needs_rehash
            except Exception:
                return False, False
            return False, False

        # Legacy unsalted SHA-256 (64-character hexadecimal digest)
        if len(stored_hash) == 64 and all(c in "0123456789abcdefABCDEF" for c in stored_hash):
            legacy_computed = hashlib.sha256(password.encode("utf-8")).hexdigest()
            if hmac.compare_digest(legacy_computed.lower(), stored_hash.lower()):
                # Valid legacy credentials; flag for immediate transparent rehash
                return True, True

        return False, False

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
                evidence_status TEXT NOT NULL DEFAULT 'self_reported' CHECK (evidence_status IN ('self_reported', 'advisor_verified')),
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
                verification_status TEXT NOT NULL DEFAULT 'self_reported' CHECK (verification_status IN ('self_reported', 'advisor_verified')),
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
                verification_status TEXT NOT NULL DEFAULT 'submitted' CHECK (verification_status IN ('submitted', 'advisor_verified', 'rejected')),
                content_sha256 TEXT,
                mime_type TEXT,
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
                skill_level_gain TEXT,
                week_end INTEGER,
                description TEXT,
                resource_url TEXT,
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

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS advisor_assignments (
                advisor_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL UNIQUE,
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (advisor_id, student_id),
                FOREIGN KEY (advisor_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (student_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """)

            # Lightweight forward migrations for databases created by earlier
            # versions. SQLite cannot add CHECK constraints here, so every write
            # path validates these values as well.
            self._ensure_column(cursor, "student_skills", "evidence_status", "TEXT NOT NULL DEFAULT 'self_reported'")
            self._ensure_column(cursor, "student_projects", "verification_status", "TEXT NOT NULL DEFAULT 'self_reported'")
            self._ensure_column(cursor, "student_documents", "verification_status", "TEXT NOT NULL DEFAULT 'submitted'")
            self._ensure_column(cursor, "student_documents", "content_sha256", "TEXT")
            self._ensure_column(cursor, "student_documents", "mime_type", "TEXT")
            self._ensure_column(cursor, "roadmaps", "skill_level_gain", "TEXT")
            self._ensure_column(cursor, "roadmaps", "week_end", "INTEGER")
            self._ensure_column(cursor, "roadmaps", "description", "TEXT")
            self._ensure_column(cursor, "roadmaps", "resource_url", "TEXT")
            cursor.execute("""
                INSERT OR IGNORE INTO advisor_assignments (advisor_id, student_id)
                SELECT advisor.id, student.id
                FROM users AS advisor, users AS student
                WHERE advisor.username = 'advisor' AND advisor.role = 'advisor'
                  AND student.username = 'student_demo' AND student.role = 'student'
            """)
            conn.commit()

    # --- Authentication Methods ---
    def create_user(self, username, password, role, full_name, reg_no=None, email=None):
        if role not in {"student", "advisor", "coordinator", "admin"}:
            raise ValueError("Invalid user role")
        if not isinstance(password, str) or len(password) < 8:
            raise ValueError("Password must contain at least 8 characters")
        if not str(username).strip() or not str(full_name).strip():
            raise ValueError("Username and full name are required")
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
        if not username or not password:
            return None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            if not row:
                return None
            user = dict(row)
            stored_hash = user.get("password_hash", "")
            is_valid, needs_rehash = self.verify_password(password, stored_hash)
            if is_valid:
                if needs_rehash:
                    new_hash = self.hash_password(password)
                    cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user["id"]))
                    conn.commit()
                    user["password_hash"] = new_hash
                return user
            return None

    def get_user_by_id(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_username(self, username: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
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

    def get_student_skill_records(self, user_id):
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT skill_name, proficiency_level, evidence_status, last_updated FROM student_skills WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def set_student_skill(self, user_id, skill_name, level, evidence_status="self_reported"):
        if level not in SKILL_LEVELS:
            raise ValueError("Invalid skill level")
        if evidence_status not in {"self_reported", "advisor_verified"}:
            raise ValueError("Invalid skill evidence status")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO student_skills (user_id, skill_name, proficiency_level, evidence_status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, skill_name) DO UPDATE SET
                proficiency_level=excluded.proficiency_level,
                evidence_status=excluded.evidence_status,
                last_updated=CURRENT_TIMESTAMP
            """, (user_id, skill_name, level, evidence_status))
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

    def add_student_project(self, user_id, title, description, technologies, github_url, demonstrated_skills, verification_status="self_reported"):
        if verification_status not in {"self_reported", "advisor_verified"}:
            raise ValueError("Invalid project verification status")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO student_projects (user_id, title, description, technologies, github_url, demonstrated_skills, verification_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, title, description, technologies, github_url, demonstrated_skills, verification_status))
            conn.commit()

    def get_student_documents(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student_documents WHERE user_id = ? ORDER BY upload_date DESC", (user_id,))
            return [dict(r) for r in cursor.fetchall()]

    def add_student_document(self, user_id, doc_type, filename, extracted_metadata=None, content_sha256=None, mime_type=None, verification_status="submitted"):
        clean_filename = Path(filename).name
        if not clean_filename or clean_filename in (".", ".."):
            raise ValueError("Invalid document filename")
        valid_types = {"CV", "Transcript", "Certificate", "Other"}
        if doc_type not in valid_types:
            raise ValueError("Invalid document type")
        if verification_status not in {"submitted", "advisor_verified", "rejected"}:
            raise ValueError("Invalid document verification status")
        if content_sha256 and (len(content_sha256) != 64 or any(c not in "0123456789abcdef" for c in content_sha256.lower())):
            raise ValueError("Invalid SHA-256 digest")
        meta_str = json.dumps(extracted_metadata) if extracted_metadata else None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO student_documents (user_id, doc_type, filename, extracted_metadata, verification_status, content_sha256, mime_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, doc_type, clean_filename, meta_str, verification_status, content_sha256, mime_type))
            conn.commit()

    def _assert_advisor_access(self, actor_user_id, actor_role, student_id):
        if actor_role == "admin":
            return
        if actor_role != "advisor":
            raise PermissionError("Advisor or administrator access required")
        with self.get_connection() as conn:
            assignment = conn.execute(
                "SELECT 1 FROM advisor_assignments WHERE advisor_id = ? AND student_id = ?",
                (actor_user_id, student_id),
            ).fetchone()
        if not assignment:
            raise PermissionError("This student is not assigned to the current advisor")

    def assign_advisor(self, advisor_id, student_id):
        with self.get_connection() as conn:
            roles = conn.execute(
                "SELECT id, role FROM users WHERE id IN (?, ?)", (advisor_id, student_id)
            ).fetchall()
            role_by_id = {row["id"]: row["role"] for row in roles}
            if role_by_id.get(advisor_id) != "advisor" or role_by_id.get(student_id) != "student":
                raise ValueError("Assignments require an advisor and a student")
            conn.execute(
                "INSERT INTO advisor_assignments (advisor_id, student_id) VALUES (?, ?) "
                "ON CONFLICT(student_id) DO UPDATE SET advisor_id=excluded.advisor_id, assigned_at=CURRENT_TIMESTAMP",
                (advisor_id, student_id),
            )
            conn.commit()

    def verify_student_project(self, project_id, student_id, actor_user_id, actor_role, status="advisor_verified"):
        self._assert_advisor_access(actor_user_id, actor_role, student_id)
        if status not in {"self_reported", "advisor_verified"}:
            raise ValueError("Invalid project verification status")
        with self.get_connection() as conn:
            cur = conn.execute(
                "UPDATE student_projects SET verification_status = ? WHERE id = ? AND user_id = ?",
                (status, project_id, student_id),
            )
            conn.commit()
            return cur.rowcount == 1

    def verify_student_document(self, document_id, student_id, actor_user_id, actor_role, status="advisor_verified"):
        self._assert_advisor_access(actor_user_id, actor_role, student_id)
        if status not in {"submitted", "advisor_verified", "rejected"}:
            raise ValueError("Invalid document verification status")
        with self.get_connection() as conn:
            cur = conn.execute(
                "UPDATE student_documents SET verification_status = ? WHERE id = ? AND user_id = ?",
                (status, document_id, student_id),
            )
            conn.commit()
            return cur.rowcount == 1

    def verify_student_skill(self, student_id, skill_name, actor_user_id, actor_role, status="advisor_verified"):
        self._assert_advisor_access(actor_user_id, actor_role, student_id)
        if status not in {"self_reported", "advisor_verified"}:
            raise ValueError("Invalid skill evidence status")
        with self.get_connection() as conn:
            cur = conn.execute(
                "UPDATE student_skills SET evidence_status = ?, last_updated = CURRENT_TIMESTAMP "
                "WHERE user_id = ? AND skill_name = ?",
                (status, student_id, skill_name),
            )
            conn.commit()
            return cur.rowcount == 1

    # --- Roadmaps & Progress Tracking ---
    def save_roadmap(self, user_id, target_career, activities_list):
        """Saves generated roadmap activities."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM roadmaps WHERE user_id = ? AND target_career = ?", (user_id, target_career))
            for item in activities_list:
                cursor.execute("""
                INSERT INTO roadmaps (user_id, target_career, activity_id, activity_title, activity_type, week_number, week_end, estimated_hours, main_skill, skill_level_gain, description, resource_url, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, target_career, item["activity_id"], item["activity_title"], item["activity_type"],
                      item["week_number"], item.get("week_end", item["week_number"]), item["estimated_hours"], item["main_skill"],
                      item.get("skill_level_gain"), item.get("description", ""), item.get("resource_url", ""), item.get("status", "Planned")))
            conn.commit()

    def get_roadmap(self, user_id, target_career=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if target_career:
                cursor.execute("SELECT * FROM roadmaps WHERE user_id = ? AND target_career = ? ORDER BY week_number ASC, id ASC", (user_id, target_career))
            else:
                cursor.execute("SELECT * FROM roadmaps WHERE user_id = ? ORDER BY week_number ASC, id ASC", (user_id,))
            return [dict(r) for r in cursor.fetchall()]

    def update_roadmap_activity_status(self, roadmap_id, user_id, new_status):
        if new_status == "In Progress":
            new_status = "In-Progress"
        valid_statuses = {"Planned", "In-Progress", "Completed"}
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid activity status '{new_status}'. Allowed: {valid_statuses}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE roadmaps
            SET status = ?, completion_date = CASE WHEN ? = 'Completed' THEN CURRENT_TIMESTAMP ELSE NULL END
            WHERE id = ? AND user_id = ?
            """, (new_status, new_status, roadmap_id, user_id))
            conn.commit()
            return cursor.rowcount == 1

    # --- Advisor Notes ---
    def add_advisor_note(self, student_id, advisor_name, feedback, action_items="", actor_user_id=None, actor_role=None):
        self._assert_advisor_access(actor_user_id, actor_role, student_id)
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
    def get_students_for_advisor(self, advisor_user_id, actor_role="advisor"):
        if actor_role == "admin":
            return self.get_all_students_summary("admin", include_identifiers=True)
        if actor_role != "advisor":
            raise PermissionError("Advisor access required")
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT u.id, u.full_name, u.reg_no, p.degree, p.year, p.gpa, p.target_career,
                       p.last_readiness_score, p.weekly_hours
                FROM advisor_assignments a
                JOIN users u ON u.id = a.student_id
                JOIN student_profiles p ON p.user_id = u.id
                WHERE a.advisor_id = ?
                ORDER BY u.full_name
            """, (advisor_user_id,)).fetchall()
            return [dict(row) for row in rows]

    def get_all_students_summary(self, actor_role, include_identifiers=False):
        if actor_role not in {"coordinator", "admin"}:
            raise PermissionError("Coordinator or administrator access required")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT u.id, u.full_name, u.reg_no, p.degree, p.year, p.gpa, p.target_career, p.last_readiness_score, p.weekly_hours
            FROM users u
            JOIN student_profiles p ON u.id = p.user_id
            WHERE u.role = 'student'
            """)
            results = [dict(r) for r in cursor.fetchall()]
            if actor_role == "coordinator" or not include_identifiers:
                for index, row in enumerate(results, start=1):
                    row.pop("full_name", None)
                    row.pop("reg_no", None)
                    row.pop("id", None)
                    row["student_key"] = f"Cohort member {index}"
            return results

    def get_all_student_skills_summary(self, actor_role):
        if actor_role not in {"coordinator", "admin"}:
            raise PermissionError("Coordinator or administrator access required")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT skill_name, proficiency_level, COUNT(*) as count
            FROM student_skills
            GROUP BY skill_name, proficiency_level
            """)
            return [dict(r) for r in cursor.fetchall()]
