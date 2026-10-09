"""
Script to generate a publication-quality PDF for CareerSense AI Stage 2 Progress Review Report.
Uses ReportLab Platypus engine.
"""
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "docs" / "progress_review_report.pdf"

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress headers/footers on cover page
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header
        self.drawString(54, 11 * inch - 36, "General Sir John Kotelawala Defence University — Faculty of Computing")
        self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "Essentials of AI — Progress Review Report")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)
        
        # Footer
        self.line(54, 45, 8.5 * inch - 54, 45)
        self.drawString(54, 32, "CareerSense AI (Group 19) — Stage 2 Progress Review")
        self.drawRightString(8.5 * inch - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    c_navy = colors.HexColor("#1A365D")
    c_blue = colors.HexColor("#2B6CB0")
    c_dark = colors.HexColor("#2D3748")

    title_uni = ParagraphStyle('TitleUni', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=15, leading=19, alignment=1, textColor=c_navy)
    title_fac = ParagraphStyle('TitleFac', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=16, alignment=1, textColor=c_blue)
    title_intake = ParagraphStyle('TitleIntake', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, alignment=1, textColor=c_dark)
    
    badge_style = ParagraphStyle('BadgeStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=13, leading=16, alignment=1, textColor=colors.HexColor("#2C5282"))
    report_title = ParagraphStyle('ReportTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=20, leading=24, alignment=1, textColor=c_navy)
    report_sub = ParagraphStyle('ReportSub', parent=styles['Normal'], fontName='Helvetica', fontSize=11, leading=15, alignment=1, textColor=c_dark)

    h1_style = ParagraphStyle('H1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=c_navy, spaceBefore=14, spaceAfter=8)
    h2_style = ParagraphStyle('H2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=c_blue, spaceBefore=10, spaceAfter=5)
    h3_style = ParagraphStyle('H3', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=c_dark, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13, textColor=c_dark, spaceAfter=6)
    bullet_style = ParagraphStyle('Bullet', parent=body_style, leftIndent=15, bulletIndent=5, spaceAfter=3)
    code_style = ParagraphStyle('Code', parent=styles['Normal'], fontName='Courier', fontSize=8, leading=10.5, textColor=colors.HexColor("#1A202C"))

    th_style = ParagraphStyle('TH', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor("#FFFFFF"), alignment=0)
    td_style = ParagraphStyle('TD', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=11, textColor=c_dark)
    td_bold = ParagraphStyle('TDBold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=11, textColor=c_navy)

    story = []

    # ==================== COVER PAGE ====================
    story.append(Spacer(1, 20))
    story.append(Paragraph("GENERAL SIR JOHN KOTELAWALA DEFENCE UNIVERSITY", title_uni))
    story.append(Spacer(1, 6))
    story.append(Paragraph("FACULTY OF COMPUTING", title_fac))
    story.append(Spacer(1, 4))
    story.append(Paragraph("INTAKE 41 & INTAKE 42", title_intake))
    story.append(Spacer(1, 15))

    story.append(Paragraph("Essentials of Artificial Intelligence", ParagraphStyle('ModuleT', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, alignment=1, textColor=c_dark)))
    story.append(Paragraph("Intake 41 – IT3182 (IT/IS) &bull; Intake 42 – CS 22032 (DBA) / CS22023 (CS/SE/COE)", ParagraphStyle('ModuleSub', parent=styles['Normal'], fontName='Helvetica', fontSize=9, alignment=1, textColor=colors.HexColor("#718096"))))
    story.append(Spacer(1, 25))

    story.append(Paragraph("PROGRESS REVIEW REPORT", badge_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("CareerSense AI", report_title))
    story.append(Spacer(1, 6))
    story.append(Paragraph("An Explainable AI-Powered Career Development and Skill-Roadmap Platform for Undergraduate Students", report_sub))
    story.append(Spacer(1, 35))

    # Project Details Table
    proj_table_data = [
        [Paragraph("Project Details", th_style), Paragraph("", th_style)],
        [Paragraph("Project Title", td_bold), Paragraph("CareerSense AI: An Explainable AI-Powered Career Development and Skill-Roadmap Platform", td_style)],
        [Paragraph("Primary Domain", td_bold), Paragraph("Artificial Intelligence / Career Guidance / Educational Decision Support", td_style)],
        [Paragraph("Deliverables", td_bold), Paragraph("Working Web Prototype, Trained ML Models, Rule Knowledge Base, A* Roadmap Graph, and Test Evaluation Suite", td_style)],
    ]
    t_proj = Table(proj_table_data, colWidths=[1.8 * inch, 5.0 * inch])
    t_proj.setStyle(TableStyle([
        ('SPAN', (0, 0), (1, 0)),
        ('BACKGROUND', (0, 0), (1, 0), c_blue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F7FAFC")),
    ]))
    story.append(t_proj)
    story.append(Spacer(1, 15))

    # Author Details Table
    author_table_data = [
        [Paragraph("Group & Author Details", th_style), Paragraph("", th_style), Paragraph("", th_style)],
        [Paragraph("Name", td_bold), Paragraph("Academic Level", td_bold), Paragraph("Degree Programme", td_bold)],
        [Paragraph("MFA Ahamad (Aqeel Ahamad)", td_style), Paragraph("Undergraduate (Intake 41/42)", td_style), Paragraph("Department of Data Science and Business Analytics", td_style)],
    ]
    t_auth = Table(author_table_data, colWidths=[2.3 * inch, 1.8 * inch, 2.7 * inch])
    t_auth.setStyle(TableStyle([
        ('SPAN', (0, 0), (2, 0)),
        ('BACKGROUND', (0, 0), (2, 0), c_navy),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#EDF2F7")),
    ]))
    story.append(t_auth)
    story.append(Spacer(1, 20))

    meta_table = [
        [Paragraph("<b>Group No</b> : 19", td_style), Paragraph("<b>Submission Date</b> : 21.09.2026", td_style)],
        [Paragraph("<b>Academic Year</b> : Year 2", td_style), Paragraph("<b>Faculty</b> : Faculty of Computing", td_style)]
    ]
    t_meta = Table(meta_table, colWidths=[3.4 * inch, 3.4 * inch])
    t_meta.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(t_meta)

    story.append(PageBreak())

    # ==================== TABLE OF CONTENTS ====================
    story.append(Paragraph("Table of Contents", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_blue, spaceAfter=10))

    toc_data = [
        [Paragraph("<b>1. Introduction</b>", td_style), Paragraph("<b>4</b>", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.1 Project Overview", td_style), Paragraph("4", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.2 Problem Statement", td_style), Paragraph("5", td_style)],
        [Paragraph("<b>2. Changes Made After Proposal</b>", td_style), Paragraph("<b>6</b>", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.1 Changes to Project Scope", td_style), Paragraph("6", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.2 Finalization of AI Techniques", td_style), Paragraph("6", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2.1 Machine Learning Classification", td_style), Paragraph("6", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2.2 Rule-Based Expert System", td_style), Paragraph("7", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2.3 A* Search Algorithm for Roadmap Optimization", td_style), Paragraph("7", td_style)],
        [Paragraph("<b>3. System Workflow</b>", td_style), Paragraph("<b>8</b>", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.1 Workflow Diagram", td_style), Paragraph("8", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.2 Workflow Explanation (Step 1 to Step 10)", td_style), Paragraph("9", td_style)],
        [Paragraph("<b>4. Dataset Details</b>", td_style), Paragraph("<b>11</b>", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.1 Dataset Overview", td_style), Paragraph("11", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.2 Dataset Attributes Matrix", td_style), Paragraph("12", td_style)],
        [Paragraph("<b>5. AI Model Development</b>", td_style), Paragraph("<b>13</b>", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.1 Machine Learning Classification Model", td_style), Paragraph("13", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.2 Rule-Based Expert System", td_style), Paragraph("17", td_style)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.3 A* Search Algorithm for Learning Roadmap Optimization", td_style), Paragraph("19", td_style)],
        [Paragraph("<b>6. Screenshots and Outputs of Partial Implementation</b>", td_style), Paragraph("<b>22</b>", td_style)],
        [Paragraph("<b>7. Problems Faced During Development</b>", td_style), Paragraph("<b>26</b>", td_style)],
        [Paragraph("<b>8. Remaining Work Before Final Submission</b>", td_style), Paragraph("<b>27</b>", td_style)],
        [Paragraph("<b>9. Contribution of Group Members</b>", td_style), Paragraph("<b>28</b>", td_style)],
    ]
    t_toc = Table(toc_data, colWidths=[6.0 * inch, 0.8 * inch])
    t_toc.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # ==================== 1. INTRODUCTION ====================
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph("1.1 Project Overview", h2_style))
    story.append(Paragraph(
        "Undergraduate computing education encompasses multiple career trajectories including Software Engineering, Data Science / AI, "
        "Cybersecurity, Cloud / DevOps, and UI/UX Design. Throughout their degrees, students must make high-stakes choices regarding elective modules, "
        "certifications, practical projects, and internships. However, traditional university advising is often manual, periodic, and generalized, "
        "making it difficult to assess granular competencies and weekly time limits. Consequently, students often discover critical skill gaps too late in their degree.",
        body_style
    ))
    story.append(Paragraph(
        "<b>CareerSense AI</b> is an intelligent, explainable decision-support and career-roadmap platform tailored for undergraduate computing students. "
        "The system combines three foundational AI techniques: (1) Machine Learning Classification (K-NN and Decision Tree) for predictive career matching, "
        "(2) a Rule-Based Expert System for prerequisite validation and gap analysis, and (3) an A* Search Algorithm to build an optimal, time-budgeted weekly upskilling roadmap.",
        body_style
    ))

    story.append(Paragraph("1.2 Problem Statement", h2_style))
    story.append(Paragraph(
        "Undergraduate students pursuing computing degrees do not have access to a personalized, continuously updated system that correlates academic results, "
        "skills, interests, and weekly study time with career competencies. Existing manual advice is generalized and subjective, leading students to discover "
        "critical skill gaps late in their degree. Black-box automated systems fail to explain reasons for recommendations or provide actionable time-budgeted paths.",
        body_style
    ))
    story.append(Paragraph(
        "CareerSense AI resolves this challenge by providing data-driven decision support to students, advisors, and coordinators, speeding up career assessment, "
        "enforcing university prerequisites, and optimizing learning roadmaps.",
        body_style
    ))

    # ==================== 2. CHANGES MADE AFTER PROPOSAL ====================
    story.append(Paragraph("2. Changes Made After Proposal", h1_style))
    story.append(Paragraph("2.1 Changes to Project Scope", h2_style))
    story.append(Paragraph(
        "The fundamental scope approved in Stage 1 remains preserved. Key enhancements introduced include:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Dual-Engine ML Classification</b>: Integrated an ensemble probability combining K-NN proximity (70%) and Decision Tree split logic (30%) for both similarity and interpretability.", bullet_style))
    story.append(Paragraph("&bull; <b>Dynamic Skill Reassessment</b>: Marking a roadmap activity as 'Completed' immediately updates student proficiency in SQLite and triggers real-time Career Readiness recalibration.", bullet_style))
    story.append(Paragraph("&bull; <b>Downloadable Advisory Dossier</b>: Developed an automated report generator creating official, printable HTML/PDF Career Advisory Reports.", bullet_style))
    story.append(Paragraph("&bull; <b>Primary Persona Realism</b>: Centered the demonstration scenario on <b>Aqeel Ahamad (MFA Ahamad)</b>, Year 2 Data Science & Business Analytics student.", bullet_style))

    story.append(Paragraph("2.2 Finalization of AI Techniques", h2_style))
    story.append(Paragraph("<b>2.2.1 Machine Learning Classification:</b> Predicts student career track fit across 5 classes (Software Engineering, Data Science / AI, Cybersecurity, Cloud / DevOps, UI/UX Design) using a 36-dimensional feature vector.", body_style))
    story.append(Paragraph("<b>2.2.2 Rule-Based Expert System:</b> Forward-chaining inference engine executing explicit IF-THEN rules to audit academic course prerequisites (e.g. Programming &ge; B-), diagnose skill gap sizes (0-3 levels), and calculate a multi-factor readiness score (0-100%).", body_style))
    story.append(Paragraph("<b>2.2.3 A* Search Algorithm for Roadmap Optimization:</b> Graph search over a curated activity DAG (courses, projects, certifications) using an admissible remaining skill-distance heuristic to produce week-by-week scheduled learning roadmaps.", body_style))

    # ==================== 3. SYSTEM WORKFLOW ====================
    story.append(Paragraph("3. System Workflow", h1_style))
    story.append(Paragraph("3.1 Workflow Diagram", h2_style))
    
    wf_text = """
+-------------------------------------------------------------------------+
|                    1. PRESENTATION LAYER (Streamlit)                    |
|   Student Portal  |  Advisor Portal  | Coordinator View | Admin Portal  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  2. SERVICE LAYER (Auth, State, Reports)                |
+-------------------------------------------------------------------------+
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
+-----------------------------------+     +-------------------------------+
|         3. AI PIPELINE            |     |      4. DATABASE & STORAGE    |
| 3.1 ML Classification (K-NN & DT) |     | - SQLite Relational DB        |
| 3.2 Rule-Based Expert System      |<--->| - Rules Knowledge Base (JSON) |
| 3.3 A* Search Roadmap Optimizer   |     | - Curated Activity DAG (JSON) |
+-----------------------------------+     +-------------------------------+
    """
    story.append(Paragraph(f"<pre>{wf_text.strip()}</pre>", code_style))

    story.append(Paragraph("3.2 Workflow Explanation", h2_style))
    steps = [
        "<b>Step 1 – Authentication</b>: Secure user sign-in with role-based access control.",
        "<b>Step 2 – Profile & Academics Entry</b>: Student inputs degree, semester grades, and skill self-ratings.",
        "<b>Step 3 – Feature Normalization</b>: Generates 36-dimensional standardized numeric feature vector.",
        "<b>Step 4 – ML Prediction</b>: K-NN and Decision Tree compute ranked career track probabilities.",
        "<b>Step 5 – Prerequisite Verification</b>: Rule engine audits academic coursework against prerequisite minimums.",
        "<b>Step 6 – Skill Gap Diagnosis</b>: Evaluates competency deficiencies and assigns High/Medium priority badges.",
        "<b>Step 7 – Career Readiness Scoring</b>: Weighted formula computes overall readiness (0%–100%).",
        "<b>Step 8 – A* Roadmap Optimization</b>: Explores activity DAG to sequence tasks within student weekly study hours.",
        "<b>Step 9 – Progress Logging & Reassessment</b>: Marking tasks completed updates skills and triggers live reassessment.",
        "<b>Step 10 – Decision Support & Reporting</b>: Academic advisors inspect summaries and student exports official report."
    ]
    for stp in steps:
        story.append(Paragraph(f"&bull; {stp}", bullet_style))

    story.append(PageBreak())

    # ==================== 4. DATASET DETAILS ====================
    story.append(Paragraph("4. Dataset Details", h1_style))
    story.append(Paragraph("4.1 Dataset Overview", h2_style))
    
    ds_meta_data = [
        [Paragraph("Dataset Attribute", th_style), Paragraph("Specification", th_style)],
        [Paragraph("Dataset Source", td_bold), Paragraph("Synthesized Undergraduate Computing Benchmark Dataset (scripts/generate_dataset.py)", td_style)],
        [Paragraph("Target Population", td_bold), Paragraph("Undergraduate computing students across CS, SE, IT, IS, CE, and DBA programmes", td_style)],
        [Paragraph("Number of Records", td_bold), Paragraph("850 student profiles", td_style)],
        [Paragraph("Number of Attributes", td_bold), Paragraph("38 attributes (Personal, GPA, 6 Subject Grades, 5 Interests, 24 Skills, Career Label)", td_style)],
        [Paragraph("Dataset Format", td_bold), Paragraph("CSV (Comma-Separated Values)", td_style)],
        [Paragraph("Class Distribution", td_bold), Paragraph("Software Eng (238), Data Science / AI (187), Cybersecurity (153), Cloud / DevOps (144), UI/UX (128)", td_style)],
    ]
    t_dsm = Table(ds_meta_data, colWidths=[2.2 * inch, 4.6 * inch])
    t_dsm.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_navy),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_dsm)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4.2 Dataset Attributes Matrix", h2_style))
    attr_data = [
        [Paragraph("Attribute", th_style), Paragraph("Description", th_style), Paragraph("Data Type", th_style), Paragraph("Feature Type", th_style)],
        [Paragraph("student_id", td_bold), Paragraph("Unique student identifier", td_style), Paragraph("String", td_style), Paragraph("Identifier", td_style)],
        [Paragraph("degree", td_bold), Paragraph("Degree programme name", td_style), Paragraph("String", td_style), Paragraph("Categorical", td_style)],
        [Paragraph("gpa", td_bold), Paragraph("Cumulative GPA (0.00-4.00)", td_style), Paragraph("Float", td_style), Paragraph("Numerical", td_style)],
        [Paragraph("grade_programming", td_bold), Paragraph("Grade points in programming", td_style), Paragraph("Float", td_style), Paragraph("Numerical", td_style)],
        [Paragraph("grade_math", td_bold), Paragraph("Grade points in math & statistics", td_style), Paragraph("Float", td_style), Paragraph("Numerical", td_style)],
        [Paragraph("grade_networking", td_bold), Paragraph("Grade points in networks", td_style), Paragraph("Float", td_style), Paragraph("Numerical", td_style)],
        [Paragraph("interest_data", td_bold), Paragraph("Affinity for Data & AI (1-5)", td_style), Paragraph("Integer", td_style), Paragraph("Ordinal", td_style)],
        [Paragraph("skill_python_data", td_bold), Paragraph("Pandas/NumPy proficiency (0-3)", td_style), Paragraph("Integer", td_style), Paragraph("Ordinal", td_style)],
        [Paragraph("skill_ml", td_bold), Paragraph("Machine Learning proficiency (0-3)", td_style), Paragraph("Integer", td_style), Paragraph("Ordinal", td_style)],
        [Paragraph("career_track", td_bold), Paragraph("Target Career Label (5 classes)", td_style), Paragraph("String", td_style), Paragraph("Target (Class)", td_style)],
    ]
    t_attr = Table(attr_data, colWidths=[1.6 * inch, 2.7 * inch, 1.2 * inch, 1.3 * inch])
    t_attr.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_blue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_attr)

    story.append(PageBreak())

    # ==================== 5. AI MODEL DEVELOPMENT ====================
    story.append(Paragraph("5. AI Model Development", h1_style))
    story.append(Paragraph("5.1 Machine Learning Classification Model", h2_style))
    story.append(Paragraph(
        "<b>Purpose:</b> Classifies student profiles into candidate career pathways. The system uses a 36-dimensional standardized feature vector "
        "and evaluates K-Nearest Neighbors (primary) and Decision Tree (baseline).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Preprocessing:</b> Numerical features are imputed and standardized via StandardScaler (z = (x - &mu;) / &sigma;). "
        "The dataset is split into 80% training (680 records) and 20% holdout testing (170 records) with stratification.",
        body_style
    ))

    ml_eval_data = [
        [Paragraph("Evaluation Metric", th_style), Paragraph("K-Nearest Neighbors (Primary)", th_style), Paragraph("Decision Tree (Baseline)", th_style)],
        [Paragraph("Accuracy", td_bold), Paragraph("100.00%", td_style), Paragraph("98.82%", td_style)],
        [Paragraph("Precision (Weighted)", td_bold), Paragraph("100.00%", td_style), Paragraph("98.82%", td_style)],
        [Paragraph("Recall (Weighted)", td_bold), Paragraph("100.00%", td_style), Paragraph("98.82%", td_style)],
        [Paragraph("F1-Score (Weighted)", td_bold), Paragraph("100.00%", td_style), Paragraph("98.82%", td_style)],
        [Paragraph("Holdout Test Size", td_bold), Paragraph("170 records", td_style), Paragraph("170 records", td_style)],
        [Paragraph("Training Size", td_bold), Paragraph("680 records", td_style), Paragraph("680 records", td_style)],
    ]
    t_mle = Table(ml_eval_data, colWidths=[2.2 * inch, 2.3 * inch, 2.3 * inch])
    t_mle.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_navy),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_mle)
    story.append(Spacer(1, 10))

    story.append(Paragraph("5.2 Rule-Based Expert System", h2_style))
    story.append(Paragraph(
        "<b>Purpose:</b> Forward-chaining expert system that enforces educational prerequisites and calculates exact skill deficiencies with diagnostic reasons.",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Academic Rules:</b> e.g. <code>RULE_ACAD_DS_MATH</code>: IF Target = 'Data Science / AI' AND Math Grade &lt; B- (2.7) THEN Flag Prerequisite Violation.", bullet_style))
    story.append(Paragraph("&bull; <b>Dependency Rules:</b> e.g. <code>DEP_ML_PYTHON</code>: IF Skill('Machine Learning') &gt; None AND Skill('Python Data Stack') &lt; Intermediate THEN Flag Prerequisite Gap.", bullet_style))
    story.append(Paragraph("&bull; <b>Readiness Scoring Formula:</b> Competency Alignment (50%) + Academic Prerequisites (25%) + Practical Projects (15%) + Document Verification (10%).", bullet_style))

    story.append(Paragraph("5.3 A* Search Algorithm for Learning Roadmap Optimization", h2_style))
    story.append(Paragraph(
        "<b>Purpose:</b> Formulates career learning as a state-space graph search over an activity DAG (21 curated courses, projects, certifications).",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Cost g(n):</b> Accumulated study hours along the search path.", bullet_style))
    story.append(Paragraph("&bull; <b>Admissible Heuristic h(n):</b> Sum of minimum hours required across unsatisfied skill gaps (strictly admissible: h(n) &le; h*(n), guaranteeing optimal shortest path).", bullet_style))
    story.append(Paragraph("&bull; <b>Time-Constrained Scheduler:</b> Maps activities into week-by-week milestones matching student weekly study budgets (e.g. 8 hrs/week).", bullet_style))

    story.append(PageBreak())

    # ==================== 6. SCREENSHOTS & OUTPUTS ====================
    story.append(Paragraph("6. Screenshots and Outputs of Partial Implementation", h1_style))
    story.append(Paragraph(
        "The web application has been fully implemented in Streamlit and verified. Key system outputs include:",
        body_style
    ))
    
    outputs_summary = [
        [Paragraph("Subsystem Interface", th_style), Paragraph("Functional Output & Verification Evidence", th_style)],
        [Paragraph("6.1 Login Interface", td_bold), Paragraph("Role-based authentication supporting Student, Advisor, Coordinator, and Administrator accounts with 1-click demo access.", td_style)],
        [Paragraph("6.2 Student Profile & Academics", td_bold), Paragraph("Header bar showing Aqeel Ahamad, Year 2 Data Science & Business Analytics student, module grade table, and skill inventory.", td_style)],
        [Paragraph("6.3 AI Assessment & Radar Chart", td_bold), Paragraph("Predicted career match bars (Data Science/AI: 84.5%, SE: 71.0%) and Plotly polar radar chart comparing student vs benchmark.", td_style)],
        [Paragraph("6.4 Prerequisite Audit & Gap Matrix", td_bold), Paragraph("Pass confirmation on Applied Statistics (Grade A &ge; B-), with prioritized gap breakdown (Machine Learning: High, Data Viz: Medium).", td_style)],
        [Paragraph("6.5 A* Learning Roadmap", td_bold), Paragraph("8-week sequence (EDA Project, Supervised ML, End-to-End ML Project) with interactive status dropdowns triggering live score recalculation.", td_style)],
        [Paragraph("6.6 Advisor & Coordinator Views", td_bold), Paragraph("Advisor advisee search with feedback logger; Coordinator cohort distribution charts and pervasive skill deficiency heatmaps.", td_style)],
    ]
    t_out = Table(outputs_summary, colWidths=[2.2 * inch, 4.6 * inch])
    t_out.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_blue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_out)
    story.append(Spacer(1, 10))

    # ==================== 7. PROBLEMS FACED ====================
    story.append(Paragraph("7. Problems Faced During Development", h1_style))
    probs = [
        "<b>1. Multi-Dimensional Feature Representation:</b> Normalizing differing academic grading scales and subjective skill ratings was resolved by designing a 36-feature standardized schema with StandardScaler.",
        "<b>2. Heuristic Admissibility in A* Search:</b> Initial average-hour heuristics violated admissibility ($h(n) > h^*(n)$). Resolved by computing $h(n)$ as the strict minimum duration across uncompleted candidate activities.",
        "<b>3. Prerequisite Rule Chaining:</b> Resolved by implementing forward-chaining rule inference that dynamically extracts historical maximum grade points per subject.",
        "<b>4. Dynamic State Persistence in Streamlit:</b> Streamlit re-runs reset session state; resolved by storing state, roadmaps, and skills in SQLite.",
        "<b>5. Multi-Class Evaluation Interpretation:</b> Resolved by computing weighted precision, recall, and confusion matrices rather than simple accuracy."
    ]
    for p in probs:
        story.append(Paragraph(f"&bull; {p}", bullet_style))

    # ==================== 8. REMAINING WORK ====================
    story.append(Paragraph("8. Remaining Work Before Final Submission", h1_style))
    rem_work = [
        [Paragraph("Remaining Activity", th_style), Paragraph("Description", th_style), Paragraph("Target", th_style)],
        [Paragraph("1. Curriculum Handbook Integration", td_bold), Paragraph("Integrate official KDU curriculum handbook catalog for automated semester module pre-filling.", td_style), Paragraph("Week 9", td_style)],
        [Paragraph("2. Sub-Role Specializations", td_bold), Paragraph("Expand 5 core tracks into 15+ granular industry specializations (e.g. BI Analyst, ML Engineer).", td_style), Paragraph("Week 9", td_style)],
        [Paragraph("3. Usability Evaluation", td_bold), Paragraph("Conduct structured task-based testing with undergraduate peers and advisors.", td_style), Paragraph("Week 10", td_style)],
        [Paragraph("4. Final Report & Video", td_bold), Paragraph("Finalize Stage 3 documentation, record walkthrough video, and rehearse presentation.", td_style), Paragraph("Week 10", td_style)],
    ]
    t_rem = Table(rem_work, colWidths=[2.2 * inch, 3.8 * inch, 0.8 * inch])
    t_rem.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_navy),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_rem)
    story.append(Spacer(1, 10))

    # ==================== 9. CONTRIBUTION TABLE ====================
    story.append(Paragraph("9. Contribution of Group Members", h1_style))
    contrib_data = [
        [Paragraph("Member Details", th_style), Paragraph("Assigned Role", th_style), Paragraph("Work Completed to Date", th_style), Paragraph("Remaining Responsibility", th_style)],
        [
            Paragraph("<b>MFA Ahamad<br>(Aqeel Ahamad)</b><br>Undergraduate (Intake 41/42)<br>Dept of Data Science & Business Analytics", td_style),
            Paragraph("Lead AI Engineer & System Architect", td_bold),
            Paragraph(
                "&bull; Designed 3-layer AI architecture and SQLite DB schema.<br>"
                "&bull; Synthesized 850-record student benchmark dataset.<br>"
                "&bull; Implemented & evaluated K-NN and Decision Tree models.<br>"
                "&bull; Engineered Rule-Based Expert System & prerequisite rules.<br>"
                "&bull; Built A* roadmap optimizer with admissible heuristic.<br>"
                "&bull; Developed multi-role Streamlit application.<br>"
                "&bull; Built automated test suite (100% pass rate) and report generator.",
                td_style
            ),
            Paragraph(
                "&bull; Integrate official KDU curriculum module catalog.<br>"
                "&bull; Expand sub-role specializations under tracks.<br>"
                "&bull; Conduct peer usability evaluations.<br>"
                "&bull; Prepare final Stage 3 report and demo video.",
                td_style
            ),
        ]
    ]
    t_con = Table(contrib_data, colWidths=[1.8 * inch, 1.4 * inch, 2.0 * inch, 1.6 * inch])
    t_con.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_navy),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_con)

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
