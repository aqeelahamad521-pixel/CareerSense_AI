"""
Personalized Career Development Report Generator for CareerSense AI.
Generates comprehensive, beautifully formatted HTML and text reports
summarizing student profile, AI career match probabilities, rule-based
gap analysis, readiness breakdown, and weekly A* learning roadmap.
"""
from datetime import datetime

class CareerReportGenerator:
    @staticmethod
    def generate_html_report(student_profile: dict, user_info: dict, explanation_data: dict, readiness_data: dict, roadmap_data: list) -> str:
        user_info = user_info or {}
        student_profile = student_profile or {}
        explanation_data = explanation_data or {}
        readiness_data = readiness_data or {}
        roadmap_data = roadmap_data or []

        name = user_info.get("full_name") or "Aqeel Ahamad (Demo Student)"
        raw_reg = user_info.get("reg_no")
        reg_no = str(raw_reg) if raw_reg else "STU-DEMO-01"
        ref_code = reg_no.replace("/", "").replace(" ", "").replace("-", "")
        degree = student_profile.get("degree") or "BSc (Hons) in Data Science & Business Analytics"
        year = student_profile.get("year", 2)
        try:
            gpa = float(student_profile.get("gpa", 3.40))
        except (ValueError, TypeError):
            gpa = 3.40
        target_track = explanation_data.get("target_track", "Data Science / AI")
        date_str = datetime.now().strftime("%B %d, %Y")
        
        readiness_score = readiness_data.get("total_readiness_score", 0.0)
        tier = readiness_data.get("readiness_tier", "Foundational")
        breakdown = readiness_data.get("breakdown", {})

        # Build gaps table rows
        gaps_html = ""
        for g in explanation_data.get("skill_gaps", []):
            p_color = "#e53e3e" if g["priority"] == "High" else ("#dd6b20" if g["priority"] == "Medium" else "#38a169")
            gaps_html += f"""
            <tr>
                <td><strong>{g['skill']}</strong></td>
                <td><span style="background:#edf2f7; padding:2px 8px; border-radius:4px;">{g['current_level']}</span></td>
                <td><span style="background:#e2e8f0; padding:2px 8px; border-radius:4px; font-weight:bold;">{g['required_level']}</span></td>
                <td><span style="color:{p_color}; font-weight:bold;">{g['priority']}</span></td>
                <td>{g['reason']}</td>
            </tr>
            """

        # Build roadmap table rows
        roadmap_html = ""
        for act in roadmap_data:
            roadmap_html += f"""
            <tr>
                <td style="text-align:center; font-weight:bold;">Week {act.get('week_number', 1)}</td>
                <td><strong>{act.get('activity_title', '')}</strong></td>
                <td><span style="background:#e6fffa; color:#234e52; padding:2px 6px; border-radius:4px;">{act.get('activity_type', '')}</span></td>
                <td>{act.get('estimated_hours', 0)} hrs</td>
                <td>{act.get('main_skill', '')}</td>
                <td><span style="font-weight:bold;">{act.get('status', 'Planned')}</span></td>
            </tr>
            """

        # Build ranked career matches
        matches_html = ""
        for m in explanation_data.get("ranked_matches", [])[:4]:
            matches_html += f"""
            <div style="margin-bottom: 8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                    <span><strong>{m['track']}</strong></span>
                    <span>{m['probability']}%</span>
                </div>
                <div style="background:#edf2f7; height:8px; border-radius:4px; overflow:hidden;">
                    <div style="background:#3182ce; height:8px; width:{m['probability']}%;"></div>
                </div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CareerSense AI - Career Advisory Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #2d3748; background: #fff; margin: 30px; }}
        .header {{ border-bottom: 3px solid #2b6cb0; padding-bottom: 15px; margin-bottom: 25px; }}
        .header h1 {{ margin: 0; color: #1a365d; font-size: 26px; }}
        .header p {{ margin: 4px 0 0 0; color: #718096; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; }}
        .card {{ background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; }}
        .card h3 {{ margin-top: 0; color: #2b6cb0; font-size: 16px; border-bottom: 1px solid #cbd5e0; padding-bottom: 6px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #edf2f7; color: #4a5568; }}
        .score-pill {{ display: inline-block; font-size: 24px; font-weight: bold; color: #2b6cb0; margin-bottom: 5px; }}
        .footer {{ margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 15px; font-size: 12px; color: #a0aec0; text-align: center; }}
        @media print {{
            body {{ margin: 10mm; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h1>CareerSense AI — Career Advisory Report</h1>
                <p>Faculty of Computing | General Sir John Kotelawala Defence University</p>
            </div>
            <div style="text-align:right;">
                <p style="margin:0; font-weight:bold;">Date: {date_str}</p>
                <p style="margin:0;">Ref: CS-AI-{ref_code}</p>
            </div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>Student Profile Summary</h3>
            <p><strong>Name:</strong> {name}</p>
            <p><strong>Registration No:</strong> {reg_no}</p>
            <p><strong>Programme:</strong> {degree}</p>
            <p><strong>Current Year:</strong> Year {year} &nbsp;|&nbsp; <strong>Cumulative GPA:</strong> {gpa:.2f}</p>
            <p><strong>Selected Target Career:</strong> <span style="color:#2b6cb0; font-weight:bold;">{target_track}</span></p>
        </div>

        <div class="card">
            <h3>AI Career Readiness Assessment</h3>
            <div class="score-pill">{readiness_score}%</div>
            <p style="margin:0 0 10px 0; font-weight:600; color:#4a5568;">Status: {tier}</p>
            <p style="font-size:12px; margin:2px 0;">Competency Alignment: {breakdown.get('competency_score', 0)} / 50 pts</p>
            <p style="font-size:12px; margin:2px 0;">Academic Prerequisites: {breakdown.get('academic_score', 0)} / 25 pts</p>
            <p style="font-size:12px; margin:2px 0;">Project Evidence: {breakdown.get('project_score', 0)} / 15 pts</p>
            <p style="font-size:12px; margin:2px 0;">Documentation & CV: {breakdown.get('document_score', 0)} / 10 pts</p>
        </div>
    </div>

    <div class="card" style="margin-bottom: 25px;">
        <h3>AI Career Track Affinity (K-NN & Decision Tree Classification)</h3>
        {matches_html}
    </div>

    <div class="card" style="margin-bottom: 25px;">
        <h3>Rule-Based Competency Gap Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>Competency / Skill</th>
                    <th>Current Level</th>
                    <th>Required Benchmark</th>
                    <th>Priority</th>
                    <th>Reason / Expert Rule</th>
                </tr>
            </thead>
            <tbody>
                {gaps_html}
            </tbody>
        </table>
    </div>

    <div class="card" style="margin-bottom: 25px;">
        <h3>Personalized A* Optimized Learning Roadmap</h3>
        <p style="font-size:12px; color:#718096; margin-top:-4px;">
            Constructed using state-space search exploring activity DAG, respecting student weekly availability.
        </p>
        <table>
            <thead>
                <tr>
                    <th style="text-align:center;">Timeline</th>
                    <th>Activity Title</th>
                    <th>Type</th>
                    <th>Effort</th>
                    <th>Target Skill</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {roadmap_html}
            </tbody>
        </table>
    </div>

    <div class="footer">
        <p>CareerSense AI Decision Support System &bull; Generated for Academic Advising & Career Development &bull; KDU Faculty of Computing</p>
    </div>
</body>
</html>
"""
        return html_content
