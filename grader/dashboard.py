"""
Dash dashboard for visualizing student grades.

Run with: python -m grader.dashboard --grades-dir ./grades
"""

import argparse
import sys
from pathlib import Path

from .grades_aggregator import load_grades_from_dir
from .models import GradeResult


def create_dashboard(grades: list[GradeResult], grades_dir: Path | None = None):
    """
    Create a Dash dashboard to visualize grades.

    Args:
        grades: List of GradeResult objects.
        grades_dir: Path to the grades directory.
    """
    import os
    from flask import send_from_directory
    try:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from dash import Dash, dash_table, dcc, html, ALL
        from dash.dependencies import Input, Output, State
        from .config import GRADE_OUTPUT_FILENAME
    except ImportError:
        print("Dashboard requires additional dependencies. Install with:")
        print("  uv pip install dash pandas plotly")
        sys.exit(1)

    # Use the provided grades_dir, or resolve default if None
    if grades_dir is None:
        grades_dir = Path("grades").resolve()

    # Convert grades to DataFrame
    data = []
    for grade in grades:
        row = {
            "Student": grade.student_id.replace("hm3-ecg-data-analysis-", ""),
            "Total Score": grade.total_score,
            "Max Score": grade.max_score,
            "Percentage": (grade.total_score / grade.max_score * 100) if grade.max_score > 0 else 0,
            "Tests Passed": "Yes" if grade.code_execution_passed else "No",
        }
        for section in grade.sections:
            row[section.section_name] = section.points_earned
        data.append(row)

    df = pd.DataFrame(data)

    # Get section columns
    section_cols = [s.section_name for s in grades[0].sections] if grades else []

    # Create Dash app
    # Note: Dash's mathjax=True currently works best with MathJax v2 (MathJax.Hub)
    external_scripts = [
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"
    ]
    app = Dash(__name__, suppress_callback_exceptions=True, external_scripts=external_scripts)
    
    # Add route to serve local files (images, reports)
    @app.server.route("/files/<path:path>")
    def serve_files(path):
        return send_from_directory(grades_dir, path)

    # Calculate statistics
    avg_score = df["Percentage"].mean() if not df.empty else 0
    max_score = df["Percentage"].max() if not df.empty else 0
    min_score = df["Percentage"].min() if not df.empty else 0
    passed_count = (df["Tests Passed"] == "Yes").sum() if not df.empty else 0

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Grader Pod Dashboard", style={"color": "#2c3e50", "marginBottom": "5px"}),
            html.P(f"Total Students: {len(grades)}", style={"color": "#7f8c8d", "fontSize": "14px"}),
        ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "#ecf0f1"}),

        # Statistics cards
        html.Div([
            html.Div([
                html.H3(f"{avg_score:.1f}%", style={"color": "#3498db", "margin": "0"}),
                html.P("Average Score", style={"color": "#7f8c8d", "margin": "0"}),
            ], style={"flex": "1", "textAlign": "center", "padding": "20px", "backgroundColor": "white", "borderRadius": "8px", "margin": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

            html.Div([
                html.H3(f"{max_score:.1f}%", style={"color": "#27ae60", "margin": "0"}),
                html.P("Highest Score", style={"color": "#7f8c8d", "margin": "0"}),
            ], style={"flex": "1", "textAlign": "center", "padding": "20px", "backgroundColor": "white", "borderRadius": "8px", "margin": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

            html.Div([
                html.H3(f"{min_score:.1f}%", style={"color": "#e74c3c", "margin": "0"}),
                html.P("Lowest Score", style={"color": "#7f8c8d", "margin": "0"}),
            ], style={"flex": "1", "textAlign": "center", "padding": "20px", "backgroundColor": "white", "borderRadius": "8px", "margin": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

            html.Div([
                html.H3(f"{passed_count}/{len(grades)}", style={"color": "#9b59b6", "margin": "0"}),
                html.P("Tests Passed", style={"color": "#7f8c8d", "margin": "0"}),
            ], style={"flex": "1", "textAlign": "center", "padding": "20px", "backgroundColor": "white", "borderRadius": "8px", "margin": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
        ], style={"display": "flex", "justifyContent": "center", "padding": "10px 20px"}),

        # Charts row
        html.Div([
            # Bar chart - scores by student
            html.Div([
                dcc.Graph(
                    id="scores-bar",
                    figure=px.bar(
                        df.sort_values("Percentage", ascending=False),
                        x="Student",
                        y="Percentage",
                        color="Tests Passed",
                        color_discrete_map={"Yes": "#27ae60", "No": "#e74c3c"},
                        title="Scores by Student",
                    ).update_layout(
                        xaxis_tickangle=-45,
                        showlegend=True,
                        plot_bgcolor="white",
                        yaxis_title="Score (%)",
                    )
                )
            ], style={"flex": "1", "padding": "10px"}),

            # Box plot - score distribution by section
            html.Div([
                dcc.Graph(
                    id="section-box",
                    figure=px.box(
                        df.melt(
                            id_vars=["Student"],
                            value_vars=section_cols,
                            var_name="Section",
                            value_name="Score"
                        ),
                        x="Section",
                        y="Score",
                        title="Score Distribution by Section",
                        color="Section",
                    ).update_layout(
                        xaxis_tickangle=-45,
                        showlegend=False,
                        plot_bgcolor="white",
                    )
                )
            ], style={"flex": "1", "padding": "10px"}),
        ], style={"display": "flex", "padding": "10px 20px"}),

        # Heatmap
        html.Div([
            dcc.Graph(
                id="heatmap",
                figure=px.imshow(
                    df[section_cols].values if section_cols else [],
                    x=section_cols,
                    y=df["Student"].tolist(),
                    color_continuous_scale="RdYlGn",
                    title="Section Scores Heatmap",
                    labels={"color": "Score"},
                ).update_layout(
                    height=max(400, len(grades) * 30),
                )
            )
        ], style={"padding": "10px 20px"}),

        # Grades table
        html.Div([
            html.H3("Detailed Grades", style={"color": "#2c3e50", "marginBottom": "10px"}),
            dash_table.DataTable(
                id="grades-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.round(2).to_dict("records"),
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "fontSize": "14px",
                },
                style_header={
                    "backgroundColor": "#3498db",
                    "color": "white",
                    "fontWeight": "bold",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{Tests Passed} = No"},
                        "backgroundColor": "#fadbd8",
                    },
                    {
                        "if": {"filter_query": "{Percentage} >= 90"},
                        "backgroundColor": "#d5f5e3",
                    },
                ],
            ),
        ], style={"padding": "20px", "backgroundColor": "white", "margin": "20px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

        # Student Feedback section
        html.Div([
            html.Div([
                html.H3("Student Feedback", style={"color": "#2c3e50", "margin": "0"}),
                html.Button(
                    "Submit All Grades to GitHub",
                    id="submit-grades-btn",
                    style={
                        "backgroundColor": "#2ecc71",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "fontWeight": "bold",
                        "marginLeft": "20px"
                    }
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
            dcc.Dropdown(
                id="student-dropdown",
                options=[{"label": g.student_id.replace("hm3-ecg-data-analysis-", ""), "value": g.student_id} for g in grades],
                value=grades[0].student_id if grades else None,
                style={"marginBottom": "10px"},
            ),
            html.Div(id="submit-status", style={"marginBottom": "10px"}),
            html.Div(id="feedback-content"),
        ], style={"padding": "20px", "backgroundColor": "white", "margin": "20px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
        
        # Student Report section
        html.Div(id="report-section", style={"padding": "20px", "backgroundColor": "white", "margin": "20px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})

    ], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f5f6fa", "minHeight": "100vh"})

    # Callback for feedback
    @app.callback(
        [Output("feedback-content", "children"),
         Output("report-section", "children")],
        Input("student-dropdown", "value")
    )
    def update_feedback(student_id: str):
        if not student_id:
            return html.P("Select a student to view feedback."), []

        grade = next((g for g in grades if g.student_id == student_id), None)
        if not grade:
            return html.P("Grade not found."), []

        # Render feedback
        feedback_content = html.Div([
            html.Div([
                html.H4(f"Overall Feedback: {grade.total_score:.1f}/{grade.max_score:.1f} ({grade.total_score/grade.max_score*100:.1f}%)" if grade.max_score > 0 else "N/A", style={"margin": "0"}),
                html.Button(
                    "Save Changes",
                    id="save-grades-btn",
                    style={
                        "backgroundColor": "#3498db",
                        "color": "white",
                        "border": "none",
                        "padding": "8px 15px",
                        "borderRadius": "5px",
                        "cursor": "pointer"
                    }
                ),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"}),
            
            dcc.Textarea(
                id="overall-feedback-input",
                value=grade.overall_feedback,
                style={"width": "100%", "height": "100px", "padding": "10px", "borderRadius": "5px", "border": "1px solid #ddd"}
            ),
            
            html.H5("Section Details:", style={"marginTop": "20px"}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Strong(f"{s.section_name}: "),
                        dcc.Input(
                            id={'type': 'section-points', 'index': i},
                            type='number',
                            value=s.points_earned,
                            max=s.max_points,
                            min=0,
                            step=0.1,
                            style={"width": "70px", "marginLeft": "5px", "padding": "5px"}
                        ),
                        html.Span(f" / {s.max_points}")
                    ], style={"marginBottom": "10px"}),
                    dcc.Textarea(
                        id={'type': 'section-feedback', 'index': i},
                        value=s.feedback,
                        style={"width": "100%", "height": "60px", "padding": "10px", "borderRadius": "5px", "border": "1px solid #ddd"}
                    ),
                ], style={"marginBottom": "15px", "padding": "15px", "backgroundColor": "#fcfcfc", "borderRadius": "8px", "border": "1px solid #eee"})
                for i, s in enumerate(grade.sections)
            ]),
            html.Div(id="save-status", style={"marginTop": "10px", "fontWeight": "bold"})
        ])
        
        # Render report if available
        # Find path using the passed grades_dir
        report_path = grades_dir / student_id / "report.md"
        report_content_div = []
        
        if report_path.exists():
            with open(report_path, "r") as f:
                md_text = f.read()
            
            # Simple but effective escaping for MathJax inside Markdown
            # We replace \\ with \\\\ to survive the Markdown -> HTML conversion
            # and ensure LaTeX environments like bmatrix work correctly.
            import re
            
            def escape_math(match):
                # Double the backslashes inside math blocks
                return match.group(0).replace("\\", "\\\\")

            # Escape display math: $$ ... $$
            md_text = re.sub(r'(\$\$.*?\$\$)', escape_math, md_text, flags=re.DOTALL)
            # Escape inline math: $ ... $ (avoiding double escaping if already done by display math)
            # This regex avoids matching display math blocks
            md_text = re.sub(r'(?<!\$)\$([^\$]+?)\$(?!\$)', escape_math, md_text)

            report_content_div = [
                html.H3("Student Report", style={"color": "#2c3e50", "marginBottom": "10px"}),
                dcc.Markdown(md_text, mathjax=True, style={"padding": "20px", "border": "1px solid #eee", "borderRadius": "5px"}),
            ]
        
        return feedback_content, report_content_div

    # Callback for submitting grades
    @app.callback(
        Output("submit-status", "children"),
        Input("submit-grades-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def submit_grades(n_clicks):
        if not n_clicks:
            return ""

        import subprocess
        results = []
        for grade in grades:
            if not grade.github_repo:
                results.append(html.P(f"⚠️ {grade.student_id}: No GitHub repository found.", style={"color": "#e67e22"}))
                continue

            try:
                # Create issue title and body
                title = f"Grade: {grade.total_score:.1f}/{grade.max_score:.1f}"
                body = f"## Overall Feedback\n\n{grade.overall_feedback}\n\n"
                body += "### Section Details\n\n"
                for section in grade.sections:
                    body += f"- **{section.section_name}**: {section.points_earned:.1f}/{section.max_points:.1f}\n"
                    body += f"  {section.feedback}\n\n"

                # Run gh issue create
                cmd = [
                    "gh", "issue", "create",
                    "--repo", grade.github_repo,
                    "--title", title,
                    "--body", body
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)

                if result.returncode == 0:
                    results.append(html.P(f"✅ {grade.student_id}: Grade submitted to {grade.github_repo}", style={"color": "#27ae60"}))
                else:
                    results.append(html.P(f"❌ {grade.student_id}: Failed to submit to {grade.github_repo}. Error: {result.stderr}", style={"color": "#e74c3c"}))
            except Exception as e:
                results.append(html.P(f"❌ {grade.student_id}: Unexpected error: {str(e)}", style={"color": "#e74c3c"}))

        return html.Div([
            html.H4("Submission Status:"),
            html.Div(results, style={"maxHeight": "200px", "overflowY": "auto", "padding": "10px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"})
        ])

    # Callback for saving changes
    @app.callback(
        Output("save-status", "children"),
        Input("save-grades-btn", "n_clicks"),
        [State("student-dropdown", "value"),
         State("overall-feedback-input", "value"),
         State({'type': 'section-points', 'index': ALL}, 'value'),
         State({'type': 'section-feedback', 'index': ALL}, 'value')],
        prevent_initial_call=True
    )
    def save_changes(n_clicks, student_id, overall_feedback, section_points, section_feedbacks):
        if not n_clicks:
            return ""

        grade = next((g for g in grades if g.student_id == student_id), None)
        if not grade:
            return html.Span("❌ Error: Grade not found", style={"color": "#e74c3c"})

        # Update in-memory object
        grade.overall_feedback = overall_feedback
        for i, s in enumerate(grade.sections):
            if i < len(section_points):
                s.points_earned = section_points[i]
            if i < len(section_feedbacks):
                s.feedback = section_feedbacks[i]
        
        # Recalculate total score
        grade.total_score = sum(s.points_earned for s in grade.sections)

        # Persistence
        try:
            # 1. Update individual JSON in grades_dir
            individual_path = grades_dir / f"{grade.student_id}.json"
            with open(individual_path, "w", encoding="utf-8") as f:
                f.write(grade.model_dump_json(indent=2))
            
            # 2. Update original grade.json if submission_path is available
            if grade.submission_path:
                original_path = Path(grade.submission_path) / GRADE_OUTPUT_FILENAME
                with open(original_path, "w", encoding="utf-8") as f:
                    f.write(grade.model_dump_json(indent=2))

            # 3. Update summary files
            from .grades_aggregator import GradesAggregator
            aggregator = GradesAggregator(output_dir=grades_dir)
            aggregator.grades = grades
            aggregator.save_all()

            return html.Span("✅ Changes saved to disk and updated in memory.", style={"color": "#27ae60"})
        except Exception as e:
            return html.Span(f"❌ Error saving: {str(e)}", style={"color": "#e74c3c"})

    return app


def main() -> int:
    """
    Main entry point for the dashboard.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Grader Pod Dashboard - Visualize student grades"
    )
    parser.add_argument(
        "--grades-dir",
        type=Path,
        default=Path("grades"),
        help="Path to the grades directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()

    if not args.grades_dir.exists():
        print(f"Error: Grades directory not found: {args.grades_dir}")
        print("Run the grader first to generate grades.")
        return 1

    grades = load_grades_from_dir(args.grades_dir)
    if not grades:
        print("No grades found in the directory.")
        return 1

    print(f"Loaded {len(grades)} grades from {args.grades_dir}")
    print(f"Starting dashboard at http://localhost:{args.port}")

    app = create_dashboard(grades)
    app.run(debug=args.debug, port=args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())

