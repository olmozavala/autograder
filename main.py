"""
CLI entrypoint for the Grader Pod system.

Orchestrates the complete grading pipeline:
1. Parse rubric from README.md
2. Iterate through student submissions
3. Run tests in Docker containers
4. Grade with LLM
5. Output results as JSON
"""

import argparse
import json
import re
import sys
import shutil
import webbrowser
import re
from pathlib import Path

from grader.config import ANSWERS_FILENAME, DEFAULT_GRADES_DIR, GRADE_OUTPUT_FILENAME, REPORT_FILENAME
from grader.config_loader import load_config
from grader.dashboard import create_dashboard
from grader.local_runner import LocalRunner
from grader.grades_aggregator import GradesAggregator, load_grades_from_dir
from grader.llm_grader import LLMGrader
from grader.models import GradeResult, StudentSubmission
from grader.rubric_parser import parse_readme


def find_submissions(submissions_dir: Path) -> list[StudentSubmission]:
    """
    Find all student submission directories.

    Args:
        submissions_dir: Path to directory containing student folders.

    Returns:
        List of StudentSubmission objects.
    """
    submissions: list[StudentSubmission] = []

    for item in sorted(submissions_dir.iterdir()):
        if not item.is_dir():
            continue

        # Skip hidden directories and common non-submission dirs
        if item.name.startswith(".") or item.name in ("__pycache__", "tests"):
            continue

        answers_path = item / ANSWERS_FILENAME
        report_path = item / REPORT_FILENAME

        # Read report content if it exists
        report_content = ""
        if report_path.exists():
            try:
                report_content = report_path.read_text(encoding="utf-8")
            except Exception:
                report_content = "[Error reading report]"

        submissions.append(
            StudentSubmission(
                student_id=item.name,
                submission_path=str(item.resolve()),
                has_answers_file=answers_path.exists(),
                has_report_file=report_path.exists(),
                report_content=report_content,
            )
        )

    return submissions


def extract_images(report_content: str) -> list[tuple[str, str]]:
    """
    Extract image links and captions from markdown.

    Matches ![caption](path) or [caption](path) where path ends in an image extension.

    Args:
        report_content: Markdown content of the report.

    Returns:
        List of (filename, caption) tuples.
    """
    # Regex for ![caption](path)
    img_regex = r"!\[(.*?)\]\((.*?)\)"
    matches = re.findall(img_regex, report_content)

    # Filter for image extensions if not already explicit
    images = []
    extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

    for caption, path in matches:
        if any(path.lower().endswith(ext) for ext in extensions):
            images.append((path, caption))

    return images


def save_grade(submission_path: Path, grade: GradeResult) -> None:
    """
    Save grade result as JSON to the submission directory.

    Args:
        submission_path: Path to student's submission directory.
        grade: GradeResult to save.
    """
    output_path = submission_path / GRADE_OUTPUT_FILENAME
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(grade.model_dump_json(indent=2))
    print(f"  Saved grade to {output_path}")


def print_grade_summary(grade: GradeResult) -> None:
    """
    Print a summary of the grade to console.

    Args:
        grade: GradeResult to summarize.
    """
    print(f"\n  {'='*50}")
    print(f"  Student: {grade.student_id}")
    print(f"  Total Score: {grade.total_score:.1f}/{grade.max_score:.1f}")
    print(f"  Tests Passed: {'Yes' if grade.code_execution_passed else 'No'}")
    print(f"  {'='*50}")

    for section in grade.sections:
        status = "+" if section.points_earned >= section.max_points * 0.7 else "-"
        print(f"  [{status}] {section.section_name}: {section.points_earned:.1f}/{section.max_points:.1f}")

    print()


def run_grading_pipeline(
    submissions_dir: Path,
    readme_path: Path,
    tests_dir: Path | None = None,
    test_data_dir: Path | None = None,
    grades_dir: Path | None = None,
    skip_llm: bool = False,
    dashboard_port: int = 8050,
    verbose: bool = False,
) -> list[GradeResult]:
    """
    Run the complete grading pipeline.

    Args:
        submissions_dir: Path to directory containing student submissions.
        readme_path: Path to the assignment README.md.
        tests_dir: Optional path to shared test files.
        test_data_dir: Optional path to test data folder (e.g., test_folder).
        grades_dir: Optional path to save aggregated grades.
        skip_llm: Skip LLM grading (use for testing runner only).
        dashboard_port: Port to run the dashboard on (default: 8050).
        verbose: Print verbose output.

    Returns:
        List of GradeResult objects for all students.
    """
    # Parse rubric
    print(f"Parsing rubric from {readme_path}...")
    rubric = parse_readme(readme_path)
    print(f"Found {len(rubric.sections)} sections, {rubric.total_points} total points")

    if verbose:
        for section in rubric.sections:
            extra = " [EXTRA]" if section.is_extra else ""
            print(f"  - {section.name}: {section.points} pts{extra}")

    # Find submissions
    print(f"\nScanning {submissions_dir} for submissions...")
    submissions = find_submissions(submissions_dir)
    print(f"Found {len(submissions)} submissions")

    if not submissions:
        print("No submissions found!")
        return []

    # Initialize components
    runner = LocalRunner(
        tests_source_dir=tests_dir,
        test_data_dir=test_data_dir,
    )
    llm_grader: LLMGrader | None = None

    if not skip_llm:
        print("Initializing LLM grader...")
        try:
            llm_grader = LLMGrader()
        except ValueError as e:
            print(f"Warning: {e}")
            print("LLM grading will be skipped.")
            skip_llm = True

    # Initialize grades aggregator
    aggregator = GradesAggregator(output_dir=grades_dir or DEFAULT_GRADES_DIR)

    # Process each submission
    results: list[GradeResult] = []

    for i, submission in enumerate(submissions, 1):
        print(f"\n[{i}/{len(submissions)}] Processing {submission.student_id}...")

        submission_path = Path(submission.submission_path)

        # Check for required files
        if not submission.has_answers_file:
            print(f"  Warning: {ANSWERS_FILENAME} not found, skipping...")
            continue

        # Run Docker tests
        from grader.models import ExecutionResult
        execution_result = ExecutionResult(
            success=True,
            setup_log="Docker execution skipped",
            test_log="",
            exit_code=0,
        )

        if runner:
            print("  Running tests...")
            execution_result = runner.run_submission(submission_path)

            if execution_result.success:
                print("  Tests: PASSED")
            else:
                print(f"  Tests: FAILED (exit code {execution_result.exit_code})")

            if verbose and execution_result.test_log:
                print("  --- Test Log ---")
                for line in execution_result.test_log.split("\n")[:20]:
                    print(f"  {line}")
                print("  ----------------")

        # LLM grading
        if llm_grader and not skip_llm:
            # Process images if present
            figure_descriptions = []
            if submission.has_report_file:
                image_links = extract_images(submission.report_content)
                if image_links:
                    print(f"  Analyzing {len(image_links)} images from report...")
                    for img_path_str, caption in image_links:
                        # Try to find the image file relative to submission
                        img_file = submission_path / img_path_str
                        if not img_file.exists():
                            # Try just the filename if path is complex
                            img_file = submission_path / Path(img_path_str).name

                        if img_file.exists():
                            print(f"    Describing {img_file.name}...")
                            desc = llm_grader.describe_image(img_file, caption)
                            figure_descriptions.append(desc)
                        else:
                            print(f"    Warning: Image file not found: {img_path_str}")

            print("  Grading with LLM...")
            grade = llm_grader.grade_submission(
                student_id=submission.student_id,
                rubric=rubric,
                execution_result=execution_result,
                report_content=submission.report_content,
                figure_descriptions=figure_descriptions,
            )
        else:
            # Create placeholder grade
            from grader.models import SectionGrade
            sections = [
                SectionGrade(
                    section_name=s.name,
                    points_earned=0,
                    max_points=s.points,
                    feedback="LLM grading skipped",
                )
                for s in rubric.sections
            ]
            grade = GradeResult(
                student_id=submission.student_id,
                total_score=0,
                max_score=sum(s.points for s in rubric.sections),
                overall_feedback="LLM grading skipped. Check 'Execution' tab for test results.",
                sections=sections,
                code_execution_passed=execution_result.success,
                execution_logs=execution_result.setup_log + "\n" + execution_result.test_log,
            )

        # Copy report and images to output folder (Always run this)
        student_output_dir = grades_dir / submission.student_id
        student_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Copy report (md or pdf)
            # Handle .md specially to rewrite image paths
            report_md_file = submission_path / "report.md"
            if report_md_file.exists():
                 with open(report_md_file, "r") as f:
                     content = f.read()
                 
                 # Rewrite image paths: ![alt](path) -> ![alt](/files/{student_id}/{path})
                 # avoiding absolute paths
                 def replace_link(match):
                     alt = match.group(1)
                     link = match.group(2)
                     if link.startswith("http") or link.startswith("/"):
                         return match.group(0)
                     # Clean up any ./ prefix
                     if link.startswith("./"):
                         link = link[2:]
                     return f"![{alt}](/files/{submission.student_id}/{link})"
                 
                 content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_link, content)
                 
                 output_report_path = student_output_dir / "report.md"
                 with open(output_report_path, "w") as f:
                     f.write(content)
                 
                 if verbose:
                     print(f"    Processed and copied report.md")

            # Handle pdf separately (just copy)
            report_pdf_file = submission_path / "report.pdf"
            if report_pdf_file.exists():
                shutil.copy2(report_pdf_file, student_output_dir / "report.pdf")
                if verbose:
                    print(f"    Copied report.pdf")

            # Copy images (for markdown rendering)
            for img_ext in ["*.png", "*.jpg", "*.jpeg", "*.gif"]:
                for img_file in submission_path.glob(img_ext):
                    shutil.copy2(img_file, student_output_dir / img_file.name)
        except Exception as e:
            print(f"    Warning: Failed to copy report files: {e}")

        # Save and display results
        save_grade(submission_path, grade)
        aggregator.add_grade(grade)
        print_grade_summary(grade)
        results.append(grade)

    # Save aggregated grades
    if results:
        print("\nSaving aggregated grades...")
        output_files = aggregator.save_all()
        print(f"  Summary JSON: {output_files.get('summary_json')}")
        print(f"  Summary CSV:  {output_files.get('summary_csv')}")

    # Print summary
    print("\n" + "=" * 60)
    print("GRADING COMPLETE")
    print("=" * 60)
    print(f"Total submissions processed: {len(results)}")

    if results:
        avg_score = sum(r.total_score for r in results) / len(results)
        avg_max = sum(r.max_score for r in results) / len(results)
        print(f"Average score: {avg_score:.1f}/{avg_max:.1f} ({100*avg_score/avg_max:.1f}%)")

        passed = sum(1 for r in results if r.code_execution_passed)
        print(f"Tests passed: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")

        print("\nLaunching Dashboard...")
        print("Press Ctrl+C to stop the server.")
        try:
            import os
            # Only open browser on the main process, not the reloader
            if not os.environ.get("WERKZEUG_RUN_MAIN"):
                url = f"http://127.0.0.1:{dashboard_port}"
                print(f"Opening {url} in browser...")
                webbrowser.open(url)
            
            app = create_dashboard(results, grades_dir=aggregator.output_dir)
            app.run(debug=verbose, port=dashboard_port)
        except Exception as e:
            print(f"Error launching dashboard: {e}")

    return results


def main() -> int:
    """
    Main CLI entrypoint.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Grader Pod: Automated homework grading with Docker + LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grade all submissions
  python main.py --submissions ./hm3-submissions/ --readme ./README.md

  # Grade with shared tests
  python main.py --submissions ./submissions/ --readme ./README.md --tests ./tests/

  # Skip Docker (test LLM only)
  python main.py --submissions ./submissions/ --readme ./README.md --skip-docker

  # Skip LLM (test Docker only)
  python main.py --submissions ./submissions/ --readme ./README.md --skip-llm
""",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--submissions",
        type=Path,
        help="Path to directory containing student submission folders",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        help="Path to the assignment README.md with rubric",
    )
    parser.add_argument(
        "--tests",
        type=Path,
        default=None,
        help="Path to shared test files to copy into submissions",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="Path to test data folder (e.g., test_folder with ECG data)",
    )
    parser.add_argument(
        "--grades-dir",
        type=Path,
        default=None,
        help="Path to save aggregated grades (default: ./grades)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM grading (for testing execution)",
    )
    parser.add_argument(
        "--only-dashboard",
        action="store_true",
        help="Launch dashboard with existing grades (skip grading)",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8050,
        help="Port to run the dashboard on",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Load configuration from file if provided
    submissions_dir = args.submissions
    readme_path = args.readme
    tests_dir = args.tests
    test_data_dir = args.test_data
    grades_dir = args.grades_dir
    skip_llm = args.skip_llm
    only_dashboard = args.only_dashboard
    dashboard_port = args.dashboard_port
    verbose = args.verbose

    # Default to grader_config.yml if not provided and it exists
    if not args.config:
        default_config = Path("grader_config.yml")
        if default_config.exists():
            args.config = default_config

    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded configuration from {args.config}")
            
            # CLI args override config
            if not submissions_dir: submissions_dir = config.submissions_dir
            if not readme_path: readme_path = config.readme_path
            if not tests_dir: tests_dir = config.tests_dir
            if not test_data_dir: test_data_dir = config.test_data_dir
            if not grades_dir: grades_dir = config.grades_dir
            
            # Boolean flags
            if not skip_llm: skip_llm = config.skip_llm
            if not only_dashboard: only_dashboard = config.only_dashboard

            if args.dashboard_port == 8050: dashboard_port = config.dashboard_port
            if not verbose: verbose = config.verbose
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1

    # Check for only_dashboard mode
    if only_dashboard:
        grades_dir = grades_dir or DEFAULT_GRADES_DIR
        if not grades_dir.exists():
            print(f"Error: Grades directory not found: {grades_dir}")
            return 1
            
        print(f"Launching dashboard from {grades_dir}...")
        grades = load_grades_from_dir(grades_dir)
        if not grades:
             print("No grades found.")
             return 1
             
        try:
            import os
            # Only open browser on the main process, not the reloader
            if not os.environ.get("WERKZEUG_RUN_MAIN"):
                url = f"http://127.0.0.1:{dashboard_port}"
                print(f"Opening {url} in browser...")
                webbrowser.open(url)
            
            app = create_dashboard(grades, grades_dir=grades_dir)
            app.run(debug=verbose, port=dashboard_port)
            return 0
        except Exception as e:
            print(f"Error launching dashboard: {e}")
            return 1
    # Validate required arguments
    if not submissions_dir:
        print("Error: --submissions argument or specificiation in config file is required")
        return 1
    if not readme_path:
        print("Error: --readme argument or specification in config file is required")
        return 1

    # Validate paths
    if not submissions_dir.exists():
        print(f"Error: Submissions directory not found: {submissions_dir}")
        return 1

    if not readme_path.exists():
        print(f"Error: README not found: {readme_path}")
        return 1

    if tests_dir and not tests_dir.exists():
        print(f"Warning: Tests directory not found: {tests_dir}")
        print("Proceeding without shared tests (grading based on report/submission only).")
        tests_dir = None

    if test_data_dir and not test_data_dir.exists():
        print(f"Warning: Test data directory not found: {test_data_dir}")
        test_data_dir = None

    try:
        run_grading_pipeline(
            submissions_dir=submissions_dir,
            readme_path=readme_path,
            tests_dir=tests_dir,
            test_data_dir=test_data_dir,
            grades_dir=grades_dir,
            skip_llm=skip_llm,
            dashboard_port=dashboard_port,
            verbose=verbose,
        )
        return 0
    except KeyboardInterrupt:
        print("\nGrading interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
