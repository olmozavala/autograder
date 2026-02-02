"""
Grader Pod: Automated homework grading with Docker + LLM

Usage:
  main.py [--config=PATH]
  main.py (-h | --help)

Options:
  --config=PATH  Path to YAML configuration file [default: grader_config.yml].
  -h --help      Show this screen.
"""

from docopt import docopt
import json
import re
import sys
import shutil
import webbrowser
import re
import subprocess
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
        if item.name.startswith(".") or item.name in ("__pycache__", "tests", "GRADES", "grades"):
            continue

        answers_path = item / ANSWERS_FILENAME
        
        # Fuzzy match report file
        report_path = None
        report_file_name = None
        
        # First check for exact match
        if (item / REPORT_FILENAME).exists():
            report_path = item / REPORT_FILENAME
            report_file_name = REPORT_FILENAME
        else:
            # Look for any file with "report" in the name (case-insensitive) but NOT "example"
            candidates = [f for f in item.iterdir() if f.is_file() and "report" in f.name.lower() and "example" not in f.name.lower()]
            # Prioritize markdown files
            md_candidates = [f for f in candidates if f.suffix.lower() == ".md"]
            if md_candidates:
                report_path = md_candidates[0]
                report_file_name = report_path.name
            elif candidates:
                report_path = candidates[0]
                report_file_name = report_path.name

        # Read report content if it exists
        report_content = ""
        if report_path and report_path.exists():
            try:
                report_content = report_path.read_text(encoding="utf-8")
            except Exception:
                report_content = "[Error reading report]"

        # Try to find GitHub repository
        github_repo = None
        try:
            # Run git remote get-url origin in the submission directory
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(item),
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Parse owner/repo from URL
                # Handle git@github.com:owner/repo.git or https://github.com/owner/repo.git
                match = re.search(r"github\.com[:/](.+?)\.git", url)
                if match:
                    github_repo = match.group(1)
        except Exception:
            pass

        submissions.append(
            StudentSubmission(
                student_id=item.name,
                submission_path=str(item.resolve()),
                has_answers_file=answers_path.exists(),
                has_report_file=report_path is not None and report_path.exists(),
                report_content=report_content,
                report_file_path=report_file_name,
                github_repo=github_repo,
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

    images = []
    from grader.config import IMAGE_EXTENSIONS
    
    for caption, path in matches:
        if any(path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
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

        # Prepare execution result
        from grader.models import ExecutionResult
        execution_result = None

        # Check for required files
        if not submission.has_answers_file:
            print(f"  Warning: {ANSWERS_FILENAME} not found for student '{submission.student_id}'. Proceeding with report-only grading.")
            execution_result = ExecutionResult(
                success=False,
                setup_log=f"Error: Required file {ANSWERS_FILENAME} is missing.",
                test_log=f"Deterministic tests could not be run because {ANSWERS_FILENAME} was not found in the submission directory.",
                exit_code=1,
            )

        # Run tests if answers file exists and runner is available
        if submission.has_answers_file and runner:
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
        
        # Ensure execution_result is never None
        if execution_result is None:
            execution_result = ExecutionResult(
                success=True,
                setup_log="Execution skipped (no runner or file check bypassed)",
                test_log="",
                exit_code=0,
            )

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
            # Propagate github_repo and submission_path from submission
            grade.github_repo = submission.github_repo
            grade.submission_path = submission.submission_path
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
                github_repo=submission.github_repo,
                submission_path=submission.submission_path,
            )

        # Copy report and images to output folder (Always run this)
        student_output_dir = grades_dir / submission.student_id
        student_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Copy report file if found
            if submission.report_file_path:
                source_path = submission_path / submission.report_file_path
                suffix = source_path.suffix.lower()

                if suffix == ".md":
                    # Process markdown: rewrite image paths and save as report.md
                    with open(source_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    def replace_link(match):
                        alt = match.group(1)
                        link = match.group(2)
                        if link.startswith("http") or link.startswith("/"):
                            return match.group(0)
                        if link.startswith("./"):
                            link = link[2:]
                        return f"![{alt}](/files/{submission.student_id}/{link})"

                    content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_link, content)
                    
                    with open(student_output_dir / "report.md", "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    if verbose:
                        print(f"    Processed and verified report.md")

                elif suffix == ".pdf":
                    # Copy PDF as report.pdf
                    shutil.copy2(source_path, student_output_dir / "report.pdf")
                    if verbose:
                        print(f"    Copied report.pdf")
                else:
                    # Copy other types with original name (or standardize if needed)
                    shutil.copy2(source_path, student_output_dir / "report.txt")

            # Copy images (for markdown rendering)
            from grader.config import IMAGE_EXTENSIONS
            for ext in IMAGE_EXTENSIONS:
                pattern = f"**/*{ext}" if not ext.startswith("*") else f"**/{ext}"
                for img_file in submission_path.glob(pattern):
                    # Calculate relative path to support subfolders (e.g. figures/plot.png)
                    rel_path = img_file.relative_to(submission_path)
                    dest_path = student_output_dir / rel_path
                    
                    # Ensure parent directory exists
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(img_file, dest_path)
                    if verbose and img_file.parent != submission_path:
                         print(f"    Copied image from subfolder: {rel_path}")

        except Exception as e:
            print(f"    Warning: Failed to copy report files: {e}")

        # Save and display results
        save_grade(submission_path, grade)
        aggregator.add_grade(grade)
        print_grade_summary(grade)
        results.append(grade)

    # Sort results by github_repo
    results.sort(key=lambda x: (x.github_repo or "", x.student_id))

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
    arguments = docopt(__doc__)
    config_path = Path(arguments["--config"])

    if not config_path.exists():
        # Try default if not specified explicitly and default exists
        if config_path.name == "grader_config.yml" and not config_path.exists():
             print(f"Error: Configuration file not found at {config_path}")
             return 1
        elif not config_path.exists():
             print(f"Error: Configuration file not found at {config_path}")
             return 1

    try:
        config = load_config(config_path)
        print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Check for only_dashboard mode
    if config.only_dashboard:
        grades_dir = config.grades_dir or DEFAULT_GRADES_DIR
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
                url = f"http://127.0.0.1:{config.dashboard_port}"
                print(f"Opening {url} in browser...")
                webbrowser.open(url)
            
            app = create_dashboard(grades, grades_dir=grades_dir)
            app.run(debug=config.verbose, port=config.dashboard_port)
            return 0
        except Exception as e:
            print(f"Error launching dashboard: {e}")
            return 1

    # Validate required paths from config
    if not config.submissions_dir:
        print("Error: submissions_dir must be specified in the configuration file")
        return 1
    if not config.readme_path:
        print("Error: readme_path must be specified in the configuration file")
        return 1

    if not config.submissions_dir.exists():
        print(f"Error: Submissions directory not found: {config.submissions_dir}")
        return 1

    if not config.readme_path.exists():
        print(f"Error: README not found: {config.readme_path}")
        return 1

    try:
        run_grading_pipeline(
            submissions_dir=config.submissions_dir,
            readme_path=config.readme_path,
            tests_dir=config.tests_dir,
            test_data_dir=config.test_data_dir,
            grades_dir=config.grades_dir,
            skip_llm=config.skip_llm,
            dashboard_port=config.dashboard_port,
            verbose=config.verbose,
        )
        return 0
    except KeyboardInterrupt:
        print("\nGrading interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
