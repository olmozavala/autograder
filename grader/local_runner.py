"""
Local execution runner for student code.

Executes pytest on the host machine using uv, capturing results
without Docker isolation.
"""

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import (
    ANSWERS_FILENAME,
    TEACHER_ANSWERS_FILENAME,
    POSSIBLE_ANSWERS_FILENAMES,
    EXECUTION_TIMEOUT_SECONDS,
    PYTEST_ARGS,
    TEST_REPORT_FILENAME,
)
from .models import ExecutionResult, TestResult


class LocalRunner:
    """
    Runs student code locally on the host machine.

    Uses subprocess to execute pytest and captures results.
    """

    def __init__(
        self,
        timeout_seconds: int = EXECUTION_TIMEOUT_SECONDS,
        tests_source_dir: Path | None = None,
        test_data_dir: Path | None = None,
        venv_python: str | None = None,
    ) -> None:
        """
        Initialize the Local runner.

        Args:
            timeout_seconds: Maximum execution time per student.
            tests_source_dir: Path to shared test files to copy into submissions.
            test_data_dir: Path to test data folder (e.g., test_folder) to copy.
            venv_python: Path to the python executable in the uv environment.
        """
        self.timeout_seconds = timeout_seconds
        self.tests_source_dir = tests_source_dir
        self.test_data_dir = test_data_dir
        self.venv_python = venv_python

    def run_submission(self, submission_path: Path) -> ExecutionResult:
        """
        Run a student submission locally.

        Args:
            submission_path: Path to the student's submission directory.

        Returns:
            ExecutionResult containing logs, test results, and exit codes.
        """
        # Validate submission
        if not submission_path.exists():
            return ExecutionResult(
                success=False,
                setup_log=f"Submission path not found: {submission_path}",
                exit_code=-1,
            )

        answers_file = None
        for filename in POSSIBLE_ANSWERS_FILENAMES:
            if (submission_path / filename).exists():
                answers_file = submission_path / filename
                break

        if not answers_file:
            filenames_str = " or ".join([f"'{f}'" for f in POSSIBLE_ANSWERS_FILENAMES])
            return ExecutionResult(
                success=False,
                setup_log=f"Required file ({filenames_str}) not found in {submission_path}",
                exit_code=-1,
            )

        # Copy shared tests if configured
        if self.tests_source_dir and self.tests_source_dir.exists():
            self._copy_tests(submission_path)

        # Ensure answers.py exists (copy from teacher_answers.py if needed)
        dest_answers = submission_path / ANSWERS_FILENAME
        src_teacher = submission_path / TEACHER_ANSWERS_FILENAME
        if not dest_answers.exists() and src_teacher.exists():
            shutil.copy2(src_teacher, dest_answers)

        try:
            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{submission_path.resolve()}:{env.get('PYTHONPATH', '')}"

            # Prepare pytest command
            # Using uv run to handle dependencies locally
            # Prepare pytest command
            if self.venv_python:
                # Use configured python executable directly
                cmd = [
                    self.venv_python, "-m", "pytest",
                    "tests/",
                    f"--junitxml={TEST_REPORT_FILENAME}",
                    "-v",
                    "--tb=short",
                ]
            else:
                 # Fallback to sys.executable (current environment)
                import sys
                cmd = [
                    sys.executable, "-m", "pytest",
                    "tests/",
                    f"--junitxml={TEST_REPORT_FILENAME}",
                    "-v",
                    "--tb=short",
                ]

            print(f"  Executing: {' '.join(cmd)}")
            
            # Run pytest
            process = subprocess.run(
                cmd,
                cwd=str(submission_path.resolve()),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )

            test_log = process.stdout + process.stderr
            test_exit = process.returncode

            # Parse test results from JUnit XML if available
            tests = self._parse_junit_xml(submission_path / TEST_REPORT_FILENAME)

            success = test_exit == 0

            return ExecutionResult(
                success=success,
                setup_log="Local environment used",
                test_log=test_log,
                exit_code=test_exit,
                tests=tests,
                timeout_exceeded=False,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                setup_log="Execution timed out",
                exit_code=-1,
                timeout_exceeded=True,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                setup_log=f"Local execution error: {str(e)}",
                exit_code=-1,
            )

    def _copy_tests(self, submission_path: Path) -> None:
        """
        Copy shared test files and test data to the submission directory.
        (Mirrors DockerRunner implementation)
        """
        if self.tests_source_dir and self.tests_source_dir.exists():
            dest_tests = submission_path / "tests"
            if dest_tests.exists():
                shutil.rmtree(dest_tests, ignore_errors=True)
            shutil.copytree(self.tests_source_dir, dest_tests)

        if self.test_data_dir and self.test_data_dir.exists():
            dest_data = submission_path / self.test_data_dir.name
            if dest_data.exists():
                shutil.rmtree(dest_data, ignore_errors=True)
            shutil.copytree(self.test_data_dir, dest_data)

    def _parse_junit_xml(self, xml_path: Path) -> list[TestResult]:
        """
        Parse pytest JUnit XML output.
        (Mirrors DockerRunner implementation)
        """
        if not xml_path.exists():
            return []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            results: list[TestResult] = []

            for testcase in root.iter("testcase"):
                name = testcase.get("name", "unknown")
                time_str = testcase.get("time", "0")
                duration = float(time_str) if time_str else 0.0

                failure = testcase.find("failure")
                error = testcase.find("error")

                if failure is not None:
                    results.append(
                        TestResult(
                            test_name=name,
                            passed=False,
                            error_message=failure.text or failure.get("message", ""),
                            duration_seconds=duration,
                        )
                    )
                elif error is not None:
                    results.append(
                        TestResult(
                            test_name=name,
                            passed=False,
                            error_message=error.text or error.get("message", ""),
                            duration_seconds=duration,
                        )
                    )
                else:
                    results.append(
                        TestResult(
                            test_name=name,
                            passed=True,
                            duration_seconds=duration,
                        )
                    )

            return results
        except Exception:
            return []
