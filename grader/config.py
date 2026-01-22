"""
Configuration constants for the Grader Pod system.
"""

from pathlib import Path



# Execution configuration
EXECUTION_TIMEOUT_SECONDS: int = 120

# File patterns
ANSWERS_FILENAME: str = "answers.py"
TEACHER_ANSWERS_FILENAME: str = "teacher_answers.py"
POSSIBLE_ANSWERS_FILENAMES: list[str] = [ANSWERS_FILENAME, TEACHER_ANSWERS_FILENAME]
REPORT_FILENAME: str = "report.md"
GRADE_OUTPUT_FILENAME: str = "grade.json"
TEST_REPORT_FILENAME: str = "test_report.xml"
IMAGE_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]

# Pytest configuration
PYTEST_ARGS: list[str] = [
    "pytest",
    "tests/",
    "--junitxml=test_report.xml",
    "-v",
    "--tb=short",
]

# OpenAI configuration
# Available models: "gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"
# gpt-4o-mini is the most cost-effective for grading tasks
OPENAI_MODEL: str = "gpt-4o-mini"
MAX_TOKENS: int = 4096

# Default paths (can be overridden via CLI)
DEFAULT_TESTS_DIR: Path = Path("tests")
DEFAULT_GRADES_DIR: Path = Path("grades")
GRADES_SUMMARY_FILENAME: str = "grades_summary.json"
GRADES_CSV_FILENAME: str = "grades_summary.csv"

# Rubric parsing patterns
# Matches: "# Section Name (10 pts)" or "# Section Name (Extra 10 pts)"
SECTION_PATTERN: str = r"^#+\s+(.+?)\s*\((?:Extra\s+)?(\d+)\s*(?:pts|points)?\s*\)"
EXTRA_CREDIT_KEYWORDS: list[str] = ["extra", "bonus", "optional"]

