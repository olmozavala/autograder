"""
Pydantic models for the Grader Pod system.

Defines structured data types for rubric parsing, grading results,
and OpenAI structured output schemas.
"""

from pydantic import BaseModel, Field


class ImageDescription(BaseModel):
    """
    Semantic description of an image found in report.md.

    Attributes:
        filename: Relative path to the image file.
        caption: Caption text associated with the image.
        description: AI-generated description of the image content.
    """

    filename: str = Field(..., description="Relative path to image file")
    caption: str = Field(default="", description="Image caption from markdown")
    description: str = Field(..., description="AI-generated semantic description")


class RubricSection(BaseModel):
    """
    Represents a single grading section parsed from the README.md.

    Attributes:
        name: The section title (e.g., "Database description").
        points: Maximum points available for this section.
        is_extra: Whether this is an extra credit section.
        description: Optional description text from the README.
        expected_functions: List of function names expected in answers.py.
    """

    name: str = Field(..., description="Section title from README")
    points: float = Field(..., ge=0, description="Maximum points for this section")
    is_extra: bool = Field(default=False, description="Whether this is extra credit")
    description: str = Field(default="", description="Section description text")
    expected_functions: list[str] = Field(
        default_factory=list, description="Expected function names in answers.py"
    )


class Rubric(BaseModel):
    """
    Complete rubric parsed from README.md.

    Attributes:
        title: Assignment title.
        total_points: Total possible points (excluding extra credit).
        sections: List of grading sections.
        raw_content: Original README content for LLM context.
    """

    title: str = Field(..., description="Assignment title")
    total_points: float = Field(..., ge=0, description="Total points (excluding extra)")
    sections: list[RubricSection] = Field(
        default_factory=list, description="Grading sections"
    )
    raw_content: str = Field(default="", description="Original README content")


class SectionGrade(BaseModel):
    """
    Grade result for a single rubric section.

    Attributes:
        section_name: Name of the section being graded.
        points_earned: Points awarded for this section.
        max_points: Maximum possible points.
        feedback: Detailed feedback explaining the grade.
    """

    section_name: str = Field(..., description="Name of the graded section")
    points_earned: float = Field(..., ge=0, description="Points awarded")
    max_points: float = Field(..., ge=0, description="Maximum possible points")
    feedback: str = Field(..., description="Detailed feedback for this section")


class TestResult(BaseModel):
    """
    Result from pytest execution.

    Attributes:
        test_name: Name of the test function.
        passed: Whether the test passed.
        error_message: Error message if test failed.
        duration_seconds: Time taken to run the test.
    """

    test_name: str = Field(..., description="Test function name")
    passed: bool = Field(..., description="Whether the test passed")
    error_message: str = Field(default="", description="Error message if failed")
    duration_seconds: float = Field(default=0.0, ge=0, description="Test duration")


class ExecutionResult(BaseModel):
    """
    Result from running student code in Docker container.

    Attributes:
        success: Whether execution completed without errors.
        setup_log: Output from dependency installation.
        test_log: Output from pytest execution.
        exit_code: Process exit code.
        tests: Parsed test results from JUnit XML.
        timeout_exceeded: Whether the execution timed out.
    """

    success: bool = Field(..., description="Whether execution succeeded")
    setup_log: str = Field(default="", description="Dependency installation output")
    test_log: str = Field(default="", description="Pytest execution output")
    exit_code: int = Field(default=-1, description="Process exit code")
    tests: list[TestResult] = Field(default_factory=list, description="Parsed test results")
    timeout_exceeded: bool = Field(default=False, description="Whether timeout was exceeded")


class GradeResult(BaseModel):
    """
    Complete grading result for a student submission.

    This model is used as the structured output schema for OpenAI's API,
    ensuring consistent JSON output from the LLM.

    Attributes:
        student_id: Student identifier (folder name).
        sections: Per-section grade breakdowns.
        code_execution_passed: Whether all tests passed.
        total_score: Total points earned.
        max_score: Maximum possible points.
        overall_feedback: Summary feedback for the student.
    """

    student_id: str = Field(..., description="Student identifier (folder name)")
    sections: list[SectionGrade] = Field(
        ..., description="Per-section grade breakdowns"
    )
    code_execution_passed: bool = Field(
        ..., description="Whether all deterministic tests passed"
    )
    total_score: float = Field(..., ge=0, description="Total points earned")
    max_score: float = Field(..., ge=0, description="Maximum possible points")
    overall_feedback: str = Field(
        ..., description="Constructive summary feedback for the student"
    )
    github_repo: str | None = Field(default=None, description="GitHub repository owner/name")
    submission_path: str | None = Field(default=None, description="Path to the original submission directory")


class StudentSubmission(BaseModel):
    """
    Represents a student's submission for grading.

    Attributes:
        student_id: Student identifier (folder name).
        submission_path: Path to the submission directory.
        has_answers_file: Whether answers.py exists.
        has_report_file: Whether report.md exists.
        report_content: Content of report.md if it exists.
    """

    student_id: str = Field(..., description="Student identifier")
    submission_path: str = Field(..., description="Path to submission directory")
    has_answers_file: bool = Field(..., description="Whether answers.py exists")
    has_report_file: bool = Field(..., description="Whether report.md exists")
    report_content: str = Field(default="", description="Content of report.md")
    report_file_path: str | None = Field(default=None, description="Relative path to report file")
    github_repo: str | None = Field(default=None, description="GitHub repository owner/name")
    figure_descriptions: list[ImageDescription] = Field(
        default_factory=list, description="Semantic descriptions of images in report"
    )

