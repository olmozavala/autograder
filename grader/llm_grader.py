"""
LLM-based semantic grading using OpenAI's GPT-4o.

Analyzes student reports and correlates with pytest execution results
to provide comprehensive grading with structured output.
"""

import base64
import netrc
import os
from pathlib import Path

from openai import OpenAI

from .config import MAX_TOKENS, OPENAI_MODEL
from .models import ExecutionResult, GradeResult, ImageDescription, Rubric, SectionGrade
from .rubric_parser import format_rubric_for_llm


class LLMGrader:
    """
    Semantic grader using OpenAI's GPT-4o with structured output.

    Combines rubric requirements, execution results, and report content
    to produce comprehensive grades with detailed feedback.
    """

    def __init__(self, model: str = OPENAI_MODEL, api_key: str | None = None) -> None:
        """
        Initialize the LLM grader.

        Args:
            model: OpenAI model to use (default: gpt-4o).
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.model = model

        # Priority: 1. Argument, 2. .netrc (machine OPENAI), 3. Environment variable
        if api_key is None:
            # Try to read from .netrc
            try:
                secrets = netrc.netrc()
                auth = secrets.authenticators("OPENAI")
                if auth:
                    api_key = auth[0]  # auth[0] is the login info
            except (FileNotFoundError, netrc.NetrcParseError, Exception):
                pass

        api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable, "
                "add machine OPENAI to your .netrc file, or pass api_key parameter."
            )

        self.client = OpenAI(api_key=api_key)

    def grade_submission(
        self,
        student_id: str,
        rubric: Rubric,
        execution_result: ExecutionResult,
        report_content: str,
        figure_descriptions: list[ImageDescription] | None = None,
    ) -> GradeResult:
        """
        Grade a student submission using GPT-4o.

        Analyzes the report for scientific correctness and correlates
        with deterministic test results.

        Args:
            student_id: Student identifier (folder name).
            rubric: Parsed assignment rubric.
            execution_result: Results from pytest execution.
            report_content: Content of student's report.md.

        Returns:
            GradeResult with per-section scores and feedback.
        """
        prompt = self._build_prompt(rubric, execution_result, report_content, figure_descriptions)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format=GradeResult,
                max_completion_tokens=MAX_TOKENS,
            )

            result = completion.choices[0].message.parsed
            if result is None:
                return self._create_fallback_result(student_id, rubric, execution_result)

            # Ensure student_id is set correctly
            result.student_id = student_id

            # Enforce rubric section names and max points
            llm_sections = {s.section_name: s for s in result.sections}
            new_sections = []
            
            print(f"    [DEBUG] LLM returned sections: {list(llm_sections.keys())}")
            
            for rubric_section in rubric.sections:
                matched = None
                # Try exact match first
                if rubric_section.name in llm_sections:
                    matched = llm_sections[rubric_section.name]
                else:
                    # Try fuzzy match (case-insensitive, substring in either direction)
                    r_name = rubric_section.name.lower()
                    for l_name in llm_sections:
                        l_name_lower = l_name.lower()
                        if l_name_lower == r_name or l_name_lower in r_name or r_name in l_name_lower:
                            matched = llm_sections[l_name]
                            print(f"    [DEBUG] Matched rubric '{rubric_section.name}' to LLM '{l_name}'")
                            break
                
                if matched:
                    # Enforce rubric values
                    points_earned = min(matched.points_earned, rubric_section.points)
                    new_sections.append(SectionGrade(
                        section_name=rubric_section.name,
                        points_earned=points_earned,
                        max_points=rubric_section.points,
                        feedback=matched.feedback
                    ))
                else:
                    print(f"    [DEBUG] No match for rubric section: '{rubric_section.name}'")
                    new_sections.append(SectionGrade(
                        section_name=rubric_section.name,
                        points_earned=0,
                        max_points=rubric_section.points,
                        feedback="Section not found or not graded by LLM."
                    ))
            
            # Replace sections and recalculate totals
            result.sections = new_sections
            result.total_score = sum(s.points_earned for s in new_sections if not any(
                kw in s.section_name.lower() for kw in ["extra", "bonus"]
            ))
            result.max_score = rubric.total_points
            
            return result

        except Exception as e:
            print(f"LLM grading failed for {student_id}: {e}")
            return self._create_fallback_result(student_id, rubric, execution_result)

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the grading LLM.

        Returns:
            System prompt string.
        """
        return """You are a strict but fair academic grader for a graduate Scientific Computing course.

Your role is to:
1. Evaluate student submissions against the provided rubric
2. Correlate code execution results with report quality
3. Provide constructive, specific feedback

CRITICAL GRADING RULES:

1. **TEST FAILURES = NEAR-ZERO CREDIT**
   - If pytest tests FAILED for a section, give at most 1-2 points out of 10
   - Failed tests indicate the code does not work, regardless of explanations
   - Only give minimal credit if the report shows partial conceptual understanding

2. **PASSING TESTS ARE NOT ENOUGH**
   - Even if tests PASS, the report MUST demonstrate:
     * Clear explanation of the approach and methodology
     * Verification that results are correct (e.g., sample outputs shown)
     * Understanding of WHY the code works, not just THAT it works
   - Penalize 20-40% if tests pass but explanation is missing or superficial

3. **REPORT MUST SHOW VERIFICATION**
   - Student must show they ran their code and checked the outputs
   - Include example outputs, screenshots, or result summaries
   - Simply showing code without demonstrating it works = significant penalty

Grading Scale:
- 10/10: Tests pass AND excellent explanation with verified results
- 7-9/10: Tests pass AND good explanation, minor gaps in verification
- 4-6/10: Tests pass BUT weak/missing explanation or no result verification
- 1-3/10: Tests FAILED but shows some conceptual understanding
- 0/10: Tests failed AND no meaningful attempt or explanation

Additional Guidelines:
- Extra credit sections: only award points if attempted AND done well
- Be specific in feedback about what was missing or incorrect
- Encourage students to show their work and verify results"""

    def _build_prompt(
        self,
        rubric: Rubric,
        execution_result: ExecutionResult,
        report_content: str,
        figure_descriptions: list[ImageDescription] | None = None,
    ) -> str:
        """
        Build the grading prompt for the LLM.

        Args:
            rubric: Parsed assignment rubric.
            execution_result: Results from pytest execution.
            report_content: Content of student's report.md.

        Returns:
            Complete prompt string.
        """
        # Format test results
        test_summary = self._format_test_results(execution_result)

        # Format rubric
        rubric_text = format_rubric_for_llm(rubric)

        # Build section list for expected output
        sections_list = ", ".join(f'"{s.name}"' for s in rubric.sections)

        prompt = f"""
# GRADING TASK

## Assignment Rubric
{rubric_text}

## Code Execution Results (Deterministic Tests)

### Setup Log:
```
{execution_result.setup_log[:2000] if execution_result.setup_log else "No setup log available"}
```

### Test Execution Log:
```
{execution_result.test_log[:3000] if execution_result.test_log else "No test log available"}
```

### Test Summary:
{test_summary}

### Overall Execution Status:
- Success: {execution_result.success}
- Exit Code: {execution_result.exit_code}
- Timeout Exceeded: {execution_result.timeout_exceeded}

## Student Report (report.md):
```markdown
{report_content[:8000] if report_content else "NO REPORT SUBMITTED"}
```

## Figure Descriptions (from student images):
{self._format_figure_descriptions(figure_descriptions) if figure_descriptions else "No images found in report or analyzed."}

## Instructions

Grade this submission according to the rubric above. For each section ({sections_list}), provide:
1. Points earned (out of the max for that section)
2. Specific feedback explaining the grade

Consider:
- Did the code tests pass or fail? This is ground truth for code correctness.
- Does the report demonstrate understanding of the concepts?
- Is the report complete and well-written?
- For visualization sections, note if images are present/referenced in the report

Provide overall feedback that is constructive and helps the student improve.
"""
        return prompt

    def _format_test_results(self, execution_result: ExecutionResult) -> str:
        """
        Format test results for the prompt.

        Args:
            execution_result: Results from pytest execution.

        Returns:
            Formatted test results string.
        """
        if not execution_result.tests:
            if execution_result.success:
                return "No detailed test results available, but execution completed successfully."
            else:
                return "No detailed test results available. Execution may have failed before tests could run."

        lines = []
        passed = sum(1 for t in execution_result.tests if t.passed)
        failed = len(execution_result.tests) - passed

        lines.append(f"Total: {len(execution_result.tests)} tests, {passed} passed, {failed} failed")
        lines.append("")

        for test in execution_result.tests:
            status = "PASSED" if test.passed else "FAILED"
            lines.append(f"- {test.test_name}: {status}")
            if not test.passed and test.error_message:
                # Truncate long error messages
                error_preview = test.error_message[:300]
                if len(test.error_message) > 300:
                    error_preview += "..."
                lines.append(f"  Error: {error_preview}")

        return "\n".join(lines)

    def _format_figure_descriptions(self, figure_descriptions: list[ImageDescription]) -> str:
        """Format figure descriptions for the prompt."""
        if not figure_descriptions:
            return "No figure descriptions available."
        
        lines = []
        for fig in figure_descriptions:
            caption_str = f" (Caption: {fig.caption})" if fig.caption else ""
            lines.append(f"### Figure: {fig.filename}{caption_str}")
            lines.append(f"Description: {fig.description}")
            lines.append("")
        return "\n".join(lines)

    def describe_image(self, image_path: Path, caption: str = "") -> ImageDescription:
        """
        Use OpenAI Vision to describe an image.
        
        Args:
            image_path: Path to the image file.
            caption: Caption found in markdown.
            
        Returns:
            ImageDescription object.
        """
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use vision-capable model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a scientific assistant. Describe the provided image concisely, focusing on content, data shown, and any labels or titles visible. Your description will be used for grading an academic assignment.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Please describe this image. It has the following caption: {caption}" if caption else "Please describe this image."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    },
                ],
                max_tokens=500,
            )

            description = response.choices[0].message.content
            return ImageDescription(
                filename=image_path.name,
                caption=caption,
                description=description or "Failed to generate description."
            )
        except Exception as e:
            print(f"Failed to describe image {image_path}: {e}")
            return ImageDescription(
                filename=image_path.name,
                caption=caption,
                description=f"Error analyzing image: {str(e)}"
            )

    def _create_fallback_result(
        self,
        student_id: str,
        rubric: Rubric,
        execution_result: ExecutionResult,
    ) -> GradeResult:
        """
        Create a fallback grade result when LLM grading fails.

        Uses execution results to provide basic scores.

        Args:
            student_id: Student identifier.
            rubric: Parsed assignment rubric.
            execution_result: Results from pytest execution.

        Returns:
            Basic GradeResult based on execution results.
        """
        sections = []
        for section in rubric.sections:
            # Give partial credit based on execution success
            if execution_result.success:
                points = section.points * 0.7  # 70% if tests pass
                feedback = "Tests passed. Manual review needed for full grading."
            else:
                points = section.points * 0.3  # 30% base
                feedback = "Tests failed. Manual review required."

            sections.append(
                SectionGrade(
                    section_name=section.name,
                    points_earned=points,
                    max_points=section.points,
                    feedback=feedback,
                )
            )

        total_score = sum(s.points_earned for s in sections if not any(
            kw in s.section_name.lower() for kw in ["extra", "bonus"]
        ))
        max_score = rubric.total_points

        return GradeResult(
            student_id=student_id,
            sections=sections,
            code_execution_passed=execution_result.success,
            total_score=total_score,
            max_score=max_score,
            overall_feedback=(
                "LLM grading failed. This is a fallback result based on test execution. "
                "Manual review is required for accurate grading."
            ),
        )
