"""
Dynamic README.md parser for extracting grading rubrics.

Parses markdown headings to extract section names, point values,
and identifies expected function signatures from code blocks.
"""

import re
from pathlib import Path

from .config import EXTRA_CREDIT_KEYWORDS, SECTION_PATTERN
from .models import Rubric, RubricSection


def parse_readme(readme_path: Path) -> Rubric:
    """
    Parse a README.md file to extract the grading rubric.

    Extracts section titles with point values from markdown headings
    in the format: `# Section Name (X pts)` or `# Section Name (X)`.

    Also detects function signatures from Python code blocks.

    Args:
        readme_path: Path to the README.md file.

    Returns:
        Rubric object containing all parsed sections and metadata.

    Raises:
        FileNotFoundError: If the README.md file doesn't exist.
        ValueError: If no valid sections are found.
    """
    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path}")

    content = readme_path.read_text(encoding="utf-8")

    # Extract title (and its point value to exclude from sections)
    title, title_points = _extract_title(content, readme_path)

    # Extract sections, excluding the title heading
    sections = _extract_sections(content, exclude_title=title)

    if not sections:
        raise ValueError(f"No grading sections found in {readme_path}")

    # Calculate total points (excluding extra credit)
    total_points = sum(s.points for s in sections if not s.is_extra)

    return Rubric(
        title=title,
        total_points=total_points,
        sections=sections,
        raw_content=content,
    )


def _extract_title(content: str, readme_path: Path) -> tuple[str, float]:
    """
    Extract the assignment title and total points from README content.

    Looks for the first H1 heading with total points, falling back to the directory name.

    Args:
        content: README markdown content.
        readme_path: Path to the README file.

    Returns:
        Tuple of (assignment title, total points from title or 0).
    """
    # Look for first H1 heading with optional point value
    # Pattern: # Title (40) or # Title (40 pts)
    title_match = re.search(
        r"^#\s+(.+?)\s*(?:\((\d+)\s*(?:pts|points)?\s*\))?$",
        content,
        re.MULTILINE,
    )
    if title_match:
        title = title_match.group(1).strip()
        points = float(title_match.group(2)) if title_match.group(2) else 0
        return title, points

    # Fallback to parent directory name
    return readme_path.parent.name, 0


def _extract_sections(content: str, exclude_title: str = "") -> list[RubricSection]:
    """
    Extract all grading sections from README content.

    Parses headings matching the pattern `# Section Name (X pts)`
    and extracts any function signatures from following code blocks.

    Args:
        content: README markdown content.
        exclude_title: Title to exclude from sections (typically the main heading).

    Returns:
        List of RubricSection objects.
    """
    sections: list[RubricSection] = []
    pattern = re.compile(SECTION_PATTERN, re.MULTILINE | re.IGNORECASE)

    # Split content by headings to associate descriptions with sections
    heading_positions = list(pattern.finditer(content))

    for i, match in enumerate(heading_positions):
        name = match.group(1).strip()

        # Skip the title heading
        if exclude_title and name.lower() == exclude_title.lower():
            continue

        points = float(match.group(2))

        # Check if this is extra credit
        is_extra = _is_extra_credit(name)

        # Extract section content (text between this heading and next)
        start_pos = match.end()
        end_pos = heading_positions[i + 1].start() if i + 1 < len(heading_positions) else len(content)
        section_content = content[start_pos:end_pos]

        # Extract description (first paragraph after heading)
        description = _extract_description(section_content)

        # Extract expected function names from code blocks
        expected_functions = _extract_function_signatures(section_content)

        sections.append(
            RubricSection(
                name=name,
                points=points,
                is_extra=is_extra,
                description=description,
                expected_functions=expected_functions,
            )
        )

    return sections


def _is_extra_credit(section_name: str) -> bool:
    """
    Determine if a section is extra credit based on its name.

    Args:
        section_name: The section title.

    Returns:
        True if the section appears to be extra credit.
    """
    name_lower = section_name.lower()
    return any(keyword in name_lower for keyword in EXTRA_CREDIT_KEYWORDS)


def _extract_description(section_content: str) -> str:
    """
    Extract the description paragraph from section content.

    Gets the first non-empty paragraph after the heading.

    Args:
        section_content: Content between section heading and next heading.

    Returns:
        Description string (first 500 chars max).
    """
    # Remove code blocks to avoid extracting code as description
    content_no_code = re.sub(r"```[\s\S]*?```", "", section_content)

    # Split into paragraphs and get first non-empty one
    paragraphs = content_no_code.strip().split("\n\n")
    for para in paragraphs:
        cleaned = para.strip()
        # Skip numbered lists and short lines
        if cleaned and len(cleaned) > 20 and not cleaned.startswith(("1.", "2.", "3.", "-", "*")):
            # Limit length
            return cleaned[:500] + ("..." if len(cleaned) > 500 else "")

    return ""


def _extract_function_signatures(section_content: str) -> list[str]:
    """
    Extract expected function names from Python code blocks.

    Looks for `def function_name` patterns in code blocks.

    Args:
        section_content: Content between section heading and next heading.

    Returns:
        List of function names found in code blocks.
    """
    functions: list[str] = []

    # Find all Python code blocks
    code_blocks = re.findall(r"```python\s*([\s\S]*?)```", section_content, re.IGNORECASE)

    for block in code_blocks:
        # Extract function definitions
        func_matches = re.findall(r"def\s+(\w+)\s*\(", block)
        functions.extend(func_matches)

    # Also look for function names mentioned in text like "called `function_name`"
    inline_funcs = re.findall(r"called\s+[`'](\w+)[`']", section_content)
    functions.extend(inline_funcs)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_functions: list[str] = []
    for func in functions:
        if func not in seen:
            seen.add(func)
            unique_functions.append(func)

    return unique_functions


def format_rubric_for_llm(rubric: Rubric) -> str:
    """
    Format the rubric as a string for LLM context.

    Creates a structured representation of the rubric
    suitable for including in LLM prompts.

    Args:
        rubric: Parsed Rubric object.

    Returns:
        Formatted string representation.
    """
    lines = [
        f"# Assignment: {rubric.title}",
        f"Total Points: {rubric.total_points}",
        "",
        "## Grading Sections:",
    ]

    for section in rubric.sections:
        extra_tag = " [EXTRA CREDIT]" if section.is_extra else ""
        lines.append(f"\n### {section.name} ({section.points} pts){extra_tag}")

        if section.description:
            lines.append(f"Description: {section.description}")

        if section.expected_functions:
            lines.append(f"Expected Functions: {', '.join(section.expected_functions)}")

    return "\n".join(lines)

