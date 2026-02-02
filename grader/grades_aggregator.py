"""
Grades aggregator for collecting and exporting all student grades.

Saves grades to a centralized folder with JSON and CSV summaries.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from .config import (
    DEFAULT_GRADES_DIR,
    GRADES_CSV_FILENAME,
    GRADES_SUMMARY_FILENAME,
)
from .models import GradeResult


class GradesAggregator:
    """
    Aggregates grades from multiple students and exports to various formats.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        """
        Initialize the grades aggregator.

        Args:
            output_dir: Directory to save aggregated grades. Defaults to ./grades/
        """
        self.output_dir = output_dir or DEFAULT_GRADES_DIR
        self.grades: list[GradeResult] = []
        self.timestamp = datetime.now().isoformat()

    def add_grade(self, grade: GradeResult) -> None:
        """
        Add a grade result to the aggregator.

        Args:
            grade: GradeResult to add.
        """
        self.grades.append(grade)

    def save_all(self) -> dict[str, Path]:
        """
        Save all grades to the output directory.

        Creates:
        - Individual JSON files per student
        - Summary JSON with all grades
        - Summary CSV for easy import to gradebook

        Returns:
            Dictionary of output file paths.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_files: dict[str, Path] = {}

        # Sort grades by github_repo (handle None)
        self.grades.sort(key=lambda x: (x.github_repo or "", x.student_id))

        # Save individual JSON files
        for grade in self.grades:
            individual_path = self.output_dir / f"{grade.student_id}.json"
            with open(individual_path, "w", encoding="utf-8") as f:
                f.write(grade.model_dump_json(indent=2))
            output_files[grade.student_id] = individual_path

        # Save summary JSON
        summary_path = self.output_dir / GRADES_SUMMARY_FILENAME
        summary_data = {
            "timestamp": self.timestamp,
            "total_students": len(self.grades),
            "statistics": self._calculate_statistics(),
            "grades": [grade.model_dump() for grade in self.grades],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        output_files["summary_json"] = summary_path

        # Save CSV
        csv_path = self.output_dir / GRADES_CSV_FILENAME
        self._save_csv(csv_path)
        output_files["summary_csv"] = csv_path

        return output_files

    def _calculate_statistics(self) -> dict:
        """
        Calculate summary statistics for all grades.

        Returns:
            Dictionary with statistics.
        """
        if not self.grades:
            return {}

        scores = [g.total_score for g in self.grades]
        max_scores = [g.max_score for g in self.grades]
        passed = sum(1 for g in self.grades if g.code_execution_passed)

        return {
            "average_score": sum(scores) / len(scores),
            "max_possible": max_scores[0] if max_scores else 0,
            "highest_score": max(scores),
            "lowest_score": min(scores),
            "tests_passed_count": passed,
            "tests_passed_percent": (passed / len(self.grades)) * 100,
        }

    def _save_csv(self, csv_path: Path) -> None:
        """
        Save grades as CSV file.

        Args:
            csv_path: Path to save CSV file.
        """
        if not self.grades:
            return

        # Get all section names from first grade
        section_names = [s.section_name for s in self.grades[0].sections]

        # Build header
        header = ["student_id", "total_score", "max_score", "percentage", "tests_passed", "github_repo"]
        header.extend(section_names)
        header.append("overall_feedback")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for grade in self.grades:
                row = [
                    grade.student_id,
                    grade.total_score,
                    grade.max_score,
                    f"{(grade.total_score / grade.max_score * 100):.1f}%" if grade.max_score > 0 else "0%",
                    "Yes" if grade.code_execution_passed else "No",
                    grade.github_repo or "",
                ]
                # Add section scores
                for section in grade.sections:
                    row.append(f"{section.points_earned}/{section.max_points}")
                row.append(grade.overall_feedback[:200])  # Truncate feedback
                writer.writerow(row)


def load_grades_from_dir(grades_dir: Path) -> list[GradeResult]:
    """
    Load all grade results from a grades directory.

    Args:
        grades_dir: Path to the grades directory.

    Returns:
        List of GradeResult objects.
    """
    grades: list[GradeResult] = []

    # Try loading from summary file first
    summary_path = grades_dir / GRADES_SUMMARY_FILENAME
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for grade_data in data.get("grades", []):
                grades.append(GradeResult(**grade_data))
        # Sort by github_repo before returning
        grades.sort(key=lambda x: (x.github_repo or "", x.student_id))
        return grades

    # Fall back to individual files
    for json_file in grades_dir.glob("*.json"):
        if json_file.name == GRADES_SUMMARY_FILENAME:
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                grades.append(GradeResult(**data))
        except Exception:
            pass

    # Sort by github_repo before returning
    grades.sort(key=lambda x: (x.github_repo or "", x.student_id))
    return grades

