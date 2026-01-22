"""
Configuration loader for the Grader Pod system.

Handles parsing and validation of YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class GraderConfig(BaseModel):
    """
    Configuration model for the grader.
    """
    submissions_dir: Path = Field(..., description="Path to directory containing student submissions")
    source_path: Optional[Path] = Field(None, description="Base path for assignment source files")
    readme_path: Path = Field(..., description="Path to the assignment README.md with rubric")
    tests_dir: Optional[Path] = Field(None, description="Path to shared test files")
    test_data_dir: Optional[Path] = Field(None, description="Path to test data folder")
    grades_dir: Optional[Path] = Field(None, description="Path to save aggregated grades")
    
    # Flags can also be configured
    skip_llm: bool = Field(False, description="Skip LLM grading")
    only_dashboard: bool = Field(False, description="Launch dashboard with existing grades (skip grading)")
    dashboard_port: int = Field(8050, description="Port for the dashboard")
    verbose: bool = Field(False, description="Enable verbose output")


def load_config(config_path: Path) -> GraderConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        GraderConfig object with loaded values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
        ValidationError: If config data is invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if not config_data:
        return GraderConfig() # Return empty/defaults if file is empty but valid? No, required fields will fail.

    # Resolve relative paths relative to the config file location
    config_dir = config_path.parent
    
    # First resolve source_path if present
    source_base = config_dir
    if "source_path" in config_data and config_data["source_path"]:
        source_path = Path(config_data["source_path"])
        if not source_path.is_absolute():
            source_path = config_dir / source_path
        config_data["source_path"] = source_path
        source_base = source_path

    # Resolve paths relative to source_base
    for path_field in ["readme_path", "tests_dir", "test_data_dir"]:
        if path_field in config_data and config_data[path_field]:
            path = Path(config_data[path_field])
            if not path.is_absolute():
                config_data[path_field] = source_base / path
    
    # Resolve submissions and grades relative to config dir (unless absolute)
    for path_field in ["submissions_dir", "grades_dir"]:
        if path_field in config_data and config_data[path_field]:
            path = Path(config_data[path_field])
            if not path.is_absolute():
                config_data[path_field] = config_dir / path

    return GraderConfig(**config_data)
