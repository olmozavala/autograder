# Grader Pod: Automated Grading System

Grader Pod is an advanced automated grading tool designed to streamline the evaluation of student programming assignments. It combines traditional unit testing with Large Language Model (LLM) analysis to provide comprehensive feedback on both code correctness and report quality.

## Features

- **Automated Code Testing**: Executes student submissions against a defined test suite (using `pytest`).
- **LLM-Powered Grading**: Utilizes OpenAI's models to grade written reports, analyze figures, and provide qualitative feedback based on a rubric.
- **Dynamic Rubric Parsing**: Automatically extracts grading criteria and point values directly from the assignment's `README.md`.
- **Interactive Dashboard**: Visualizes class performance, individual student scores, and detailed feedback through a web-based dashboard.
- **Configurable Pipeline**: flexible configuration via `grader_config.yml` to support various assignment structures.
- **Secure Execution**: Support for running student code in isolated environments (configurable).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd autograder
    ```

2.  **Install Dependencies**:
    This project uses standard Python packages. It is recommended to use a virtual environment (e.g., `uv`, `venv`, `conda`).

    ```bash
    # Basic dependencies
    pip install -r grader/requirements.txt
    
    # Dashboard dependencies (required for visualization)
    pip install dash pandas plotly flask
    ```

## Configuration

The system is configured using `grader_config.yml`. You can specify paths to your submissions, tests, and assignment details here.

**Example `grader_config.yml`:**

```yaml
# Grader Configuration

# Source paths (templates, tests)
source_path: "/path/to/assignment/template"

# Dependent paths (relative to source_path)
readme_path: "README.md"       # Contains the rubric
tests_dir: "tests"             # Shared tests folder
test_data_dir: "test_folder"   # Optional data folder for tests

# Submission paths
submissions_dir: "/path/to/student/submissions"
grades_dir: "./grades"         # Output directory for results

# Execution flags
skip_llm: false        # Set to true to skip AI grading (tests only)
only_dashboard: false  # Set to true to skip grading and view existing results
verbose: true

# Dashboard settings
dashboard_port: 8050
```

## Usage

### Running the Grader

To start the grading pipeline, simply run `main.py`. The script will look for `grader_config.yml` in the current directory by default.

```bash
python main.py
```

You can also specify a different config file or override arguments via CLI:

```bash
python main.py --config my_config.yml --skip-llm
```

### Viewing Results

After grading is complete, the interactive dashboard will launch automatically (unless configured otherwise). You can browse:
- **Class Overview**: Statistics like average score, highest/lowest scores.
- **Score Distributions**: Histograms and box plots of scores per section.
- **Individual Grades**: Detailed breakdown of points and feedback for each student.
- **Student Reports**: View the rendered markdown report for each student.

### Launching Dashboard Only

If you have already run the grader and just want to view the results, you can launch the dashboard directly without re-running the grading pipeline:

```bash
python main.py --only-dashboard
```
(Or set `only_dashboard: true` in your configuration file).

## Project Structure

- `main.py`: Entry point for the grading CLI.
- `grader/`: Core logic package.
    - `dashboard.py`: Dash application for visualization.
    - `llm_grader.py`: Interface for LLM-based grading.
    - `rubric_parser.py`: Logic to parse rubrics from markdown.
    - `local_runner.py`: Handles code execution (local/docker).
    - `grades_aggregator.py`: Collects and saves results.
- `grader_config.yml`: Main configuration file.

## Requirements

- Python 3.10+
- OpenAI API Key (set as environment variable `OPENAI_API_KEY`) for LLM grading.
- `pdflatex` (optional, if compiling PDFs).
