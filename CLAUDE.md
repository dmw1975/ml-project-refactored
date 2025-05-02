# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Run all tests: `python test_setup.py && python test_features_data.py && python test_xgboost.py && python test_visualization.py && python test_sector_models.py`
- Run a single test: `python <test_file.py>` (e.g., `python test_xgboost.py`)
- Run with specific flags: `python main.py --<flag>` (common flags: `--train`, `--evaluate`, `--visualize`)
- Format code: `black .`
- Lint code: `flake8`

## Code Style Guidelines
- **Imports**: Standard library first, third-party packages second, local modules last
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Docstrings**: NumPy-style docstrings with parameters and return values documented
- **Formatting**: Follow PEP 8 guidelines, use black for formatting
- **Error Handling**: Use descriptive error messages, validate inputs, use try/except for anticipated errors
- **Project Structure**: Keep code organized in modules (data, models, evaluation, visualization)
- **File I/O**: Use utils.io module for file operations to ensure consistent handling
- **Paths**: Use pathlib.Path for path manipulation, reference paths from settings.py
- **Testing**: Each component should have a corresponding test file (test_*.py)