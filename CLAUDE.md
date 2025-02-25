# AgentScaffold Project Guide

## Project Commands
- Install: `pip install -e .`
- Test: `pytest tests/`
- Test single file: `pytest tests/test_filename.py`
- Lint: `ruff check .`
- Format: `black .`
- Type Check: `mypy agentscaffold/`

## Code Style Guidelines
- **Formatting**: Use Black with 88 character line length
- **Imports**: Group imports (standard library first, then third-party, then local)
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Types**: Use explicit type annotations with mypy compatibility
- **Error Handling**: Use specific exceptions and appropriate error messages
- **Documentation**: Use docstrings for modules, classes, and functions
- **Testing**: Write pytest tests for all functionality

## CLI Usage
- Create new agent: `agentscaffold new my-agent`
- Run agent: `agentscaffold run`
- Check version: `agentscaffold version`

## UV Support
AgentScaffold supports uv for fast dependency management. If uv is installed, the scaffolding will automatically create a uv-compatible virtual environment.