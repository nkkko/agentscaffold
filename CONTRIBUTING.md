# Contributing to AgentScaffold

First, thank you for considering contributing to AgentScaffold! This document provides guidelines and instructions to help you contribute effectively to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Setup](#development-setup)
  - [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Commit Messages](#commit-messages)
  - [Pull Requests](#pull-requests)
- [Coding Standards](#coding-standards)
  - [Code Style](#code-style)
  - [Documentation](#documentation)
  - [Testing](#testing)
  - [Type Annotations](#type-annotations)
- [Adding Features](#adding-features)
  - [Adding CLI Commands](#adding-cli-commands)
  - [Adding Agent Templates](#adding-agent-templates)
- [Release Process](#release-process)
- [Community](#community)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We expect all contributors to adhere to this code to ensure a welcoming and respectful environment for everyone.

## Getting Started

### Development Setup

1. **Fork and Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/AgentScaffold.git
   cd AgentScaffold
   ```

2. **Set Up Development Environment**

   With UV (recommended):
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

   With pip:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Configure Daytona (Optional for Remote Testing)**

   ```bash
   # Get a Daytona API key
   daytona api-key generate

   # Create .env file with the following variables
   echo "DAYTONA_API_KEY=your-daytona-api-key" >> .env
   echo "DAYTONA_API_URL=your-daytona-server-url" >> .env
   echo "DAYTONA_TARGET=us" >> .env
   ```

### Project Structure

```
agentscaffold/
├── agentscaffold/           # Main package
│   ├── __init__.py          # Package initialization
│   ├── agent.py             # Agent base classes
│   ├── cli.py               # Command-line interface
│   ├── scaffold.py          # Agent scaffolding logic
│   └── templates/           # Templates for new agents
│       └── basic/           # Basic agent template
│           ├── README.md.jinja
│           ├── main.py.jinja
│           ├── pyproject.toml.jinja
│           ├── requirements.in.jinja
│           └── {{package_name}}/
│               ├── __init__.py.jinja
│               └── agent.py.jinja
├── tests/                   # Test directory
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_cli.py
│   ├── test_installation.py
│   └── test_scaffold.py
├── pyproject.toml           # Project metadata and dependencies
├── README.md                # Project documentation
└── CODE_OF_CONDUCT.md       # Code of conduct
```

## Development Workflow

### Branching Strategy

- `main` - Main development branch, should always be in a deployable state
- `feature/feature-name` - For new features
- `fix/issue-name` - For bug fixes
- `chore/task-name` - For maintenance tasks
- `docs/change-name` - For documentation updates

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

Signed-off-by: Your Name <your.email@example.com>
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Example:
```
feat(cli): add support for custom agent templates

Adds the ability to specify custom templates when creating new agents.
The templates can be loaded from a local directory or from a git repository.

Closes #123

Signed-off-by: Jane Doe <jane.doe@example.com>
```

### Signing Off Commits

All commits must be signed off to certify that you have the right to contribute the code. This is done by adding a `Signed-off-by` line to your commit message:

```bash
# Sign off automatically with git commit
git commit -s -m "your commit message"
```

This adds a line to your commit message like:
```
Signed-off-by: Your Name <your.email@example.com>
```

By signing off, you certify that:

1. You have the right to submit the work under the project's license
2. You created the work (or have appropriate rights to submit it)
3. You agree to the Developer Certificate of Origin (DCO)

Make sure your Git user name and email are set correctly:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Pull Requests

1. Create a branch for your changes
2. Make your changes with appropriate tests
3. Run tests locally to ensure they pass
4. Update documentation as needed
5. Submit a pull request to the `main` branch
6. Wait for code review and address any feedback

## Coding Standards

### Code Style

- Follow PEP 8 and use Black for formatting with a line length of 88 characters
- Use Ruff for linting
- Run formatters and linters before submitting:

  ```bash
  # Format code
  black .

  # Lint code
  ruff check .
  
  # Type check
  mypy agentscaffold/
  ```

### Documentation

- All modules, classes, and functions should have docstrings
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep the README.md updated with current features and usage examples
- Document any breaking changes in your pull request description

Example docstring:
```python
def create_agent(name: str, template: str = "basic") -> bool:
    """
    Create a new agent with the specified name and template.

    Args:
        name: The name of the agent to create
        template: The template to use for the agent (default: "basic")

    Returns:
        True if the agent was created successfully, False otherwise
    
    Raises:
        ValueError: If the name is invalid or if the template doesn't exist
    """
```

### Testing

- Write tests for all new functionality
- Aim for high test coverage
- Tests should be fast and not depend on external services when possible
- Run tests before submitting:

  ```bash
  # Run all tests
  pytest

  # Run specific test file
  pytest tests/test_scaffold.py

  # Run with coverage report
  pytest --cov=agentscaffold
  ```

### Type Annotations

- Use type annotations for all function definitions
- Use `Optional` for parameters that can be `None`
- Use `Any` sparingly and only when absolutely necessary
- Use `TypeVar` for generic types

## Adding Features

### Adding CLI Commands

CLI commands are defined in `agentscaffold/cli.py` using Typer. To add a new command:

1. Add a new function with the appropriate Typer decorators
2. Implement the command logic
3. Add tests in `tests/test_cli.py`
4. Update the documentation

Example:
```python
@app.command()
def validate(
    agent_path: str = typer.Argument(
        ".", help="Path to the agent directory"
    )
) -> None:
    """Validate an agent's structure and configuration."""
    # Implementation
```

### Adding Agent Templates

To add a new agent template:

1. Create a new directory in `agentscaffold/templates/`
2. Add the necessary template files with `.jinja` extension
3. Update the `scaffold.py` file to support the new template
4. Add tests for the new template
5. Update documentation

## Release Process

1. Update version in `pyproject.toml`
2. Create a new release branch: `release/vX.Y.Z`
3. Update CHANGELOG.md with the new version and changes
4. Submit a pull request from the release branch to `main`
5. After merging, tag the release on GitHub: `vX.Y.Z`
6. Build and publish to PyPI:

   ```bash
   python -m build
   python -m twine upload dist/*
   ```

## Community

- Join discussions in GitHub Issues
- Help answer questions from other users
- Share examples of how you use AgentScaffold
- Suggest improvements or report bugs

---

Thank you for contributing to AgentScaffold! Your efforts help make this project better for everyone.