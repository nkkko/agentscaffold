"""Scaffolding functionality for creating agents."""

import os
import shutil
import jinja2
from pathlib import Path
from typing import Dict, Any, Optional


# Path to templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_new_agent(
    name: str,
    template: str = "basic",
    output_dir: Optional[str] = None,
) -> None:
    """
    Create a new agent with the specified name and template.

    Args:
        name: Name of the agent to create
        template: Template to use (default: basic)
        output_dir: Directory to output the agent (default: current directory)
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # Convert kebab-case to snake_case for the package name
    package_name = name.replace("-", "_")

    # Create agent directory
    agent_dir = os.path.join(output_dir, name)
    os.makedirs(agent_dir, exist_ok=True)

    # Create package directory
    package_dir = os.path.join(agent_dir, package_name)
    os.makedirs(package_dir, exist_ok=True)

    # Create template context
    context = {
        "agent_name": name,
        "package_name": package_name,
        "agent_class_name": "".join(x.capitalize() for x in package_name.split("_")),
    }

    # Copy and render template files from the template directory
    template_dir = TEMPLATES_DIR / template

    # Define template files to render
    template_files = [
        "README.md.jinja",
        "main.py.jinja",
        "requirements.in.jinja", 
        "pyproject.toml.jinja"
    ]
    
    # Create .env.example content
    env_example_content = """# Configure your API keys and other settings here
# Uncomment and add your API keys

# For LLM access
# OPENAI_API_KEY=your-api-key

# For Daytona remote execution (required for 'agentscaffold run')
# DAYTONA_API_KEY=your-daytona-api-key 
# DAYTONA_SERVER_URL=your-daytona-server-url
# DAYTONA_TARGET=us
"""

    # Handle basic files first
    for file_name in template_files:
        template_file_path = os.path.join(template_dir, file_name)
        if os.path.exists(template_file_path):
            _render_template_file(template_file_path, agent_dir, context)
    
    # Create .env.example file
    env_example_path = os.path.join(agent_dir, '.env.example')
    with open(env_example_path, 'w') as f:
        f.write(env_example_content)

    # Handle package files
    pkg_template_dir = template_dir / "{{package_name}}"
    if pkg_template_dir.exists():
        for file_name in os.listdir(pkg_template_dir):
            if file_name.endswith(".jinja"):
                pkg_file_path = os.path.join(pkg_template_dir, file_name)
                _render_template_file(pkg_file_path, package_dir, context, is_package_file=True)

    # Special handling for .env.example - make a copy as .env
    env_path = os.path.join(agent_dir, '.env')
    if os.path.exists(env_example_path) and not os.path.exists(env_path):
        shutil.copy(env_example_path, env_path)


def _render_template_file(
    template_path: str,
    output_dir: str,
    context: Dict[str, Any],
    is_package_file: bool = False,
) -> None:
    """
    Render a single template file and write it to the output directory.

    Args:
        template_path: Path to the template file
        output_dir: Directory to output the rendered file
        context: Template context
        is_package_file: Whether the file is in the package directory
    """
    # Read the template file
    with open(template_path, 'r') as f:
        template_content = f.read()

    # Render the template
    env = jinja2.Environment()
    template = env.from_string(template_content)
    rendered_content = template.render(**context)

    # Determine the output file path
    file_name = os.path.basename(template_path)
    if file_name.endswith('.jinja'):
        file_name = file_name[:-6]  # Remove .jinja extension

    output_file_path = os.path.join(output_dir, file_name)

    # Write the rendered content to the output file
    with open(output_file_path, 'w') as f:
        f.write(rendered_content)