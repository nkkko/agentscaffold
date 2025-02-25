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
    
    # Copy and render template files
    template_dir = TEMPLATES_DIR / template
    _render_template_directory(template_dir, agent_dir, context)


def _render_template_directory(
    template_dir: Path,
    output_dir: str,
    context: Dict[str, Any],
) -> None:
    """
    Recursively render all template files in a directory.
    
    Args:
        template_dir: Directory containing templates
        output_dir: Directory to output rendered files
        context: Template context
    """
    # For now, we'll mock this since we don't have actual templates yet
    # In a real implementation, this would render Jinja2 templates
    
    # Set up Jinja environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
    
    # Get all files in the template directory
    template_files = []
    for root, _, files in os.walk(template_dir):
        rel_path = os.path.relpath(root, template_dir)
        for file in files:
            if file.endswith('.jinja'):
                template_path = os.path.join(rel_path, file)
                if template_path.startswith('.'):
                    template_path = template_path[1:]
                template_files.append(template_path)
    
    # Render each template file
    for template_path in template_files:
        # Handle special case for package name directory
        if '{{package_name}}' in template_path:
            output_path = template_path.replace('{{package_name}}', context['package_name'])
        else:
            output_path = template_path
        
        # Remove .jinja extension
        if output_path.endswith('.jinja'):
            output_path = output_path[:-6]
        
        # Create directories if needed
        output_file_path = os.path.join(output_dir, output_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Render template
        template = env.get_template(template_path)
        rendered_content = template.render(**context)
        
        # Write output file
        with open(output_file_path, 'w') as f:
            f.write(rendered_content)
    
    # Special handling for .env.example - make a copy as .env
    env_example_path = os.path.join(output_dir, '.env.example')
    env_path = os.path.join(output_dir, '.env')
    if os.path.exists(env_example_path) and not os.path.exists(env_path):
        shutil.copy(env_example_path, env_path)