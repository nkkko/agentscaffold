"""Command-line interface for AgentScaffold."""

import os
import typer
from typing import Optional
import subprocess
import sys

from agentscaffold.scaffold import create_new_agent

app = typer.Typer(help="AgentScaffold CLI for creating and managing AI agents")


@app.command()
def new(
    name: str,
    template: Optional[str] = "basic",
    output_dir: Optional[str] = None,
):
    """
    Create a new agent with the specified name and template.
    
    Args:
        name: Name of the agent to create
        template: Template to use (default: basic)
        output_dir: Directory to output the agent (default: current directory)
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    typer.echo(f"Creating new agent '{name}' using template '{template}'...")
    create_new_agent(name, template, output_dir)
    
    # Full path to the created agent directory
    agent_dir = os.path.join(output_dir, name)
    
    # Check if UV is installed and set up the virtual environment
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        has_uv = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        has_uv = False
    
    if has_uv:
        typer.echo("Setting up virtual environment with UV...")
        subprocess.run(["uv", "venv", ".venv"], cwd=agent_dir, check=True)
        subprocess.run(["uv", "pip", "install", "-e", "."], cwd=agent_dir, check=True)
    
    typer.echo(f"✅ Agent '{name}' created successfully!")
    typer.echo(f"To run your agent:")
    typer.echo(f"  cd {name}")
    
    if has_uv:
        typer.echo(f"  # Activate the virtual environment")
        if sys.platform == "win32":
            typer.echo(f"  .venv\\Scripts\\activate")
        else:
            typer.echo(f"  source .venv/bin/activate")
    else:
        typer.echo(f"  # Create and activate a virtual environment")
        typer.echo(f"  python -m venv .venv")
        if sys.platform == "win32":
            typer.echo(f"  .venv\\Scripts\\activate")
        else:
            typer.echo(f"  source .venv/bin/activate")
        typer.echo(f"  pip install -e .")
    
    typer.echo(f"  # Run the agent")
    typer.echo(f"  python main.py")


@app.command()
def run(
    agent_dir: Optional[str] = ".",
):
    """
    Run an agent in the specified directory.
    
    Args:
        agent_dir: Directory containing the agent (default: current directory)
    """
    typer.echo(f"Running agent in '{agent_dir}'...")
    
    # Check if main.py exists
    main_py = os.path.join(agent_dir, "main.py")
    if not os.path.exists(main_py):
        typer.echo(f"Error: {main_py} not found.")
        raise typer.Exit(1)
    
    # Run the agent
    try:
        subprocess.run([sys.executable, main_py], check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running agent: {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Display the current version of AgentScaffold."""
    from agentscaffold import __version__
    typer.echo(f"AgentScaffold v{__version__}")


if __name__ == "__main__":
    app()