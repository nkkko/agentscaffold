"""Command-line interface for AgentScaffold."""

import os
import typer
from typing import Optional
import subprocess
import sys
import pathlib

from agentscaffold.scaffold import create_new_agent

app = typer.Typer(help="AgentScaffold CLI for creating and managing AI agents")


@app.command()
def new(
    name: str,
    template: Optional[str] = "basic",
    output_dir: Optional[str] = None,
    skip_install: bool = False,
):
    """
    Create a new agent with the specified name and template.
    
    Args:
        name: Name of the agent to create
        template: Template to use (default: basic)
        output_dir: Directory to output the agent (default: current directory)
        skip_install: Skip installing dependencies
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
    
    if has_uv and not skip_install:
        typer.echo("Setting up virtual environment with UV...")
        try:
            subprocess.run(["uv", "venv", ".venv"], cwd=agent_dir, check=True)
            
            # Install the local AgentScaffold package first if it exists
            agent_scaffold_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            try:
                typer.echo("Installing local AgentScaffold package...")
                subprocess.run(["uv", "pip", "install", "-e", agent_scaffold_dir], 
                              cwd=agent_dir, check=True)
            except subprocess.CalledProcessError:
                typer.echo("Warning: Failed to install local AgentScaffold package")
            
            # Now install the agent package itself
            typer.echo("Installing agent package...")
            try:
                subprocess.run(["uv", "pip", "install", "-e", "."], 
                              cwd=agent_dir, check=True)
            except subprocess.CalledProcessError:
                typer.echo("Warning: Failed to install agent package")
        except subprocess.CalledProcessError:
            typer.echo("Warning: Failed to set up virtual environment")
    
    typer.echo(f"✅ Agent '{name}' created successfully!")
    typer.echo(f"To run your agent:")
    typer.echo(f"  cd {name}")
    
    if has_uv and not skip_install:
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
    local: bool = False,
    message: Optional[str] = None,
):
    """
    Run an agent in the specified directory, using Daytona for remote execution.
    
    Args:
        agent_dir: Directory containing the agent (default: current directory)
        local: Run locally instead of using Daytona remote execution
        message: Optional message to send to the agent
    """
    # Get absolute path to agent directory
    agent_dir = os.path.abspath(agent_dir)
    typer.echo(f"Running agent in '{agent_dir}'...")
    
    # Check if main.py exists
    main_py = os.path.join(agent_dir, "main.py")
    if not os.path.exists(main_py):
        typer.echo(f"Error: {main_py} not found.")
        raise typer.Exit(1)
    
    if local:
        # Run the agent locally
        typer.echo("Running agent locally...")
        try:
            subprocess.run([sys.executable, main_py], check=True)
        except subprocess.CalledProcessError as e:
            typer.echo(f"Error running agent: {e}")
            raise typer.Exit(1)
    else:
        # Run the agent in Daytona
        typer.echo("Running agent in Daytona remote sandbox...")
        
        try:
            # Try to load environment variables from .env file
            try:
                from dotenv import load_dotenv
                env_file = os.path.join(agent_dir, ".env")
                if os.path.exists(env_file):
                    typer.echo(f"Loading environment variables from {env_file}")
                    load_dotenv(env_file)
            except ImportError:
                typer.echo("Warning: python-dotenv not installed. Environment variables may not be loaded.")
            
            # Import the DaytonaRuntime
            from agentscaffold.agent import DaytonaRuntime
            
            # Create a runtime instance
            runtime = DaytonaRuntime()
            
            # Prepare input data
            input_data = {"message": message or "Hello, agent!"}
            
            # Execute the agent
            result = runtime.execute(agent_dir, input_data)
            
            # Display the result
            if "error" in result.get("metadata", {}):
                typer.echo(f"Error: {result['response']}")
                raise typer.Exit(1)
            else:
                typer.echo("\nAgent Response:")
                typer.echo(f"  {result['response']}")
                
                if result.get("metadata"):
                    typer.echo("\nMetadata:")
                    for key, value in result["metadata"].items():
                        typer.echo(f"  {key}: {value}")
                        
        except ImportError as e:
            typer.echo(f"Error: {e}")
            typer.echo("Make sure daytona-sdk is installed. Run: pip install daytona-sdk")
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Error running agent in Daytona: {e}")
            raise typer.Exit(1)


@app.command()
def version():
    """Display the current version of AgentScaffold."""
    from agentscaffold import __version__
    typer.echo(f"AgentScaffold v{__version__}")


if __name__ == "__main__":
    app()