"""Command-line interface for AgentScaffold."""

import os
import sys
import subprocess
from typing import Optional

# Use try/except for all imports that might not be available
try:
    import typer
except ImportError:
    sys.exit("Error: typer package is required. Install with: pip install typer")

# Optional fancy UI components - gracefully degrade if not available
try:
    from halo import Halo
    HAS_HALO = True
except ImportError:
    HAS_HALO = False
    # Simple fallback spinner class
    class Halo:
        def __init__(self, text=None, spinner=None):
            self.text = text

        def start(self):
            print(f"{self.text}...")
            return self

        def succeed(self, text):
            print(f"✅ {text}")

        def fail(self, text):
            print(f"❌ {text}")

try:
    from agentscaffold.scaffold import create_new_agent
except ImportError:
    sys.exit("Error: Cannot import agentscaffold module. Make sure it's installed properly.")

app = typer.Typer(help="AgentScaffold CLI for creating and managing AI agents")

def run_command_with_spinner(command, cwd, start_message, success_message, error_message):
    """Helper function to run a command with a loading spinner."""
    if HAS_HALO:
        spinner = Halo(text=start_message, spinner='dots')
        spinner.start()
    else:
        print(start_message)
        spinner = Halo(text=start_message)  # Using our fallback

    try:
        subprocess.run(command, cwd=cwd, check=True, capture_output=True)
        spinner.succeed(success_message)
    except subprocess.CalledProcessError as e:
        spinner.fail(error_message)
        typer.echo(f"Error details: {e.stderr.decode() if e.stderr else e}")
        raise typer.Exit(1)
    except FileNotFoundError:
        spinner.fail(f"Error: Command not found: {command[0]}")
        raise typer.Exit(1)

@app.command()
def new(
    name: str = typer.Option(..., prompt="Enter agent name", help="Name of the agent to create"),
    template: str = typer.Option("basic", help="Template to use"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to output the agent (default: current directory)"),
    skip_install: bool = typer.Option(False, help="Skip installing dependencies"),
):
    """
    Create a new agent with the specified name and template.
    """
    if output_dir is None:
        output_dir = os.getcwd()

    agent_dir = os.path.join(output_dir, name)

    # Fix: use typer.style() instead of trying to call color constants
    styled_name = typer.style(name, fg=typer.colors.BRIGHT_WHITE, bold=True)
    styled_template = typer.style(template, fg=typer.colors.CYAN)
    typer.echo(f"✨ Creating new agent '{styled_name}' using template '{styled_template}'...")

    create_new_agent(name, template, output_dir)

    # Check if UV is installed
    has_uv = False
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        has_uv = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if has_uv and not skip_install:
        typer.echo(f"🛠️ Setting up virtual environment with {typer.style('UV', fg=typer.colors.CYAN)}...")
        run_command_with_spinner(
            command=["uv", "venv", ".venv"],
            cwd=agent_dir,
            start_message="Creating virtual environment...",
            success_message=f"✅ Virtual environment created with {typer.style('UV', fg=typer.colors.CYAN)}",
            error_message=f"❌ Failed to set up virtual environment with {typer.style('UV', fg=typer.colors.CYAN)}"
        )

        # Install local AgentScaffold package
        agent_scaffold_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        run_command_with_spinner(
            command=["uv", "pip", "install", "-e", agent_scaffold_dir],
            cwd=agent_dir,
            start_message="Installing local AgentScaffold package...",
            success_message=f"✅ Local AgentScaffold package installed",
            error_message=f"⚠️ Warning: Failed to install local AgentScaffold package"
        )

        # Install agent package
        run_command_with_spinner(
            command=["uv", "pip", "install", "-e", "."],
            cwd=agent_dir,
            start_message="Installing agent package...",
            success_message=f"✅ Agent package installed",
            error_message=f"⚠️ Warning: Failed to install agent package"
        )
    elif not skip_install:
        typer.echo(f"🛠️ Setting up virtual environment with {typer.style('venv', fg=typer.colors.YELLOW)} and {typer.style('pip', fg=typer.colors.YELLOW)}...")
        run_command_with_spinner(
            command=["python", "-m", "venv", ".venv"],
            cwd=agent_dir,
            start_message="Creating virtual environment...",
            success_message=f"✅ Virtual environment created with {typer.style('venv', fg=typer.colors.YELLOW)}",
            error_message=f"⚠️ Warning: Failed to set up virtual environment with {typer.style('venv', fg=typer.colors.YELLOW)}"
        )
        venv_activate_command = ".venv\\Scripts\\activate" if sys.platform == "win32" else "source .venv/bin/activate"

        # Install agent package using pip (after venv setup)
        run_command_with_spinner(
            command=[sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=agent_dir,
            start_message="Installing agent package with pip...",
            success_message=f"✅ Agent package installed with {typer.style('pip', fg=typer.colors.YELLOW)}",
            error_message=f"⚠️ Warning: Failed to install agent package with {typer.style('pip', fg=typer.colors.YELLOW)}"
        )

    typer.echo(f"🎉 Agent '{typer.style(name, bold=True)}' created successfully! 🎉")
    typer.echo("\nNext steps:")
    typer.echo(f"  cd {name}")
    if not skip_install:
        typer.echo(f"  # Activate the virtual environment:")
        activate_cmd = venv_activate_command if not has_uv else '.venv/bin/activate' if sys.platform != 'win32' else '.venv\\Scripts\\activate'
        typer.echo(f"  {typer.style(activate_cmd, fg=typer.colors.CYAN)}")
        typer.echo(f"  # Install dependencies if not already done:")
        install_cmd = 'uv pip install -e .' if has_uv else 'pip install -e .'
        typer.echo(f"  {typer.style(install_cmd, fg=typer.colors.CYAN)}")
    else:
        typer.echo(f"  # Setup virtual environment and install dependencies if skipped:")
        typer.echo(f"  python -m venv .venv")
        if sys.platform == "win32":
            typer.echo(f"  .venv\\Scripts\\activate")
        else:
            typer.echo(f"  source .venv/bin/activate")
        typer.echo(f"  pip install -e .")

    typer.echo(f"  # Run the agent:")
    typer.echo(f"  {typer.style('python main.py', fg=typer.colors.CYAN)}")


@app.command()
def run(
    agent_dir: Optional[str] = typer.Option(".", help="Directory containing the agent"),
):
    """
    Run an agent in the specified directory.
    """
    typer.echo(f"🚀 Running agent in '{typer.style(agent_dir, bold=True)}'...")

    # Check if main.py exists
    main_py = os.path.join(agent_dir, "main.py")
    if not os.path.exists(main_py):
        typer.echo(f"Error: {typer.style(main_py, fg=typer.colors.RED)} not found.")
        raise typer.Exit(1)

    # Run the agent
    try:
        subprocess.run([sys.executable, main_py], check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"🔥 Error running agent: {typer.style(str(e), fg=typer.colors.RED)}")
        raise typer.Exit(1)


@app.command()
def version():
    """Display the current version of AgentScaffold."""
    from agentscaffold import __version__
    typer.echo(f"✨ AgentScaffold v{typer.style(__version__, fg=typer.colors.CYAN)}")


if __name__ == "__main__":
    app()