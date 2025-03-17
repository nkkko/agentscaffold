"""Command-line interface for AgentScaffold."""

import os
import sys
import subprocess
from typing import Optional, List

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
    from agentscaffold.scaffold import create_new_agent, get_agent_settings
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
    name: str = typer.Argument(None, help="Name of the agent to create"),
    template: str = typer.Option("basic", help="Template to use"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to output the agent (default: current directory)"),
    skip_install: bool = typer.Option(False, help="Skip installing dependencies"),
    interactive: bool = typer.Option(True, help="Interactive prompts for configuration"),
    llm_provider: Optional[str] = typer.Option(None, help="LLM provider (e.g., openai, anthropic, daytona)"),
    search_provider: Optional[str] = typer.Option("none", help="Search provider (e.g., brave, browserbase, none)"),
    memory_provider: Optional[str] = typer.Option("none", help="Memory provider (e.g., supabase, milvus, chromadb, none)"),
    logging_provider: Optional[str] = typer.Option("none", help="Logging provider (e.g., logfire, none)"),
    utilities: Optional[List[str]] = typer.Option(["dotenv"], help="Utility packages to include, comma-separated"),
):
    """
    Create a new agent with the specified name and template.
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # If name wasn't provided as an argument, prompt for it
    if name is None:
        name = typer.prompt("Enter agent name")
    
    # Skip the prompt in get_agent_settings if we have the name
    agent_dir = os.path.join(output_dir, name)

    # Fix: use typer.style() instead of trying to call color constants
    styled_name = typer.style(name, fg=typer.colors.BRIGHT_WHITE, bold=True)
    styled_template = typer.style(template, fg=typer.colors.CYAN)
    typer.echo(f"✨ Creating new agent '{styled_name}' using template '{styled_template}'...")

    # Use manual settings if interactive is False and providers are specified
    settings = None
    if not interactive:
        # Convert kebab-case to snake_case for the package name
        package_name = name.replace("-", "_")
        
        # Create settings from command-line arguments
        settings = {
            "agent_name": name,
            "package_name": package_name,
            "agent_class_name": "".join(x.capitalize() for x in package_name.split("_")),
            "description": f"An AI agent for {name}",
            "llm_provider": llm_provider or "daytona",
            "search_provider": search_provider or "none",
            "memory_provider": memory_provider or "none",
            "logging_provider": logging_provider or "none",
            "utilities": utilities or ["dotenv"],
        }
        
        # Add dependencies and env_vars
        from agentscaffold.scaffold import generate_dependencies, generate_env_vars
        settings["dependencies"] = generate_dependencies(
            settings["llm_provider"], 
            settings["search_provider"], 
            settings["memory_provider"], 
            settings["logging_provider"], 
            settings["utilities"]
        )
        settings["env_vars"] = generate_env_vars(
            settings["llm_provider"], 
            settings["search_provider"], 
            settings["memory_provider"], 
            settings["logging_provider"]
        )
    else:
        # Get settings interactively, but pass the name so it doesn't prompt again
        from agentscaffold.scaffold import get_agent_settings
        settings = get_agent_settings(name)

    # Create the agent with settings
    create_new_agent(name, template, output_dir, settings)

    # Check if UV is installed
    has_uv = False
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        has_uv = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

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
        activate_cmd = ".venv/bin/activate" if sys.platform != 'win32' else '.venv\\Scripts\\activate'
        typer.echo(f"  {typer.style(activate_cmd, fg=typer.colors.CYAN)}")
        typer.echo(f"  # Install dependencies if not already done:")
        install_cmd = 'uv pip install -e .' if has_uv else 'pip install -e .'
        typer.echo(f"  {typer.style(install_cmd, fg=typer.colors.CYAN)}")
    else:
        typer.echo(f"  # Setup virtual environment and install dependencies if skipped:")
        typer.echo(f"  # Setup virtual environment and install dependencies if skipped:")
        typer.echo(f"  python -m venv .venv")
        if sys.platform == "win32":
            typer.echo(f"  .venv\\Scripts\\activate")
        else:
            typer.echo(f"  source .venv/bin/activate")
        typer.echo(f"  pip install -e .")

    typer.echo(f"  # Run the agent:")
    typer.echo(f"  {typer.style('python main.py', fg=typer.colors.CYAN)}")

    typer.echo(f"  # Run the agent:")
    typer.echo(f"  {typer.style('python main.py', fg=typer.colors.CYAN)}")


@app.command()
def run(
    agent_dir: Optional[str] = typer.Option(".", help="Directory containing the agent"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Message to process"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Run in interactive mode"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search query"),
    context: bool = typer.Option(False, "--context", "-c", help="Retrieve context from memory"),
    context_query: Optional[str] = typer.Option(None, "--context-query", help="Query for context retrieval"),
):
    """
    Run an agent in the specified directory.
    """
    # Check if main.py exists
    main_py = os.path.join(agent_dir, "main.py")
    if not os.path.exists(main_py):
        typer.echo(f"Error: {typer.style(main_py, fg=typer.colors.RED)} not found.")
        
        # Enhanced error message to help the user
        if agent_dir == ".":
            typer.echo("\nYou seem to be running this command from the agentscaffold project directory.")
            typer.echo("The 'run' command needs to be executed from inside an agent project directory.")
            typer.echo("\nTry one of these instead:")
            typer.echo("  1. cd into an agent project directory first:")
            typer.echo("     cd your-agent-project")
            typer.echo("     agentscaffold run")
            typer.echo("  2. Or specify an agent directory:")
            typer.echo("     agentscaffold run --agent-dir your-agent-project")
        
        raise typer.Exit(1)
    
    typer.echo(f"🚀 Running agent in '{typer.style(agent_dir, bold=True)}'...")
    
    # Prepare command with arguments
    cmd = [sys.executable, main_py]
    
    if interactive:
        cmd.append("--interactive")
    
    if message:
        cmd.extend(["--message", message])
    
    if search:
        cmd.extend(["--search", search])
    
    if context:
        cmd.append("--context")
    
    if context_query:
        cmd.extend(["--context-query", context_query])

    # Run the agent
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"🔥 Error running agent: {typer.style(str(e), fg=typer.colors.RED)}")
        typer.echo(f"🔥 Error running agent: {typer.style(str(e), fg=typer.colors.RED)}")
        raise typer.Exit(1)


@app.command()
def version():
    """Display the current version of AgentScaffold."""
    from agentscaffold import __version__
    typer.echo(f"✨ AgentScaffold v{typer.style(__version__, fg=typer.colors.CYAN)}")


@app.command()
def templates():
    """List available templates."""
    import os
    from pathlib import Path
    
    templates_dir = Path(__file__).parent / "templates"
    if templates_dir.exists():
        templates = [d.name for d in templates_dir.iterdir() if d.is_dir()]
        typer.echo(f"📚 Available templates:")
        for template in templates:
            typer.echo(f"  • {typer.style(template, fg=typer.colors.CYAN)}")
    else:
        typer.echo(f"⚠️ Templates directory not found: {templates_dir}")


if __name__ == "__main__":
    app()