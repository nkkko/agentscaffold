#!/usr/bin/env python
"""
Command-line interface for AgentScaffold.
"""

import os
import sys
import subprocess
import json
import asyncio
from typing import Optional, List, Dict, Any

# Use try/except for all imports that might not be available
try:
    import typer
    from typing_extensions import Annotated
except ImportError:
    sys.exit("Error: typer package is required. Install with: pip install typer typing-extensions")

# Optional fancy UI components
try:
    from halo import Halo
    HAS_HALO = True
except ImportError:
    HAS_HALO = False

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

# Import agent scaffolding functionality
try:
    from agentscaffold.scaffold import (
        create_new_agent,
        get_agent_settings,
        TEMPLATES_DIR,
        generate_dependencies,
        generate_env_vars,
        LLM_PROVIDERS,
        SEARCH_PROVIDERS,
        MEMORY_PROVIDERS,
        LOGGING_PROVIDERS,
        UTILITY_PACKAGES,
    )
except ImportError:
    sys.exit("Error: Cannot import agentscaffold module. Make sure it's installed properly.")

app = typer.Typer(help="AgentScaffold CLI for creating and managing AI agents")

def run_command_with_spinner(command, cwd, start_message, success_message, error_message):
    """Run a command using a spinner for feedback."""
    if HAS_HALO:
        spinner = Halo(text=start_message, spinner='dots')
        spinner.start()
    else:
        print(start_message)
        spinner = Halo(text=start_message)
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

def show_provider_options():
    """Display available providers."""
    typer.echo("\nAvailable LLM providers:")
    for key, details in LLM_PROVIDERS.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")

    typer.echo("\nAvailable search providers:")
    for key, details in SEARCH_PROVIDERS.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")

    typer.echo("\nAvailable memory providers:")
    for key, details in MEMORY_PROVIDERS.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")

    typer.echo("\nAvailable logging providers:")
    for key, details in LOGGING_PROVIDERS.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")

    typer.echo("\nAvailable utility packages:")
    for key, details in UTILITY_PACKAGES.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")
    raise typer.Exit()

@app.command()
def new(
    name: Annotated[Optional[str], typer.Argument(help="Name of the agent to create")] = None,
    template: Annotated[str, typer.Option(help="Template to use")] = "basic",
    output_dir: Annotated[Optional[str], typer.Option(help="Directory to output the agent (default: current directory)")] = None,
    skip_install: Annotated[bool, typer.Option(help="Skip installing dependencies")] = False,
    interactive: Annotated[bool, typer.Option("--interactive/--no-interactive", "-i", help="Interactive prompts for configuration")] = True,
    llm_provider: Annotated[Optional[str], typer.Option(help="LLM provider (e.g., openai, anthropic, none)")] = None,
    search_provider: Annotated[Optional[str], typer.Option(help="Search provider (e.g., brave, browserbase, none)")] = "none",
    memory_provider: Annotated[Optional[str], typer.Option(help="Memory provider (e.g., supabase, none)")] = "none",
    logging_provider: Annotated[Optional[str], typer.Option(help="Logging provider (e.g., logfire, none)")] = "none",
    utilities: Annotated[Optional[List[str]], typer.Option(help="Utility packages to include, comma-separated")] = ["dotenv"],
    list_providers: Annotated[bool, typer.Option("--list-providers", "-l", help="List available providers and exit")] = False,
    api_key: Annotated[Optional[str], typer.Option(help="API key for the selected LLM provider")] = None,
):
    """
    Create a new agent with the specified name and template.
    """
    if list_providers:
        show_provider_options()

    if output_dir is None:
        output_dir = os.getcwd()

    if name is None:
        name = typer.prompt("Enter agent name")

    agent_dir = os.path.join(output_dir, name)
    styled_name = typer.style(name, fg=typer.colors.BRIGHT_WHITE, bold=True)
    styled_template = typer.style(template, fg=typer.colors.CYAN)
    typer.echo(f"✨ Creating new agent '{styled_name}' using template '{styled_template}'...")

    if not interactive:
        package_name = name.replace("-", "_")
        if llm_provider and llm_provider not in LLM_PROVIDERS:
            typer.echo(f"Error: Invalid LLM provider '{llm_provider}'.")
            typer.echo("Use --list-providers to see available options.")
            raise typer.Exit(1)
        if search_provider and search_provider not in SEARCH_PROVIDERS:
            typer.echo(f"Error: Invalid search provider '{search_provider}'.")
            typer.echo("Use --list-providers to see available options.")
            raise typer.Exit(1)
        if memory_provider and memory_provider not in MEMORY_PROVIDERS:
            typer.echo(f"Error: Invalid memory provider '{memory_provider}'.")
            typer.echo("Use --list-providers to see available options.")
            raise typer.Exit(1)
        if logging_provider and logging_provider not in LOGGING_PROVIDERS:
            typer.echo(f"Error: Invalid logging provider '{logging_provider}'.")
            typer.echo("Use --list-providers to see available options.")
            raise typer.Exit(1)
        for util in utilities:
            if util not in UTILITY_PACKAGES and util != "none":
                typer.echo(f"Warning: Unknown utility package '{util}'.")
        settings = {
            "agent_name": name,
            "package_name": package_name,
            "agent_class_name": "".join(x.capitalize() for x in package_name.split("_")),
            "description": f"An AI agent for {name}",
            "llm_provider": llm_provider or "none",
            "search_provider": search_provider,
            "memory_provider": memory_provider,
            "logging_provider": logging_provider,
            "utilities": utilities or ["dotenv"],
        }
        if api_key and llm_provider:
            if llm_provider == "openai":
                settings["api_keys"] = {"OPENAI_API_KEY": api_key}
                typer.echo(f"Using provided API key for {llm_provider} (length: {len(api_key)})")
            elif llm_provider == "anthropic":
                settings["api_keys"] = {"ANTHROPIC_API_KEY": api_key}
                typer.echo(f"Using provided API key for {llm_provider} (length: {len(api_key)})")
            elif llm_provider == "huggingface":
                settings["api_keys"] = {"HUGGINGFACE_API_KEY": api_key}
                typer.echo(f"Using provided API key for {llm_provider} (length: {len(api_key)})")
        settings["dependencies"] = generate_dependencies(
            settings["llm_provider"],
            settings["search_provider"],
            settings["memory_provider"],
            settings["logging_provider"],
            settings["utilities"],
        )
        settings["env_vars"] = generate_env_vars(
            settings["llm_provider"],
            settings["search_provider"],
            settings["memory_provider"],
            settings["logging_provider"],
        )
    else:
        settings = get_agent_settings(name)

    create_new_agent(name, template, output_dir, settings)

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
            error_message=f"❌ Failed to set up virtual environment with {typer.style('UV', fg=typer.colors.CYAN)}",
        )
        agent_scaffold_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.exists(os.path.join(agent_scaffold_dir, "setup.py")) or os.path.exists(
            os.path.join(agent_scaffold_dir, "pyproject.toml")
        ):
            run_command_with_spinner(
                command=["uv", "pip", "install", "-e", agent_scaffold_dir],
                cwd=agent_dir,
                start_message="Installing local AgentScaffold package...",
                success_message="✅ Local AgentScaffold package installed",
                error_message="⚠️ Warning: Failed to install local AgentScaffold package",
            )
        run_command_with_spinner(
            command=["uv", "pip", "install", "-e", "."],
            cwd=agent_dir,
            start_message="Installing agent package...",
            success_message="✅ Agent package installed",
            error_message="⚠️ Warning: Failed to install agent package",
        )
    elif not skip_install:
        typer.echo(f"🛠️ Setting up virtual environment with {typer.style('venv', fg=typer.colors.YELLOW)} and {typer.style('pip', fg=typer.colors.YELLOW)}...")
        run_command_with_spinner(
            command=["python", "-m", "venv", ".venv"],
            cwd=agent_dir,
            start_message="Creating virtual environment...",
            success_message=f"✅ Virtual environment created with {typer.style('venv', fg=typer.colors.YELLOW)}",
            error_message=f"⚠️ Warning: Failed to set up virtual environment with {typer.style('venv', fg=typer.colors.YELLOW)}",
        )
        agent_scaffold_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.exists(os.path.join(agent_scaffold_dir, "setup.py")) or os.path.exists(
            os.path.join(agent_scaffold_dir, "pyproject.toml")
        ):
            install_cmd = [sys.executable, "-m", "pip", "install", "-e", agent_scaffold_dir]
            run_command_with_spinner(
                command=install_cmd,
                cwd=agent_dir,
                start_message="Installing local AgentScaffold package...",
                success_message="✅ Local AgentScaffold package installed",
                error_message="⚠️ Warning: Failed to install local AgentScaffold package",
            )
        run_command_with_spinner(
            command=[sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=agent_dir,
            start_message="Installing agent package with pip...",
            success_message=f"✅ Agent package installed with {typer.style('pip', fg=typer.colors.YELLOW)}",
            error_message=f"⚠️ Warning: Failed to install agent package with {typer.style('pip', fg=typer.colors.YELLOW)}",
        )
    typer.echo(f"🎉 Agent '{typer.style(name, bold=True)}' created successfully! 🎉")
    typer.echo("\nNext steps:")
    typer.echo(f"  cd {name}")
    if not skip_install:
        typer.echo("  # Activate the virtual environment:")
        activate_cmd = ".venv/bin/activate" if sys.platform != "win32" else ".venv\\Scripts\\activate"
        typer.echo(f"  {typer.style(activate_cmd, fg=typer.colors.CYAN)}")
    else:
        typer.echo("  # Setup virtual environment and install dependencies if skipped:")
        typer.echo("  python -m venv .venv")
        if sys.platform == "win32":
            typer.echo("  .venv\\Scripts\\activate")
        else:
            typer.echo("  source .venv/bin/activate")
        typer.echo("  pip install -e .")
    typer.echo("  # Run the agent:")
    typer.echo(f"  {typer.style('python main.py', fg=typer.colors.CYAN)}")

@app.command()
def run(
    agent_dir: Annotated[Optional[str], typer.Option(help="Directory containing the agent")] = ".",
    message: Annotated[Optional[str], typer.Option("--message", "-m", help="Message to process")] = None,
    interactive: Annotated[bool, typer.Option("--interactive/--no-interactive", "-i", help="Interactive prompts for configuration")] = True,
    search: Annotated[Optional[str], typer.Option("--search", "-s", help="Search query")] = None,
    context: Annotated[bool, typer.Option("--context", "-c", help="Retrieve context from memory")] = False,
    context_query: Annotated[Optional[str], typer.Option(help="Query for context retrieval")] = None,
    silent: Annotated[bool, typer.Option("--silent", help="Run with minimal console output")] = False,
    mcp_tool: Annotated[Optional[str], typer.Option("--mcp-tool", help="Specify MCP tool to use")] = None,
    mcp_query: Annotated[Optional[str], typer.Option("--mcp-query", help="Query for MCP tool")] = None,
):
    """
    Run an agent in the specified directory.
    """
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(agent_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            typer.echo(f"Loaded environment variables from {env_path}")
        else:
            typer.echo(f"Warning: No .env file found at {env_path}")
    except ImportError:
        typer.echo("Warning: python-dotenv not installed, using system environment variables only")
    main_py = os.path.join(agent_dir, "main.py")
    if not os.path.exists(main_py):
        typer.echo(f"Error: {typer.style(main_py, fg=typer.colors.RED)} not found.")
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
    cmd = [sys.executable, main_py]
    if interactive:
        cmd.append("--interactive")
    if silent:
        cmd.append("--silent")
    if message:
        cmd.extend(["--message", message])
    if search:
        cmd.extend(["--search", search])
    if context:
        cmd.append("--context")
    if context_query:
        cmd.extend(["--context-query", context_query])
    if mcp_tool:
        cmd.extend(["--mcp-tool", mcp_tool])
    if mcp_query:
        cmd.extend(["--mcp-query", mcp_query])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"🔥 Error running agent: {typer.style(str(e), fg=typer.colors.RED)}")
        raise typer.Exit(1)

@app.command()
def version():
    """Display the current version of AgentScaffold."""
    try:
        from agentscaffold import __version__
        typer.echo(f"✨ AgentScaffold v{typer.style(__version__, fg=typer.colors.CYAN)}")
    except (ImportError, AttributeError):
        typer.echo("⚠️ Could not determine AgentScaffold version")

@app.command()
def templates():
    """List available templates."""
    try:
        if TEMPLATES_DIR.exists():
            templates = [d.name for d in TEMPLATES_DIR.iterdir() if d.is_dir()]
            typer.echo("📚 Available templates:")
            for template in templates:
                typer.echo(f"  • {typer.style(template, fg=typer.colors.CYAN)}")
        else:
            typer.echo(f"⚠️ Templates directory not found: {TEMPLATES_DIR}")
    except ImportError:
        typer.echo("⚠️ Could not import TEMPLATES_DIR from agentscaffold.scaffold")

@app.command()
def providers():
    """List available providers for LLM, search, memory and logging."""
    show_provider_options()

# MCP commands sub-group
mcp_app = typer.Typer(help="Configure and manage MCP (Model Context Protocol) servers.")
app.add_typer(mcp_app, name="mcp")

@mcp_app.command("add")
def mcp_add(
    name: Annotated[Optional[str], typer.Argument(help="Name for the MCP server")] = None,
    server_type: Annotated[str, typer.Option("--type", "-t", help="Type of MCP server: stdio or http")] = "stdio",
    command: Annotated[Optional[str], typer.Option(help="Command to run the MCP server (for stdio)")] = None,
    url: Annotated[Optional[str], typer.Option(help="URL for the HTTP MCP server (for http)")] = None,
    args: Annotated[Optional[List[str]], typer.Option(help="Arguments for the command")] = None,
    scope: Annotated[str, typer.Option("-s", "--scope", help="Configuration scope (local, user, or project)")] = "project",
    env_var: Annotated[Optional[List[str]], typer.Option("-e", "--env", help="Environment variables in KEY=VALUE format (repeatable)")] = None,
):
    """
    Add a new MCP server.
    
    For HTTP servers, provide a URL (and optionally API key via env).
    For stdio servers, provide the command to run.
    """
    try:
        from agentscaffold.providers.mcp import load_mcp_servers, save_mcp_servers
    except ImportError:
        typer.echo("Error: MCP provider module not found.")
        raise typer.Exit(1)

    if name is None:
        name = typer.prompt("Enter a name for the MCP server")

    if server_type.lower() == "http":
        if url is None:
            url = typer.prompt("Enter the URL for the HTTP MCP server")
        server_config = {
            "type": "http",
            "url": url,
        }
    else:
        if command is None:
            command = typer.prompt("Enter the command to run the MCP server")
        if args is None:
            args = []
            while typer.confirm("Add command argument?", default=False):
                args.append(typer.prompt("Enter argument"))
        server_config = {
            "type": "stdio",
            "command": command,
            "args": args,
        }

    if env_var:
        env_dict = {}
        for env in env_var:
            if "=" in env:
                key, value = env.split("=", 1)
                env_dict[key] = value
        server_config["env"] = env_dict

    # Load existing servers based on scope
    if scope == "project":
        servers = load_mcp_servers(".")
    elif scope == "user":
        servers = load_mcp_servers(os.path.expanduser("~"))
    else:
        servers = load_mcp_servers()

    # Check if a server with the same name exists
    existing_names = [srv.get("id") for srv in servers]
    if name in existing_names:
        if not typer.confirm(f"MCP server '{name}' already exists. Override?"):
            raise typer.Exit(1)
        servers = [srv for srv in servers if srv.get("id") != name]

    # Add the id and configuration
    server_config["id"] = name
    servers.append(server_config)

    save_location = "." if scope == "project" else (os.path.expanduser("~") if scope == "user" else None)
    if save_mcp_servers(servers, save_location):
        typer.echo(f"Added MCP server '{name}' with configuration: {server_config}")
    else:
        typer.echo("Failed to save MCP server configuration.")

@mcp_app.command("list")
def mcp_list(
    scope: Annotated[str, typer.Option("-s", "--scope", help="Configuration scope (local, user, or project)")] = "project",
):
    """
    List MCP servers configured.
    """
    try:
        from agentscaffold.providers.mcp import load_mcp_servers
    except ImportError:
        typer.echo("Error: MCP provider module not found.")
        raise typer.Exit(1)

    if scope == "project":
        servers = load_mcp_servers(".")
    elif scope == "user":
        servers = load_mcp_servers(os.path.expanduser("~"))
    else:
        servers = load_mcp_servers()

    if not servers:
        typer.echo("No MCP servers configured.")
    else:
        typer.echo("Configured MCP servers:")
        for srv in servers:
            typer.echo(f"  - {srv.get('id')}: {srv}")

@mcp_app.command("remove")
def mcp_remove(
    name: Annotated[str, typer.Argument(help="Name of the MCP server to remove")],
    scope: Annotated[str, typer.Option("-s", "--scope", help="Configuration scope (local, user, or project)")] = "project",
):
    """
    Remove an MCP server from configuration.
    """
    try:
        from agentscaffold.providers.mcp import load_mcp_servers, save_mcp_servers
    except ImportError:
        typer.echo("Error: MCP provider module not found.")
        raise typer.Exit(1)

    if scope == "project":
        servers = load_mcp_servers(".")
    elif scope == "user":
        servers = load_mcp_servers(os.path.expanduser("~"))
    else:
        servers = load_mcp_servers()

    updated_servers = [srv for srv in servers if srv.get("id") != name]

    if len(updated_servers) == len(servers):
        typer.echo(f"No MCP server named '{name}' found.")
        raise typer.Exit(1)

    save_location = "." if scope == "project" else (os.path.expanduser("~") if scope == "user" else None)
    if save_mcp_servers(updated_servers, save_location):
        typer.echo(f"Removed MCP server '{name}'.")
    else:
        typer.echo("Failed to save MCP server configuration.")

@mcp_app.command("test")
def mcp_test(
    name: Annotated[str, typer.Argument(help="Name of the MCP server to test")],
    scope: Annotated[str, typer.Option("-s", "--scope", help="Configuration scope (local, user, or project)")] = "project",
):
    """
    Test connectivity of an MCP server.
    """
    try:
        from agentscaffold.providers.mcp import load_mcp_servers
        from agentscaffold.providers.mcp.client import test_mcp_connection
    except ImportError:
        typer.echo("Error: MCP provider module not found.")
        raise typer.Exit(1)

    if scope == "project":
        servers = load_mcp_servers(".")
    elif scope == "user":
        servers = load_mcp_servers(os.path.expanduser("~"))
    else:
        servers = load_mcp_servers()

    server = next((srv for srv in servers if srv.get("id") == name), None)
    if not server:
        typer.echo(f"No MCP server named '{name}' found.")
        raise typer.Exit(1)

    typer.echo(f"Testing MCP server '{name}'...")
    result = test_mcp_connection(server)
    if result.get("success"):
        typer.echo(f"✅ MCP server '{name}' is reachable. Capabilities: {result.get('capabilities')}")
    else:
        typer.echo(f"❌ MCP server '{name}' test failed. Error: {result.get('error')}")

@app.command()
def new_flask_builder(
    name: Annotated[Optional[str], typer.Argument(help="Name of the Flask app to create")] = None,
    description: Annotated[Optional[str], typer.Option(help="Description of the Flask app")] = None,
    output_dir: Annotated[Optional[str], typer.Option(help="Directory to output the app (default: current directory)")] = None,
):
    """
    Create a new Daytona-powered Flask application that builds itself from descriptions.
    """
    if name is None:
        name = typer.prompt("Enter Flask app name")
    
    if description is None:
        description = typer.prompt("Enter app description", default=f"A self-building Flask application for {name}")
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Call the Flask builder generator from scaffold module
    from agentscaffold.scaffold import create_flask_builder
    
    result = create_flask_builder(name, description, output_dir)
    
    if result:
        typer.echo(f"🎉 Daytona-powered Flask builder '{name}' created successfully!")
        typer.echo("\nNext steps:")
        typer.echo(f"  cd {name}")
        typer.echo("  pip install -r requirements.txt")
        typer.echo("  # Add your API keys to .env")
        typer.echo("  python app.py")
        typer.echo("\nThen visit http://localhost:5000 in your browser to build your application from a description.")
    else:
        typer.echo(f"❌ Failed to create Flask builder '{name}'")
    
if __name__ == "__main__":
    app()
