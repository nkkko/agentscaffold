"""Command-line interface for AgentScaffold."""

import os
import sys
import subprocess
from typing import Optional, List, Dict, Any

# Use try/except for all imports that might not be available
try:
    import typer
    from typing_extensions import Annotated
except ImportError:
    sys.exit("Error: typer package is required. Install with: pip install typer typing-extensions")

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
    from agentscaffold.scaffold import (
        LLM_PROVIDERS, SEARCH_PROVIDERS, MEMORY_PROVIDERS, 
        LOGGING_PROVIDERS, UTILITY_PACKAGES,
        generate_dependencies, generate_env_vars
    )
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

def show_provider_options(ctx=None):
    """Show available provider options."""
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
    interactive: Annotated[bool, typer.Option(help="Interactive prompts for configuration")] = True,
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

    # If name wasn't provided as an argument, prompt for it
    if name is None:
        name = typer.prompt("Enter agent name")
    
    # Skip the prompt in get_agent_settings if we have the name
    agent_dir = os.path.join(output_dir, name)

    # Fix: use typer.style() instead of trying to call color constants
    styled_name = typer.style(name, fg=typer.colors.BRIGHT_WHITE, bold=True)
    styled_template = typer.style(template, fg=typer.colors.CYAN)
    typer.echo(f"✨ Creating new agent '{styled_name}' using template '{styled_template}'...")

    # Generate settings - either interactively or from command-line arguments
    settings = None
    if not interactive:
        # Convert kebab-case to snake_case for the package name
        package_name = name.replace("-", "_")
        
        # Validate providers
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
        
        # Create settings from command-line arguments
        settings = {
            "agent_name": name,
            "package_name": package_name,
            "agent_class_name": "".join(x.capitalize() for x in package_name.split("_")),
            "description": f"An AI agent for {name}",
            "llm_provider": llm_provider or "daytona",
            "search_provider": search_provider,
            "memory_provider": memory_provider,
            "logging_provider": logging_provider,
            "utilities": utilities or ["dotenv"],
        }
        
        # Add API key if provided
        if api_key and llm_provider:
            # Determine the environment variable name from the provider
            if llm_provider == "openai":
                settings["api_keys"] = {"OPENAI_API_KEY": api_key}
                typer.echo(f"Using provided API key for {llm_provider} (length: {len(api_key)})")
            elif llm_provider == "anthropic":
                settings["api_keys"] = {"ANTHROPIC_API_KEY": api_key}
                typer.echo(f"Using provided API key for {llm_provider} (length: {len(api_key)})")
            elif llm_provider == "huggingface":
                settings["api_keys"] = {"HUGGINGFACE_API_KEY": api_key}
                typer.echo(f"Using provided API key for {llm_provider} (length: {len(api_key)})")
        
        # Add dependencies and env_vars
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

    if has_uv and not skip_install:
        typer.echo(f"🛠️ Setting up virtual environment with {typer.style('UV', fg=typer.colors.CYAN)}...")
        run_command_with_spinner(
            command=["uv", "venv", ".venv"],
            cwd=agent_dir,
            start_message="Creating virtual environment...",
            success_message=f"✅ Virtual environment created with {typer.style('UV', fg=typer.colors.CYAN)}",
            error_message=f"❌ Failed to set up virtual environment with {typer.style('UV', fg=typer.colors.CYAN)}"
        )

        # Try to install local AgentScaffold package first if available
        agent_scaffold_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.exists(os.path.join(agent_scaffold_dir, "setup.py")) or os.path.exists(os.path.join(agent_scaffold_dir, "pyproject.toml")):
            run_command_with_spinner(
                command=["uv", "pip", "install", "-e", agent_scaffold_dir],
                cwd=agent_dir,
                start_message="Installing local AgentScaffold package...",
                success_message=f"✅ Local AgentScaffold package installed",
                error_message=f"⚠️ Warning: Failed to install local AgentScaffold package"
            )

        # Install dependencies
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

        # Try to install local AgentScaffold package first if available
        agent_scaffold_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.exists(os.path.join(agent_scaffold_dir, "setup.py")) or os.path.exists(os.path.join(agent_scaffold_dir, "pyproject.toml")):
            install_cmd = [
                sys.executable, "-m", "pip", "install", "-e", agent_scaffold_dir
            ]
            run_command_with_spinner(
                command=install_cmd,
                cwd=agent_dir,
                start_message="Installing local AgentScaffold package...",
                success_message="✅ Local AgentScaffold package installed",
                error_message="⚠️ Warning: Failed to install local AgentScaffold package"
            )

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
    agent_dir: Annotated[Optional[str], typer.Option(help="Directory containing the agent")] = ".",
    message: Annotated[Optional[str], typer.Option("--message", "-m", help="Message to process")] = None,
    interactive: Annotated[bool, typer.Option("--interactive/--no-interactive", "-i", help="Interactive prompts for configuration")] = True,
    search: Annotated[Optional[str], typer.Option("--search", "-s", help="Search query")] = None,
    context: Annotated[bool, typer.Option("--context", "-c", help="Retrieve context from memory")] = False,
    context_query: Annotated[Optional[str], typer.Option(help="Query for context retrieval")] = None,
    silent: Annotated[bool, typer.Option("--silent", help="Run with minimal console output")] = False,

):
    """
    Run an agent in the specified directory.
    """
    # First load environment variables from .env file
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(agent_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            typer.echo(f"Loaded environment variables from {env_path}")
        else:
            typer.echo(f"Warning: No .env file found at {env_path}")
    except ImportError:
        typer.echo("Warning: python-dotenv not installed, using system environment variables only")

    # Check Daytona environment variables

        """Run an agent in Daytona (mandatory cloud execution)."""
        typer.echo("🚀 Starting agent in Daytona cloud environment...")
    
        # Verify Daytona requirements
        required_vars = ["DAYTONA_API_KEY", "DAYTONA_SERVER_URL"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            typer.echo(f"Error: Missing required Daytona variables: {', '.join(missing_vars)}")
            typer.echo("These must be set in your .env file or environment")
            raise typer.Exit(1)
        
        typer.echo("\nPossible solutions:")
        typer.echo("1. Add these variables to your .env file:")
        typer.echo("   DAYTONA_API_KEY=your_api_key_here")
        typer.echo("   DAYTONA_SERVER_URL=your_server_url_here")
        typer.echo("   DAYTONA_TARGET=us (optional)")
        typer.echo("\n2. Or set them in your shell environment before running:")
        typer.echo("   export DAYTONA_API_KEY=your_api_key_here")
        typer.echo("   export DAYTONA_SERVER_URL=your_server_url_here")
        typer.echo("\n3. Make sure your .env file is in the agent directory")
        
        # Show current environment values (masking sensitive data)
        typer.echo("\nCurrent environment values:")
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                masked = value[:4] + '...' + value[-4:] if len(value) > 8 else '****'
                typer.echo(f"  {var}={masked}")
            else:
                typer.echo(f"  {var}=[NOT SET]")
        
        raise typer.Exit(1)

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

    # Run the agent
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
        from agentscaffold.scaffold import TEMPLATES_DIR
        
        templates_dir = TEMPLATES_DIR
        if templates_dir.exists():
            templates = [d.name for d in templates_dir.iterdir() if d.is_dir()]
            typer.echo(f"📚 Available templates:")
            for template in templates:
                typer.echo(f"  • {typer.style(template, fg=typer.colors.CYAN)}")
        else:
            typer.echo(f"⚠️ Templates directory not found: {templates_dir}")
    except ImportError:
        typer.echo("⚠️ Could not import TEMPLATES_DIR from agentscaffold.scaffold")


@app.command()
def providers():
    """List available providers for LLM, search, memory and logging."""
    show_provider_options()


@app.group()
def mcp():
    """Manage MCP (Model Context Protocol) servers for agent integration."""
    pass


@mcp.command("add")
def mcp_add(
    name: Annotated[str, typer.Argument(help="Unique name for the MCP server")],
    url: Annotated[str, typer.Argument(help="URL of the MCP server")],
    api_key: Annotated[Optional[str], typer.Option(help="API key for the MCP server")] = None,
    auth_type: Annotated[str, typer.Option(help="Authentication type (api_key, oauth, none)")] = "api_key",
    capability: Annotated[List[str], typer.Option("--capability", "-c", help="Server capabilities (tools, resources, prompts)")] = ["tools"],
):
    """
    Add a new MCP server configuration to the current agent.
    This command must be run from inside an agent directory.
    """
    try:
        from agentscaffold.providers.mcp import load_mcp_servers, save_mcp_servers
        from agentscaffold.providers.mcp.base import MCPServer
    except ImportError:
        typer.echo("Error: MCP provider module not found.")
        raise typer.Exit(1)
    
    # Check if current directory looks like an agent directory
    if not os.path.exists("main.py") and not os.path.isdir("agent"):
        typer.echo("Error: This doesn't appear to be an agent directory.")
        typer.echo("Run this command from inside your agent directory.")
        raise typer.Exit(1)
    
    # Load existing servers
    servers = load_mcp_servers()
    
    # Check if server already exists
    if name in servers:
        if not typer.confirm(f"MCP server '{name}' already exists. Override?"):
            raise typer.Exit(1)
    
    # Create and save the new server
    server = MCPServer(
        name=name, 
        url=url, 
        api_key=api_key, 
        auth_type=auth_type, 
        capabilities=capability
    )
    
    servers[name] = server.to_dict()
    save_mcp_servers(servers)
    
    typer.echo(f"✅ Added MCP server '{typer.style(name, fg=typer.colors.CYAN)}'")
    typer.echo(f"   URL: {url}")
    typer.echo(f"   Capabilities: {', '.join(capability)}")
    
    # Check if httpx is installed for connectivity
    try:
        import httpx
    except ImportError:
        typer.echo("\n⚠️ Warning: httpx package is required for MCP connectivity.")
        typer.echo("   Install with: pip install httpx")


@mcp.command("list")
def mcp_list():
    """
    List all configured MCP servers for the current agent.
    This command must be run from inside an agent directory.
    """
    try:
        from agentscaffold.providers.mcp import load_mcp_servers
    except ImportError:
        typer.echo("Error: MCP provider module not found.")
        raise typer.Exit(1)
    
    # Check if current directory looks like an agent directory
    if not os.path.exists("main.py") and not os.path.isdir("agent"):
        typer.echo("Error: This doesn't appear to be an agent directory.")
        typer.echo("Run this command from inside your agent directory.")
        raise typer.Exit(1)
    
    servers = load_mcp_servers()
    
    if not servers:
        typer.echo("No MCP servers configured for this agent.")
        typer.echo("Use 'agentscaffold mcp add' to add a server.")
        return
    
    typer.echo("📋 Configured MCP servers:")
    for name, data in servers.items():
        typer.echo(f"  • {typer.style(name, fg=typer.colors.CYAN)}")
        typer.echo(f"    URL: {data['url']}")
        typer.echo(f"    Capabilities: {', '.join(data.get('capabilities', ['tools']))}")
        typer.echo(f"    Auth type: {data.get('auth_type', 'api_key')}")
        if data.get('api_key'):
            masked_key = "*" * (len(data['api_key']) - 4) + data['api_key'][-4:] if len(data['api_key']) > 4 else "****" 
            typer.echo(f"    API key: {masked_key}")
        typer.echo("")


@mcp.command("remove")
def mcp_remove(
    name: Annotated[str, typer.Argument(help="Name of the MCP server to remove")]
):
    """
    Remove an MCP server configuration from the current agent.
    This command must be run from inside an agent directory.
    """
    try:
        from agentscaffold.providers.mcp import load_mcp_servers, save_mcp_servers
    except ImportError:
        typer.echo("Error: MCP provider module not found.")
        raise typer.Exit(1)
    
    # Check if current directory looks like an agent directory
    if not os.path.exists("main.py") and not os.path.isdir("agent"):
        typer.echo("Error: This doesn't appear to be an agent directory.")
        typer.echo("Run this command from inside your agent directory.")
        raise typer.Exit(1)
    
    servers = load_mcp_servers()
    
    if name not in servers:
        typer.echo(f"Error: MCP server '{name}' not found.")
        raise typer.Exit(1)
    
    if typer.confirm(f"Are you sure you want to remove MCP server '{name}'?"):
        del servers[name]
        save_mcp_servers(servers)
        typer.echo(f"✅ Removed MCP server '{name}'")


@mcp.command("test")
def mcp_test(
    name: Annotated[str, typer.Argument(help="Name of the MCP server to test")]
):
    """
    Test connection to an MCP server.
    This command must be run from inside an agent directory.
    """
    try:
        from agentscaffold.providers.mcp import load_mcp_servers
        from agentscaffold.providers.mcp.client import test_mcp_connection
    except ImportError:
        typer.echo("Error: MCP provider module not found.")
        raise typer.Exit(1)
    
    # Check if current directory looks like an agent directory
    if not os.path.exists("main.py") and not os.path.isdir("agent"):
        typer.echo("Error: This doesn't appear to be an agent directory.")
        typer.echo("Run this command from inside your agent directory.")
        raise typer.Exit(1)
    
    # Check if httpx is installed
    try:
        import httpx
    except ImportError:
        typer.echo("Error: httpx package is required for MCP connectivity.")
        typer.echo("Install with: pip install httpx")
        raise typer.Exit(1)
    
    servers = load_mcp_servers()
    
    if name not in servers:
        typer.echo(f"Error: MCP server '{name}' not found.")
        raise typer.Exit(1)
    
    server_config = servers[name]
    
    with typer.progressbar(label=f"Testing connection to {name}", length=100) as progress:
        progress.update(10)
        result = test_mcp_connection(server_config)
        progress.update(100)
    
    if result["success"]:
        typer.echo(f"✅ Successfully connected to MCP server '{name}'")
        typer.echo(f"   Available capabilities: {', '.join(result.get('capabilities', ['tools']))}")
        if result.get("tools"):
            typer.echo(f"   Available tools: {', '.join(tool['name'] for tool in result['tools'] if 'name' in tool)}")
    else:
        typer.echo(f"❌ Failed to connect to MCP server '{name}'")
        typer.echo(f"   Error: {result.get('error', 'Unknown error')}")
        typer.echo("\nCheck the URL and API key, and make sure the server is running.")


if __name__ == "__main__":
    app()