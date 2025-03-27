"""Scaffolding functionality for creating agents."""

import os
import shutil
import jinja2
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
import json

# Try to import Typer-related components for CLI prompts
try:
    import typer
    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False
    # Fallback to input() for prompts

# Path to templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Config file for saved defaults
CONFIG_FILE = Path.home() / ".agentscaffold.json"

# Provider configurations
LLM_PROVIDERS = {
    "openai": {
        "env_vars": ["OPENAI_API_KEY"],
        "package": "openai>=1.0.0",
        "description": "OpenAI API (gpt-4o, etc.)"
    },
    "anthropic": {
        "env_vars": ["ANTHROPIC_API_KEY"],
        "package": "anthropic>=0.5.0",
        "description": "Anthropic API (Claude)"
    }, 
     "ollama": {
        "env_vars": ["OLLAMA_BASE_URL"],
        "package": "ollama>=0.1.0",
        "description": "Ollama local LLM server"
    },
    "none": {
        "env_vars": [],
        "package": None,
        "description": "No LLM provider"
    }
}

SEARCH_PROVIDERS = {
    "brave": {
        "env_vars": ["BRAVE_API_KEY"],
        "package": "httpx>=0.24.0",  # Using httpx instead of a dedicated brave-search package
        "description": "Brave Search API"
    },
    "browserbase": {
        "env_vars": ["BROWSERBASE_API_KEY"],
        "package": "browserbase>=0.1.0",
        "description": "BrowserBase search and browsing API"
    },
    "google": {
        "env_vars": ["GOOGLE_API_KEY", "GOOGLE_CSE_ID"],
        "package": "google-api-python-client>=2.0.0",
        "description": "Google Custom Search API"
    },
    "none": {
        "env_vars": [],
        "package": None,
        "description": "No search provider"
    }
}

MEMORY_PROVIDERS = {
    "supabase": {
        "env_vars": ["SUPABASE_URL", "SUPABASE_KEY"],
        "package": "supabase>=0.7.0",
        "description": "Supabase vector database"
    },
    "chromadb": {
        "env_vars": ["OPENAI_API_KEY"],  # For embeddings, but can be configured to use others
        "package": "chromadb>=0.4.0",
        "description": "ChromaDB local vector database"
    },
    "pinecone": {
        "env_vars": ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"],
        "package": "pinecone-client>=2.2.0",
        "description": "Pinecone vector database"
    },
    "none": {
        "env_vars": [],
        "package": None,
        "description": "No memory provider"
    }
}

# Update the LOGGING_PROVIDERS dictionary to include LogFire and other options
LOGGING_PROVIDERS = {
    "logfire": {
        "env_vars": ["LOGFIRE_API_KEY"],
        "package": "logfire>=0.9.0",
        "description": "LogFire observability platform"
    },
    "prometheus": {
        "env_vars": [],  # No API keys needed for Prometheus
        "package": "prometheus-client>=0.16.0",
        "description": "Prometheus metrics exporter"
    },
    "langfuse": {
        "env_vars": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
        "package": "langfuse>=1.0.0",
        "description": "Langfuse LLM observability"
    },
    "none": {
        "env_vars": [],
        "package": None,
        "description": "No logging provider"
    }
}

# Add more utility packages
UTILITY_PACKAGES = {
    "pydantic-ai": {
        "package": "pydantic-ai>=0.1.0",
        "description": "Pydantic-based AI model integration"
    },
    "tiktoken": {
        "package": "tiktoken>=0.4.0",
        "description": "Fast BPE tokenizer from OpenAI"
    },
    "jinja2": {
        "package": "jinja2>=3.0.0",
        "description": "Template engine for Python"
    },
    "fastapi": {
        "package": "fastapi>=0.100.0 uvicorn>=0.20.0",
        "description": "FastAPI web framework for building APIs"
    }
}

# Default Daytona configuration
DAYTONA_CONFIG = {
    "env_vars": ["DAYTONA_API_KEY", "DAYTONA_SERVER_URL", "DAYTONA_TARGET"],
    "description": "Daytona API for secure execution"
}


def load_config() -> Dict[str, Any]:
    """Load saved configuration defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_config(config: Dict[str, Any]) -> None:
    """
    Save configuration defaults.
    
    Args:
        config: Configuration to save
    """
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def prompt_choice(message: str, choices: List[str], default: Optional[str] = None) -> str:
    """
    Prompt for a choice from a list of options.
    
    Args:
        message: Prompt message
        choices: List of choices
        default: Default choice
        
    Returns:
        Selected choice
    """
    if HAS_TYPER:
        # Use typer for nicer prompts
        formatted_choices = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
        choice_prompt = f"{message}\n{formatted_choices}\nChoose an option [1-{len(choices)}]"
        
        default_idx = choices.index(default) + 1 if default in choices else None
        default_prompt = f" (default: {default_idx})" if default_idx else ""
        
        while True:
            result = typer.prompt(f"{choice_prompt}{default_prompt}", default=str(default_idx) if default_idx else None)
            try:
                idx = int(result)
                if 1 <= idx <= len(choices):
                    return choices[idx-1]
                else:
                    typer.echo("Invalid choice. Please try again.")
            except ValueError:
                typer.echo("Invalid input. Please enter a number.")
    else:
        # Fallback to regular input
        print(message)
        for i, choice in enumerate(choices):
            print(f"{i+1}. {choice}")
        
        default_prompt = f" (default: {choices.index(default) + 1})" if default in choices else ""
        while True:
            try:
                result = input(f"Choose an option [1-{len(choices)}]{default_prompt}: ")
                if not result and default in choices:
                    return default
                idx = int(result)
                if 1 <= idx <= len(choices):
                    return choices[idx-1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")


def prompt_multiple_choice(message: str, choices: List[str], defaults: Optional[List[str]] = None) -> List[str]:
    """
    Prompt for multiple choices from a list of options.
    
    Args:
        message: Prompt message
        choices: List of choices
        defaults: Default choices
        
    Returns:
        List of selected choices
    """
    if not HAS_TYPER:
        # Simple fallback for non-typer environments
        print(message)
        for i, choice in enumerate(choices):
            print(f"{i+1}. {choice}")
        
        defaults_str = ", ".join(str(choices.index(d) + 1) for d in defaults) if defaults else ""
        default_prompt = f" (defaults: {defaults_str})" if defaults else ""
        
        while True:
            try:
                result = input(f"Choose options (comma-separated) [1-{len(choices)}]{default_prompt}: ")
                if not result and defaults:
                    return defaults
                
                selected = []
                for item in result.split(","):
                    idx = int(item.strip())
                    if 1 <= idx <= len(choices):
                        selected.append(choices[idx-1])
                    else:
                        print(f"Invalid choice: {idx}. Skipping.")
                
                if selected:
                    return selected
                else:
                    print("No valid choices selected. Please try again.")
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers.")
    else:
        # Use typer for multiple choice (checkboxes not directly supported)
        choices_with_descriptions = [f"{i+1}. {choice}" for i, choice in enumerate(choices)]
        choices_display = "\n".join(choices_with_descriptions)
        
        defaults_str = ", ".join(str(choices.index(d) + 1) for d in defaults) if defaults else ""
        default_prompt = f" (defaults: {defaults_str})" if defaults else ""
        
        prompt = f"{message}\n{choices_display}\nChoose options (comma-separated)"
        
        while True:
            result = typer.prompt(f"{prompt}{default_prompt}")
            
            if not result and defaults:
                return defaults
            
            try:
                selected = []
                for item in result.split(","):
                    idx = int(item.strip())
                    if 1 <= idx <= len(choices):
                        selected.append(choices[idx-1])
                    else:
                        typer.echo(f"Invalid choice: {idx}. Skipping.")
                
                if selected:
                    return selected
                else:
                    typer.echo("No valid choices selected. Please try again.")
            except ValueError:
                typer.echo("Invalid input. Please enter comma-separated numbers.")


def prompt_yes_no(message: str, default: bool = False) -> bool:
    """
    Prompt for a yes/no answer.
    
    Args:
        message: Prompt message
        default: Default value
        
    Returns:
        True for yes, False for no
    """
    if HAS_TYPER:
        return typer.confirm(message, default=default)
    else:
        default_str = "Y/n" if default else "y/N"
        response = input(f"{message} [{default_str}]: ").strip().lower()
        if not response:
            return default
        return response.startswith('y')


def prompt_text(message: str, default: Optional[str] = None, validator: Optional[Callable[[str], bool]] = None) -> str:
    """
    Prompt for text input with optional validation.
    
    Args:
        message: Prompt message
        default: Default value
        validator: Validation function
        
    Returns:
        User input
    """
    if HAS_TYPER:
        while True:
            result = typer.prompt(message, default=default)
            if validator is None or validator(result):
                return result
            typer.echo("Invalid input. Please try again.")
    else:
        default_prompt = f" (default: {default})" if default else ""
        while True:
            result = input(f"{message}{default_prompt}: ")
            if not result and default:
                return default
            if validator is None or validator(result):
                return result
            print("Invalid input. Please try again.")


def get_agent_settings(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get agent settings from user input or defaults.
    
    Args:
        name: Agent name (optional, will prompt if not provided)
        
    Returns:
        Agent settings
    """
    # Load saved defaults
    config = load_config()
    defaults = config.get('defaults', {})
    
    # Agent name - only prompt if not provided
    if name is None:
        name = prompt_text("Enter agent name", validator=lambda s: bool(s.strip()) and not os.path.exists(s))
    
    # Convert kebab-case to snake_case for the package name
    package_name = name.replace("-", "_")
    package_name = prompt_text("Enter package name (for Python imports)", default=package_name, 
                              validator=lambda s: bool(s.strip()) and s.isidentifier())
    
    # Agent description
    description = prompt_text("Enter agent description", default=f"An AI agent for {name}")
    
    # LLM Provider
    llm_provider_choices = list(LLM_PROVIDERS.keys())
    llm_provider_display = [f"{k} - {v['description']}" for k, v in LLM_PROVIDERS.items()]
    llm_default = defaults.get('llm_provider', 'openai')
    llm_provider = prompt_choice("Select LLM provider:", llm_provider_choices, default=llm_default)
    
    # Prompt for API keys based on provider
    api_keys = {}
    
    # If OpenAI is selected, prompt for API key
    if llm_provider == "openai":
        if HAS_TYPER:
            openai_api_key = typer.prompt("Enter your OpenAI API key", hide_input=True)
            typer.echo(f"OpenAI API key received (length: {len(openai_api_key)})")
            api_keys["OPENAI_API_KEY"] = openai_api_key
        else:
            # Fallback to regular input if typer is not available
            openai_api_key = input("Enter your OpenAI API key: ")
            print(f"OpenAI API key received (length: {len(openai_api_key)})")
            api_keys["OPENAI_API_KEY"] = openai_api_key
    
    # Similarly for Anthropic, if selected
    if llm_provider == "anthropic":
        if HAS_TYPER:
            anthropic_api_key = typer.prompt("Enter your Anthropic API key", hide_input=True)
            typer.echo(f"Anthropic API key received (length: {len(anthropic_api_key)})")
            api_keys["ANTHROPIC_API_KEY"] = anthropic_api_key
        else:
            anthropic_api_key = input("Enter your Anthropic API key: ")
            print(f"Anthropic API key received (length: {len(anthropic_api_key)})")
            api_keys["ANTHROPIC_API_KEY"] = anthropic_api_key
    
    # Search Provider
    search_provider_choices = list(SEARCH_PROVIDERS.keys())
    search_provider_display = [f"{k} - {v['description']}" for k, v in SEARCH_PROVIDERS.items()]
    search_default = defaults.get('search_provider', 'none')
    search_provider = prompt_choice("Select search provider:", search_provider_choices, default=search_default)
    
    # Memory Provider
    memory_provider_choices = list(MEMORY_PROVIDERS.keys())
    memory_provider_display = [f"{k} - {v['description']}" for k, v in MEMORY_PROVIDERS.items()]
    memory_default = defaults.get('memory_provider', 'none')
    memory_provider = prompt_choice("Select memory provider:", memory_provider_choices, default=memory_default)
    
    # Logging Provider
    logging_provider_choices = list(LOGGING_PROVIDERS.keys())
    logging_provider_display = [f"{k} - {v['description']}" for k, v in LOGGING_PROVIDERS.items()]
    logging_default = defaults.get('logging_provider', 'none')
    logging_provider = prompt_choice("Select logging provider:", logging_provider_choices, default=logging_default)
    
    # Utility packages
    utility_choices = list(UTILITY_PACKAGES.keys())
    utility_display = [f"{k} - {v['description']}" for k, v in UTILITY_PACKAGES.items()]
    utility_defaults = defaults.get('utilities', [])
    utilities = prompt_multiple_choice("Select utility packages:", utility_choices, defaults=utility_defaults)
    
    # Save as defaults?
    save_as_defaults = prompt_yes_no("Save these choices as defaults for future projects?", default=False)
    if save_as_defaults:
        new_defaults = {
            'llm_provider': llm_provider,
            'search_provider': search_provider,
            'memory_provider': memory_provider,
            'logging_provider': logging_provider,
            'utilities': utilities
        }
        config['defaults'] = new_defaults
        save_config(config)
        if HAS_TYPER:
            typer.echo("✅ Settings saved as defaults")
        else:
            print("✅ Settings saved as defaults")
    
    # Gather all settings
    settings = {
        "agent_name": name,
        "package_name": package_name,
        "agent_class_name": "".join(x.capitalize() for x in package_name.split("_")),
        "description": description,
        "llm_provider": llm_provider,
        "search_provider": search_provider,
        "memory_provider": memory_provider,
        "logging_provider": logging_provider,
        "utilities": utilities,
        "dependencies": generate_dependencies(llm_provider, search_provider, memory_provider, logging_provider, utilities),
        "env_vars": generate_env_vars(llm_provider, search_provider, memory_provider, logging_provider),
        "api_keys": api_keys  # Add API keys to settings
    }
    
    return settings


def generate_dependencies(
    llm_provider: str,
    search_provider: str,
    memory_provider: str,
    logging_provider: str,
    utilities: List[str]
) -> List[str]:
    """
    Generate list of dependencies based on selected providers.
    
    Args:
        llm_provider: LLM provider
        search_provider: Search provider
        memory_provider: Memory provider
        logging_provider: Logging provider
        utilities: List of utility packages
        
    Returns:
        List of dependencies
    """
    dependencies = [
        "pydantic>=2.0.0",
        "agentscaffold",
        "pydantic-ai",
        "daytona-sdk>=0.1.0",  # Always include daytona-sdk
    ]
    
    # Add LLM provider package
    if llm_provider in LLM_PROVIDERS and LLM_PROVIDERS[llm_provider]["package"]:
        dependencies.append(LLM_PROVIDERS[llm_provider]["package"])
    
    # Add Search provider package
    if search_provider != "none" and search_provider in SEARCH_PROVIDERS and SEARCH_PROVIDERS[search_provider]["package"]:
        dependencies.append(SEARCH_PROVIDERS[search_provider]["package"])
    
    # Add Memory provider package
    if memory_provider != "none" and memory_provider in MEMORY_PROVIDERS and MEMORY_PROVIDERS[memory_provider]["package"]:
        dependencies.append(MEMORY_PROVIDERS[memory_provider]["package"])
    
    # Add Logging provider package
    if logging_provider != "none" and logging_provider in LOGGING_PROVIDERS and LOGGING_PROVIDERS[logging_provider]["package"]:
        dependencies.append(LOGGING_PROVIDERS[logging_provider]["package"])
    
    # Add utility packages
    for util in utilities:
        if util in UTILITY_PACKAGES:
            dependencies.append(UTILITY_PACKAGES[util]["package"])
    
    return dependencies


def generate_env_vars(
    llm_provider: str,
    search_provider: str,
    memory_provider: str,
    logging_provider: str
) -> Dict[str, str]:
    """
    Generate environment variables based on selected providers.
    Always includes Daytona environment variables.
    
    Args:
        llm_provider: LLM provider
        search_provider: Search provider
        memory_provider: Memory provider
        logging_provider: Logging provider
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    
    # LLM provider environment variables
    if llm_provider in LLM_PROVIDERS:
        for var in LLM_PROVIDERS[llm_provider]["env_vars"]:
            env_vars[var] = ""
    
    # Search provider environment variables
    if search_provider != "none" and search_provider in SEARCH_PROVIDERS:
        for var in SEARCH_PROVIDERS[search_provider]["env_vars"]:
            env_vars[var] = ""
    
    # Memory provider environment variables
    if memory_provider != "none" and memory_provider in MEMORY_PROVIDERS:
        for var in MEMORY_PROVIDERS[memory_provider]["env_vars"]:
            env_vars[var] = ""
    
    # Logging provider environment variables
    if logging_provider != "none" and logging_provider in LOGGING_PROVIDERS:
        for var in LOGGING_PROVIDERS[logging_provider]["env_vars"]:
            env_vars[var] = ""
    
    # Always include Daytona environment variables
    for var in DAYTONA_CONFIG["env_vars"]:
        env_vars[var] = ""
    
    # Add the FORCE_DAYTONA flag to always use Daytona
    env_vars["FORCE_DAYTONA"] = "true"
    
    return env_vars


def create_new_agent(
    name: str,
    template: str = "basic",
    output_dir: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Create a new agent with the specified name and template.

    Args:
        name: Name of the agent to create
        template: Template to use (default: basic)
        output_dir: Directory to output the agent (default: current directory)
        settings: Optional pre-defined settings (if None, will prompt for settings)
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # Get settings interactively if not provided
    if settings is None:        
        # Get the settings from user input, passing the name to avoid duplicate prompts
        settings = get_agent_settings(name)
    
    # Create agent directory
    agent_dir = os.path.join(output_dir, name)
    os.makedirs(agent_dir, exist_ok=True)

    # Create package directory
    package_dir = os.path.join(agent_dir, settings["package_name"])
    os.makedirs(package_dir, exist_ok=True)

    # Copy and render template files from the template directory
    template_dir = TEMPLATES_DIR / template
    if not template_dir.exists():
        raise ValueError(f"Template '{template}' not found in {TEMPLATES_DIR}")

    # Define template files to render
    template_files = [
        "README.md.jinja",
        "main.py.jinja",
        "requirements.in.jinja", 
        "pyproject.toml.jinja",
        ".env.example.jinja"  # New template file for .env.example
    ]
    
    # Try to render each template file
    for file_name in template_files:
        template_file_path = os.path.join(template_dir, file_name)
        if os.path.exists(template_file_path):
            _render_template_file(template_file_path, agent_dir, settings)
    
    # Handle package files
    pkg_template_dir = template_dir / "{{package_name}}"
    if pkg_template_dir.exists():
        for file_name in os.listdir(pkg_template_dir):
            if file_name.endswith(".jinja"):
                pkg_file_path = os.path.join(pkg_template_dir, file_name)
                _render_template_file(pkg_file_path, package_dir, settings, is_package_file=True)

    # Create .env.example file if the template doesn't exist
    env_example_path = os.path.join(agent_dir, '.env.example')
    if not os.path.exists(env_example_path):
        # Create a default .env.example file
        env_example_content = """# Environment variables for {{agent_name}}
# Replace placeholder values with your actual credentials

"""
        # Add environment variables for selected providers
        for env_var, value in settings["env_vars"].items():
            env_example_content += f"{env_var}={value}\n"
        
        # Always ensure Daytona environment variables are included
        env_example_content += """
# Daytona configuration (required for secure execution)
DAYTONA_API_KEY=your-daytona-api-key
DAYTONA_SERVER_URL=your-daytona-server-url
DAYTONA_TARGET=us

# Force Daytona execution (true/false)
FORCE_DAYTONA=true
"""
        
        with open(env_example_path, 'w') as f:
            f.write(env_example_content)

    # Create .env file with API keys if provided
    env_path = os.path.join(agent_dir, '.env')
    if "api_keys" in settings and settings["api_keys"]:
        # Create a .env file with the provided API keys
        env_content = """# Environment variables for {}
# Generated with actual API keys during setup

""".format(settings["agent_name"])
        
        # Add API keys from settings
        for key, value in settings["api_keys"].items():
            if value:  # Only add if the value is not empty
                env_content += f"{key}={value}\n"
        
        # Add other environment variables without values
        for env_var, value in settings["env_vars"].items():
            if env_var not in settings["api_keys"]:  # Skip if already added as an API key
                env_content += f"{env_var}={value}\n"
        
        # Always ensure Daytona environment variables are included
        env_content += """
# Daytona configuration (required for secure execution)
DAYTONA_API_KEY=
DAYTONA_SERVER_URL=
DAYTONA_TARGET=us

# Force Daytona execution (true/false)
FORCE_DAYTONA=true
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        if HAS_TYPER:
            typer.echo(f"Created {env_path} with API keys")
        else:
            print(f"Created {env_path} with API keys")
    else:
        # No API keys provided, copy .env.example as .env
        if os.path.exists(env_example_path) and not os.path.exists(env_path):
            shutil.copy(env_example_path, env_path)
            if HAS_TYPER:
                typer.echo(f"Created {env_path} (copied from .env.example)")
            else:
                print(f"Created {env_path} (copied from .env.example)")

    # Add .gitignore if it doesn't exist
    gitignore_path = os.path.join(agent_dir, '.gitignore')
    if not os.path.exists(gitignore_path):
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
.env
.venv/
venv/
ENV/
env/

# Distribution / packaging
dist/
build/
*.egg-info/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
logs/
*.log

# Local development files
.DS_Store
"""
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        if HAS_TYPER:
            typer.echo(f"Created {gitignore_path}")
        else:
            print(f"Created {gitignore_path}")


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
        
    if HAS_TYPER:
        typer.echo(f"Created {output_file_path}")
    else:
        print(f"Created {output_file_path}")