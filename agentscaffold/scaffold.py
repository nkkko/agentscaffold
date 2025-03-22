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
        "description": "OpenAI API (GPT-4, etc.)"
    },
    "anthropic": {
        "env_vars": ["ANTHROPIC_API_KEY"],
        "package": "anthropic>=0.5.0",
        "description": "Anthropic API (Claude)"
    },
    "huggingface": {
        "env_vars": ["HUGGINGFACE_API_KEY"],
        "package": "huggingface_hub>=0.16.0",
        "description": "HuggingFace Inference API"
    },
    "ollama": {
        "env_vars": ["OLLAMA_HOST"],
        "package": "ollama>=0.1.0",
        "description": "Local LLMs with Ollama"
    },
    "daytona": {
        "env_vars": ["DAYTONA_API_KEY", "DAYTONA_SERVER_URL", "DAYTONA_TARGET"],
        "package": "daytona-sdk>=0.1.0",
        "description": "Daytona Managed LLM API"
    }
}

SEARCH_PROVIDERS = {
    "brave": {
        "env_vars": ["BRAVE_API_KEY"],
        "package": "brave-search>=0.1.0",
        "description": "Brave Search API"
    },
    "browserbase": {
        "env_vars": ["BROWSERBASE_API_KEY"],
        "package": "browserbase>=0.1.0",
        "description": "BrowserBase API"
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
    "milvus": {
        "env_vars": ["MILVUS_URI"],
        "package": "pymilvus>=2.2.0",
        "description": "Milvus vector database"
    },
    "chromadb": {
        "env_vars": ["CHROMA_DB_PATH"],
        "package": "chromadb>=0.4.0",
        "description": "ChromaDB local vector database"
    },
    "none": {
        "env_vars": [],
        "package": None,
        "description": "No memory provider"
    }
}

LOGGING_PROVIDERS = {
    "logfire": {
        "env_vars": ["LOGFIRE_API_KEY"],
        "package": "logfire>=0.1.0",
        "description": "LogFire observability platform"
    },
    "none": {
        "env_vars": [],
        "package": None,
        "description": "No logging provider"
    }
}

UTILITY_PACKAGES = {
    "pypdf": {
        "package": "pypdf>=3.0.0",
        "description": "PDF parsing utility"
    },
    "puppeteer": {
        "package": "pyppeteer>=1.0.0",
        "description": "Headless Chrome/Chromium automation"
    },
    "dotenv": {
        "package": "python-dotenv>=1.0.0",
        "description": "Environment variable management"
    },
    "requests": {
        "package": "requests>=2.0.0",
        "description": "HTTP requests library"
    },
    "beautifulsoup": {
        "package": "beautifulsoup4>=4.0.0",
        "description": "HTML parsing library"
    },
    "playwright": {
        "package": "playwright>=1.0.0",
        "description": "Browser automation library"
    }
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
    llm_default = defaults.get('llm_provider', 'daytona')
    llm_provider = prompt_choice("Select LLM provider:", llm_provider_choices, default=llm_default)
    
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
    utility_defaults = defaults.get('utilities', ['dotenv'])
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
        "env_vars": generate_env_vars(llm_provider, search_provider, memory_provider, logging_provider)
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
) -> List[str]:
    """
    Generate environment variables based on selected providers.
    
    Args:
        llm_provider: LLM provider
        search_provider: Search provider
        memory_provider: Memory provider
        logging_provider: Logging provider
        
    Returns:
        List of environment variables
    """
    env_vars = []
    
    # LLM provider environment variables
    if llm_provider in LLM_PROVIDERS:
        for var in LLM_PROVIDERS[llm_provider]["env_vars"]:
            env_vars.append(f"{var}=")
    
    # Search provider environment variables
    if search_provider != "none" and search_provider in SEARCH_PROVIDERS:
        for var in SEARCH_PROVIDERS[search_provider]["env_vars"]:
            env_vars.append(f"{var}=")
    
    # Memory provider environment variables
    if memory_provider != "none" and memory_provider in MEMORY_PROVIDERS:
        for var in MEMORY_PROVIDERS[memory_provider]["env_vars"]:
            env_vars.append(f"{var}=")
    
    # Logging provider environment variables
    if logging_provider != "none" and logging_provider in LOGGING_PROVIDERS:
        for var in LOGGING_PROVIDERS[logging_provider]["env_vars"]:
            env_vars.append(f"{var}=")
    
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
        for env_var in settings["env_vars"]:
            env_example_content += f"{env_var}\n"
        
        # Add Daytona environment variables
        env_example_content += """
# Daytona configuration (required for 'agentscaffold run')
DAYTONA_API_KEY=your-daytona-api-key
DAYTONA_SERVER_URL=your-daytona-server-url
DAYTONA_TARGET=us
"""
        
        with open(env_example_path, 'w') as f:
            f.write(env_example_content)

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
        
    if HAS_TYPER:
        typer.echo(f"Created {output_file_path}")