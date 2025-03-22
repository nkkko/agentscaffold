"""Base agent implementation for AgentScaffold."""

import asyncio
import inspect
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional, Callable, Type, ClassVar, Union
import json
import os
import importlib.util
import sys
from pathlib import Path


class AgentInput(BaseModel):
    """Base class for agent inputs."""
    
    message: str = Field(..., description="Input message for the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class AgentOutput(BaseModel):
    """Base class for agent outputs."""
    
    response: str = Field(..., description="Response from the agent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DaytonaRuntime:
    """Daytona runtime for agent execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.workspace = None
        self._daytona = None
        self._CreateWorkspaceParams = None
        print("DaytonaRuntime initialized")
        
        # Check environment variables early to provide better feedback
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            print("WARNING: DAYTONA_API_KEY environment variable is not set!")
            print("Daytona execution will not be available without an API key.")
        else:
            print(f"Found Daytona API key (length: {len(api_key)})")
            
        server_url = os.environ.get("DAYTONA_SERVER_URL")
        if server_url:
            print(f"Daytona server URL: {server_url}")
        
        target = os.environ.get("DAYTONA_TARGET", "us")
        print(f"Daytona target: {target}")
        
    @property
    def daytona(self):
        """Lazy-load the Daytona SDK to avoid import errors if not installed."""
        if self._daytona is None:
            try:
                print("\n=== Daytona SDK Initialization ===")
                print("Attempting to load Daytona SDK...")
                try:
                    from daytona_sdk import Daytona, CreateWorkspaceParams, DaytonaConfig
                    print("Successfully imported daytona_sdk module")
                except ImportError as e:
                    print(f"ERROR: Failed to import daytona_sdk: {e}")
                    print("Make sure the Daytona SDK is installed: pip install daytona-sdk")
                    self._daytona = None
                    return None
                
                # Try to load .env file if not already loaded
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    print("Loaded environment variables from .env file")
                except ImportError:
                    print("Warning: python-dotenv not installed, using environment variables directly")
                    pass
                
                # Configure Daytona from environment variables or config
                api_key = os.environ.get("DAYTONA_API_KEY", self.config.get("api_key"))
                if not api_key:
                    print("ERROR: No Daytona API key found in environment variables or config.")
                    print("Set the DAYTONA_API_KEY environment variable or provide it in the config.")
                    raise ImportError("No Daytona API key found")
                    
                daytona_config = DaytonaConfig(
                    api_key=api_key,
                    server_url=os.environ.get("DAYTONA_SERVER_URL", self.config.get("server_url")),
                    target=os.environ.get("DAYTONA_TARGET", self.config.get("target", "us"))
                )
                
                print(f"Daytona config:")
                print(f"  - API Key: {'*'*8}{api_key[-4:] if len(api_key) > 4 else '****'}")
                print(f"  - Server URL: {daytona_config.server_url or 'default'}")
                print(f"  - Target: {daytona_config.target}")
                
                try:
                    self._daytona = Daytona(daytona_config)
                    self._CreateWorkspaceParams = CreateWorkspaceParams
                    print("Daytona SDK initialized successfully!")
                    print("=== End Daytona SDK Initialization ===\n")
                except Exception as init_error:
                    print(f"ERROR: Failed to initialize Daytona client: {init_error}")
                    self._daytona = None
                    return None
                
            except Exception as e:
                print(f"Unexpected error initializing Daytona SDK: {e}")
                print("Falling back to local execution")
                import traceback
                traceback.print_exc()
                self._daytona = None
                return None
                
        return self._daytona
    
    def _ensure_dependency(self, package: str, workspace) -> None:
        """Ensure that a dependency exists in the Daytona workspace."""
        try:
            # Check if package is installed
            result = workspace.process.exec(f"python -c 'import {package}'")
            if result.exit_code == 0:
                print(f"Package {package} is already installed")
                return
        except Exception as e:
            print(f"Error checking if {package} is installed: {e}")
        
        # Install the package if not already installed
        try:
            # First check if pip is installed
            pip_check = workspace.process.exec("which pip || which pip3")
            if pip_check.exit_code != 0:
                # Install pip
                print("Installing pip...")
                workspace.process.exec(
                    "curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && "
                    "python3 get-pip.py --break-system-packages"
                )
            
            # Install the package
            print(f"Installing {package}...")
            result = workspace.process.exec(f"python3 -m pip install {package} --break-system-packages")
            print(f"Install result for {package}: {result.result}")
        except Exception as e:
            print(f"Error installing {package}: {e}")
    
    def _prepare_daytona_compatible_agent(self, agent_path: str) -> str:
        """
        Create a modified version of the agent that will work in Daytona without the full agentscaffold package.
        
        Args:
            agent_path: Path to the original agent.py file
            
        Returns:
            Modified agent code as a string
        """
        # Predefined simple Daytona-compatible agent implementation
        daytona_agent_code = """# Daytona-compatible agent implementation
    # Contains no ClassVar annotations which are incompatible with standard Pydantic in remote environment

    import json
    import os

    from pydantic import BaseModel, Field, ConfigDict
    from typing import Dict, Any, Optional, List, Union

    # Basic agent classes that don't rely on external imports
    class AgentInput(BaseModel):
        \"\"\"Base input model for agents.\"\"\"
        message: str = Field(..., description="Input message for the agent")
        context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    class AgentOutput(BaseModel):
        \"\"\"Base output model for agents.\"\"\"
        response: str = Field(..., description="Response from the agent")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class DaytonaTestAgentInput(AgentInput):
        \"\"\"Input for DaytonaTestAgent agent.\"\"\"
        # Custom input fields can be added here
        pass

    class DaytonaTestAgentOutput(AgentOutput):
        \"\"\"Output for DaytonaTestAgent agent.\"\"\"
        # Custom output fields can be added here
        pass

    class DaytonaTestAgentPydanticResult(BaseModel):
        \"\"\"Result from Pydantic AI Agent.\"\"\"
        message: str = Field(description="Response message")
        additional_info: Dict[str, Any] = Field(default_factory=dict, description="Additional information")

    class BaseAgent(BaseModel):
        \"\"\"Base agent implementation.\"\"\"
        model_config = ConfigDict(arbitrary_types_allowed=True)
        
        name: str = "BaseAgent"
        description: str = ""
        
        def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            \"\"\"Process input data and generate a response.\"\"\"
            # Default implementation just echoes the input
            return {
                "response": f"Received: {input_data['message']}",
                "metadata": {}
            }

    class PydanticAgent:
        \"\"\"Minimal PydanticAgent implementation for Daytona.\"\"\"
        
        def __init__(self, model_name, result_type=None, system_prompt=None):
            self.model_name = model_name
            self.result_type = result_type
            self.system_prompt = system_prompt
        
        async def run(self, message):
            \"\"\"Run the agent with the message.\"\"\"
            try:
                if self.model_name.startswith("openai:"):
                    import openai
                    client = openai.OpenAI()
                    response = client.chat.completions.create(
                        model=self.model_name.split(":", 1)[1],
                        messages=[
                            {"role": "system", "content": self.system_prompt or "You are a helpful assistant."},
                            {"role": "user", "content": message}
                        ]
                    )
                    result_text = response.choices[0].message.content
                else:
                    # Fallback response
                    result_text = f"I received your message: {message}."
                
                # Create a simple result object
                result_data = {"message": result_text, "additional_info": {}}
                
                class MinimalResult:
                    def __init__(self, data):
                        self.data = type('ResultData', (), data)
                
                return MinimalResult(result_data)
            except Exception as e:
                # Fallback for errors
                print(f"Error processing message: {e}")
                result_data = {"message": f"I received your message, but encountered an error.", "additional_info": {"error": str(e)}}
                
                class MinimalResult:
                    def __init__(self, data):
                        self.data = type('ResultData', (), data)
                
                return MinimalResult(result_data)

    class Agent(BaseAgent):
        \"\"\"DaytonaTestAgent implementation.\"\"\"
        
        # Use ConfigDict to set model config
        model_config = ConfigDict(arbitrary_types_allowed=True)
        
        name: str = "DaytonaTestAgent"
        description: str = "A daytona-test-agent agent"
        
        def __init__(self, **data):
            super().__init__(**data)
            try:
                self.pydantic_agent = PydanticAgent(
                    "openai:gpt-4o",
                    result_type=DaytonaTestAgentPydanticResult,
                    system_prompt=(
                        "You are DaytonaTestAgent, an AI assistant designed to help with "
                        "daytona-test-agent. Be helpful, concise, and accurate in your responses."
                    )
                )
            except Exception as e:
                print(f"Warning: Error initializing PydanticAgent: {e}")
                self.pydantic_agent = PydanticAgent("fallback", result_type=DaytonaTestAgentPydanticResult)
        
        async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            \"\"\"
            Process input data and generate output.
            
            Args:
                input_data: Validated input data
                
            Returns:
                Agent output
            \"\"\"
            try:
                metadata = {}
                
                # Process with Pydantic AI
                try:
                    result = await self.pydantic_agent.run(input_data["message"])
                    response_message = result.data.message
                    additional_info = result.data.additional_info
                except Exception as e:
                    print(f"Error in pydantic_agent.run: {e}")
                    # Fallback response
                    response_message = f"I received your message: {input_data['message']}. Running on Daytona!"
                    additional_info = {"error": str(e)}
                
                # Return response with metadata
                return {
                    "response": response_message,
                    "metadata": {**additional_info, **metadata}
                }
                
            except Exception as e:
                # Handle any errors
                return {
                    "response": f"Error: {e}",
                    "metadata": {"error": True}
                }
    """
        
        # Extract agent package and class names from the original
        # (These will be different for each agent)
        import re
        try:
            with open(agent_path, 'r') as f:
                original_code = f.read()
                
            # Extract agent name from the original file - replace placeholders in our template
            package_match = re.search(r'class\s+(\w+)Input\(', original_code)
            agent_name = package_match.group(1) if package_match else "DaytonaTestAgent"
            
            # Replace placeholders in our template
            daytona_agent_code = daytona_agent_code.replace("DaytonaTestAgent", agent_name)
            
            # Replace package description
            desc_match = re.search(r'description: str = "([^"]+)"', original_code)
            if desc_match:
                description = desc_match.group(1)
                daytona_agent_code = daytona_agent_code.replace("A daytona-test-agent agent", description)
        except Exception as e:
            print(f"Error extracting information from original agent: {e}")
        
        return daytona_agent_code

def _simple_upload_agent_code(self, workspace, agent_dir: str) -> None:
    """Upload agent code to the Daytona workspace using a simpler approach."""
    from pathlib import Path
    import sys
    import os
    
    # Create the workspace directory structure in the home directory
    remote_dir = "/home/daytona/agent"
    try:
        # Use mkdir -p to create the directory (should work in home directory)
        result = workspace.process.exec(f"mkdir -p {remote_dir}")
        if result.exit_code != 0:
            print(f"Warning: Failed to create directory {remote_dir}: {result.result}")
            return
        
        # List of files uploaded successfully for debugging
        uploaded_files = []
        
        # Upload essential files only (skip .venv and other large directories)
        essential_patterns = [
            "requirements.in", 
            "requirements.txt", 
            "pyproject.toml", 
            "README.md",
            ".env"
        ]
        
        # Process each pattern except agent.py which we'll handle specially
        for pattern in essential_patterns:
            # Find files matching the pattern in the root directory
            for local_path in Path(agent_dir).glob(pattern):
                if local_path.is_file():
                    try:
                        filename = os.path.basename(local_path)
                        remote_path = f"{remote_dir}/{filename}"
                        
                        # Upload the file directly to remote dir
                        print(f"Uploading {filename}...")
                        with open(local_path, "rb") as f:
                            content = f.read()
                        workspace.fs.upload_file(remote_path, content)
                        
                        # Add to successful uploads
                        uploaded_files.append(filename)
                    except Exception as file_error:
                        print(f"Error uploading file {local_path}: {file_error}")
        
        # Create minimal __init__.py if needed
        remote_init_path = f"{remote_dir}/__init__.py"
        workspace.fs.upload_file(remote_init_path, b"# Package initialization")
        uploaded_files.append("__init__.py")
        
        # Handle agent.py specially - we'll create a Daytona-compatible version
        agent_path = Path(agent_dir) / "agent.py"
        if agent_path.exists():
            try:
                print("Creating Daytona-compatible version of agent.py...")
                
                # Use helper method to create a compatible version
                if hasattr(self, '_prepare_daytona_compatible_agent'):
                    agent_content = self._prepare_daytona_compatible_agent(agent_path)
                else:
                    # Create a standard template if helper not available
                    with open(agent_path, "r") as f:
                        agent_content = f.read()
                        agent_content = "# Standard agent.py without modifications\n" + agent_content
                
                # Upload the Daytona-compatible agent.py
                remote_agent_path = f"{remote_dir}/agent.py"
                workspace.fs.upload_file(remote_agent_path, agent_content.encode())
                uploaded_files.append("agent.py")
                
                print("Successfully created and uploaded Daytona-compatible agent.py")
            except Exception as agent_error:
                print(f"Error preparing Daytona-compatible agent.py: {agent_error}")
                # Fallback to uploading the original agent.py
                with open(agent_path, "rb") as f:
                    content = f.read()
                workspace.fs.upload_file(f"{remote_dir}/agent.py", content)
                uploaded_files.append("agent.py")
        
        # Create main.py if it doesn't exist
        main_path = Path(agent_dir) / "main.py"
        if not main_path.exists():
            print("Creating main.py file in workspace...")
            # Extract package name from directory
            dir_parts = agent_dir.split(os.sep)
            if dir_parts:
                package_name = dir_parts[-1]
                print(f"Using package name from directory: {package_name}")
                
                # Create a more robust main.py that can handle missing imports better
                main_py = f"""
import asyncio
import sys
import os
import traceback

# Add current directory to path to help with imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the agent from the package
try:
    from {package_name}.agent import Agent
    print(f"Successfully imported Agent from {package_name}.agent")
except ImportError as e:
    print(f"Error importing Agent from {package_name}.agent: {{e}}")
    # Try alternative imports
    try:
        # Look for agent.py in the current directory
        from agent import Agent
        print("Successfully imported Agent from agent.py")
    except ImportError as e2:
        print("Could not import Agent from agent.py either")
        print(f"Import error details: {{e2}}")
        sys.exit(1)

async def main(input_data):
    \"\"\"
    Main entry point for the agent.
    
    Args:
        input_data: Input data for the agent
        
    Returns:
        Agent output
    \"\"\"
    print("Starting main() function")
    try:
        agent = Agent()
        print(f"Agent initialized: {{agent.name}}")
        result = await agent.process(input_data)
        return result
    except Exception as e:
        print(f"Error in main: {{e}}")
        traceback.print_exc()
        return {{"response": f"Error: {{str(e)}}", "metadata": {{"error": True}}}}

if __name__ == "__main__":
    # This code runs when the script is executed directly
    test_input = {{
        "message": "Test message",
        "context": {{}}
    }}
    asyncio.run(main(test_input))
"""
                workspace.fs.upload_file(f"{remote_dir}/main.py", main_py.encode())
                uploaded_files.append("main.py")
        else:
            # Upload existing main.py
            print(f"Uploading main.py...")
            with open(main_path, "rb") as f:
                content = f.read()
            workspace.fs.upload_file(f"{remote_dir}/main.py", content)
            uploaded_files.append("main.py")
        
        # Create package directory
        package_name = None
        for item in os.listdir(agent_dir):
            package_path = os.path.join(agent_dir, item)
            if os.path.isdir(package_path) and os.path.exists(os.path.join(package_path, "__init__.py")):
                package_name = item
                print(f"Found package: {package_name}")
                break
        
        if package_name:
            # Create package directory
            pkg_dir = f"{remote_dir}/{package_name}"
            workspace.process.exec(f"mkdir -p {pkg_dir}")
            
            # Create package __init__.py
            workspace.fs.upload_file(f"{pkg_dir}/__init__.py", b"# Package initialization")
            uploaded_files.append(f"{package_name}/__init__.py")
            
            # Copy our agent.py to the package directory
            workspace.fs.upload_file(f"{pkg_dir}/agent.py", agent_content.encode())
            uploaded_files.append(f"{package_name}/agent.py")
        else:
            # If no package found, create one with the same name as the directory
            dir_parts = agent_dir.split(os.sep)
            if dir_parts:
                package_name = dir_parts[-1].replace('-', '_')  # Replace dashes with underscores for valid Python package name
                print(f"No package found, creating one with name: {package_name}")
                pkg_dir = f"{remote_dir}/{package_name}"
                workspace.process.exec(f"mkdir -p {pkg_dir}")
                
                # Create __init__.py in the package directory
                workspace.fs.upload_file(f"{pkg_dir}/__init__.py", b"# Package initialization")
                uploaded_files.append(f"{package_name}/__init__.py")
                
                # Copy our agent.py to the package
                workspace.fs.upload_file(f"{pkg_dir}/agent.py", agent_content.encode())
                uploaded_files.append(f"{package_name}/agent.py")
        
        # Print summary of uploaded files
        print(f"Successfully uploaded {len(uploaded_files)} files to Daytona workspace")
        
        # List directory contents to verify uploads
        ls_result = workspace.process.exec(f"ls -la {remote_dir}")
        print(f"Directory contents: {ls_result.result}")
        
        # If package was created, list its contents too
        if package_name:
            pkg_ls_result = workspace.process.exec(f"ls -la {remote_dir}/{package_name}")
            print(f"Package directory contents: {pkg_ls_result.result}")
        
    except Exception as e:
        print(f"Error uploading agent code: {e}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc(file=sys.stdout)
    
    def _get_required_packages(self, agent_dir: str) -> List[str]:
        """Get the list of required packages from requirements.in."""
        requirements_file = Path(agent_dir) / "requirements.in"
        if not requirements_file.exists():
            # Make sure to include agentscaffold if needed
            return ["pydantic", "pydantic-core", "daytona-sdk", "python-dotenv", "openai"]
        
        # Always include these essential packages
        packages = ["pydantic", "pydantic-core", "daytona-sdk", "python-dotenv", "openai"]
        
        with open(requirements_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name from line (ignoring version specifiers)
                    package = line.split("#")[0].strip().split(">=")[0].split("==")[0].strip()
                    # Skip agentscaffold since it's not yet published

                    if package and package not in packages and package != "agentscaffold":  
                        packages.append(package)
        
        return packages
        
    async def execute(self, agent_fn: Union[Callable, str], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent function with the provided input data using Daytona.
        Using the more robust code_run approach with proper string escaping.
        
        Args:
            agent_fn: Agent function to execute or path to agent directory
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        print("DaytonaRuntime.execute called")
        
        # For function-based agents, we will still force Daytona by getting the module path
        if not isinstance(agent_fn, str):
            print("Function-based agent provided, but forcing Daytona execution...")
            
            # Get the module file path where the function is defined
            import inspect
            func_module = inspect.getmodule(agent_fn)
            if func_module is None:
                print("WARNING: Cannot determine module for function, falling back to direct execution")
                return agent_fn(input_data)
            
            func_file = func_module.__file__
            if func_file is None:
                print("WARNING: Cannot determine file path for function, falling back to direct execution")
                return agent_fn(input_data)
            
            # Use the directory instead of the function
            agent_dir = os.path.dirname(os.path.abspath(func_file))
            print(f"Using function's directory for Daytona: {agent_dir}")
        else:
            agent_dir = agent_fn
            print(f"Using provided agent directory: {agent_dir}")
        
        # Try to use Daytona, with clear error messages if it fails
        # Check if Daytona SDK is available
        if not self.daytona:
            print("ERROR: Daytona SDK initialization failed, but Daytona execution is required")
            print("Please check your Daytona API key and installation")
            return {
                "response": "ERROR: Daytona execution is required but not available. Check your API key and daytona-sdk installation.",
                "metadata": {"error": True, "daytona_required": True}
            }
        
        # Create a new Daytona workspace
        try:
            print("Creating Daytona workspace...")
            params = self._CreateWorkspaceParams(language="python")
            self.workspace = self.daytona.create(params)
            print(f"Workspace created: {self.workspace.id}")
            
            # Test workspace access
            test_cmd = self.workspace.process.exec("mkdir -p /home/daytona/agent")
            if test_cmd.exit_code != 0:
                print(f"Error: Cannot create directories in Daytona workspace: {test_cmd.result}")
                return {
                    "response": f"Error: Cannot create directories in Daytona workspace: {test_cmd.result}",
                    "metadata": {"error": True}
                }
            
            # Upload agent code
            print("Uploading agent code...")
            self._simple_upload_agent_code(self.workspace, agent_dir)
            
            # Ensure required packages are installed
            required_packages = self._get_required_packages(agent_dir)
            print(f"Installing required packages: {required_packages}")
            for package in required_packages:
                self._ensure_dependency(package, self.workspace)
            
            # Generate Python code to run the agent
            print("Generating Python code to run agent...")
            
            # Convert input_data to a base64 string to avoid all escaping issues
            import base64
            input_json = json.dumps(input_data)
            input_base64 = base64.b64encode(input_json.encode('utf-8')).decode('utf-8')
            
            code = f"""
import sys
sys.path.append('/home/daytona/agent')
import json
import os
import base64
os.chdir('/home/daytona/agent')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

print("Debug: Current directory:", os.getcwd())
print("Debug: Directory contents:", os.listdir('.'))

# Parse the input data using base64 to avoid escaping issues
input_base64 = "{input_base64}"
input_json = base64.b64decode(input_base64).decode('utf-8')
input_data = json.loads(input_json)

# Find the main agent module
try:
    print("Debug: Attempting to import main module...")
    from main import main
    import asyncio
    print("Debug: Running agent asynchronously...")
    result = asyncio.run(main(input_data))
    print(json.dumps(result))
except (ImportError, AttributeError) as e:
    print(f"Debug: ImportError or AttributeError: {{str(e)}}")
    # Try to import the agent directly
    import os
    from importlib import import_module
    
    print("Debug: Looking for agent package...")
    for item in os.listdir('.'):
        print(f"Debug: Found item: {{item}} (is_dir: {{os.path.isdir(item)}})")
        if os.path.isdir(item):
            init_path = os.path.join(item, '__init__.py')
            print(f"Debug: Checking for __init__.py in {{item}}: {{os.path.exists(init_path)}}")
    
    # Find the package directory (the first directory with __init__.py)
    agent_package = None
    for item in os.listdir('.'):
        if os.path.isdir(item) and os.path.exists(os.path.join(item, '__init__.py')):
            agent_package = item
            print(f"Debug: Found agent package: {{agent_package}}")
            break
    
    if agent_package:
        try:
            print(f"Debug: Importing {{agent_package}}.agent")
            agent_module = import_module(f"{{agent_package}}.agent")
            print(f"Debug: Creating agent instance")
            agent = agent_module.Agent()
            print(f"Debug: Running agent with input data")
            import asyncio
            result = asyncio.run(agent.run(input_data))
            print(json.dumps(result))
        except Exception as e:
            # Try to import directly from agent.py in root directory as fallback
            try:
                print("Debug: Trying to import agent.py from root directory")
                from agent import Agent
                print("Debug: Creating agent instance from root agent.py")
                agent = Agent()
                print(f"Debug: Running agent with input data")
                import asyncio
                result = asyncio.run(agent.run(input_data))
                print(json.dumps(result))
            except ImportError as ie:
                import traceback
                traceback_str = traceback.format_exc()
                print(f"Debug: Error importing from root agent.py: {{str(ie)}}")
                print(f"Debug: Original error running agent: {{str(e)}}")
                print(f"Debug: Traceback: {{traceback_str}}")
                print(json.dumps({{"response": f"Error running agent: {{str(e)}}", "metadata": {{"error": True}}}}))
    else:
        # Try to import directly from agent.py in root directory as fallback
        try:
            print("Debug: No package found, trying to import agent.py from root directory")
            from agent import Agent
            print("Debug: Creating agent instance from root agent.py")
            agent = Agent()
            print(f"Debug: Running agent with input data")
            import asyncio
            result = asyncio.run(agent.run(input_data))
            print(json.dumps(result))
        except ImportError as ie:
            print(f"Debug: Error importing from root agent.py: {{str(ie)}}")
            print(f"Debug: No agent package found in directory")
            print(json.dumps({{"response": "Error: No agent package or agent.py found in directory", "metadata": {{"error": True}}}}))
except Exception as e:
    import traceback
    print(f"Debug: Unexpected error: {{str(e)}}")
    print(f"Debug: Traceback: {{traceback.format_exc()}}")
    print(json.dumps({{"response": f"Error: {{str(e)}}", "metadata": {{"error": True}}}}))
"""
            
            print("Running agent code...")
            # Using code_run instead of exec for more reliable execution
            response = self.workspace.process.code_run(code)
            
            if response.exit_code != 0:
                print(f"Error executing agent in Daytona: {response.exit_code}")
                print(f"Command output: {response.result}")
                return {
                    "response": f"Error executing agent in Daytona (exit code {response.exit_code}): {response.result}",
                    "metadata": {"error": True, "exit_code": response.exit_code, "output": response.result}
                }
            
            # Parse the response
            print("Parsing response from Daytona workspace")
            print(f"Response output (truncated): {response.result[:200]}...")
            
            # Parse the output as JSON (look for JSON in the output lines)
            lines = response.result.strip().split('\n')
            for line in reversed(lines):
                try:
                    result = json.loads(line)
                    print("Successfully parsed JSON response")
                    return result
                except json.JSONDecodeError:
                    continue
            
            # If no JSON found, return the raw output
            print("No JSON found in response, returning raw output")
            return {
                "response": response.result.strip(),
                "metadata": {"raw_output": True}
            }
            
        except Exception as e:
            print(f"Error running agent in Daytona: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "response": f"Error running agent in Daytona: {str(e)}",
                "metadata": {"error": True}
            }
        finally:
            # Cleanup the workspace if created
            if self.workspace:
                try:
                    print(f"Cleaning up Daytona workspace: {self.workspace.id}")
                    self.daytona.remove(self.workspace)
                except Exception as e:
                    print(f"Error cleaning up workspace: {e}")

class Agent(BaseModel):
    """Base agent class for AgentScaffold."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = "BaseAgent"  # Provide a default value for name
    description: str = ""
    input_class: ClassVar[Type[AgentInput]] = AgentInput
    output_class: ClassVar[Type[AgentOutput]] = AgentOutput
    runtime: Optional[DaytonaRuntime] = None

    # Initialize runtime as a class property with a getter method
    _runtime_instance = None
    @property
    def runtime(self):
        if Agent._runtime_instance is None:
            Agent._runtime_instance = DaytonaRuntime()
        return Agent._runtime_instance

    
    def __init__(self, **data):
        # Handle missing name field by providing a default
        if "name" not in data:
            data["name"] = "BaseAgent"
            
        super().__init__(**data)
        # self.runtime = DaytonaRuntime()
        print(f"Agent initialized: {self.name}")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and generate output.
        
        This method should be overridden by derived agent classes.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Agent output
        """
        # Default implementation just echoes the input
        return {"response": f"Received: {input_data['message']}", "metadata": {}}
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the provided input data.
        Always forces execution in Daytona, never runs locally.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        print(f"Agent.run called with input: {input_data.get('message', '')[:50]}...")
        
        # Handle string input
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                input_data = {"message": input_data}
        
        # Validate input
        try:
            validated_input = self.input_class(**input_data).model_dump()
        except Exception as e:
            print(f"Error validating input: {str(e)}")
            # Create a minimal valid input as fallback
            validated_input = {"message": input_data.get("message", "Error: Invalid input"), "context": {}}
    
        print("Forcing Daytona execution for all requests...")
            
        try:
            # Use the Daytona runtime to execute the agent
            import os
            import inspect
            
            # Get the module file path where the agent class is defined
            agent_module = inspect.getmodule(self.__class__)
            if agent_module is None:
                raise RuntimeError("Cannot determine agent module")
            
            agent_file = agent_module.__file__
            if agent_file is None:
                raise RuntimeError("Cannot determine agent file path")
            
            agent_dir = os.path.dirname(os.path.abspath(agent_file))
            print(f"Using agent directory path: {agent_dir}")
            
            # Execute the agent via the improved execute method
            result = await self.runtime.execute(agent_dir, validated_input)
        except Exception as e:
            print(f"Error in Daytona execution: {str(e)}")
            import traceback
            traceback.print_exc()
            result = {
                    "response": f"Error processing request in Daytona: {str(e)}",
                    "metadata": {"error": True}
            }

        print(f"Daytona result: {result.get('response', '')[:50]}...")
        
        # Validate output
        try:
            output = self.output_class(**result)
            return output.model_dump()
        except Exception as output_error:
            print(f"Error validating output: {str(output_error)}")
            # Return the raw result if validation fails
            return result
    """Base agent class for AgentScaffold."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = "BaseAgent"  # Provide a default value for name
    description: str = ""
    input_class: ClassVar[Type[AgentInput]] = AgentInput
    output_class: ClassVar[Type[AgentOutput]] = AgentOutput
    runtime: Optional[DaytonaRuntime] = None
    
    def __init__(self, **data):
        # Handle missing name field by providing a default
        if "name" not in data:
            data["name"] = "BaseAgent"
            
        super().__init__(**data)
        self.runtime = DaytonaRuntime()
        print(f"Agent initialized: {self.name}")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and generate output.
        
        This method should be overridden by derived agent classes.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Agent output
        """
        # Default implementation just echoes the input
        return {"response": f"Received: {input_data['message']}", "metadata": {}}
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the provided input data.
        Always executes in Daytona, never locally.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        print(f"Agent.run called with input: {input_data.get('message', '')[:50]}...")
        
        # Handle string input
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                input_data = {"message": input_data}
        
        # Validate input using the input class
        try:
            validated_input = self.input_class(**input_data).model_dump()
        except Exception as e:
            print(f"Error validating input: {str(e)}")
            # Fallback: create a minimal valid input
            validated_input = {"message": input_data.get("message", "Error: invalid input"), "context": {}}
        
        # Force Daytona execution by always skipping local processing
        print("Forcing Daytona execution for all requests...")
        
        try:
            # Use the Daytona runtime to execute the agent
            import os
            import inspect
            
            # Get the module file path where the agent class is defined
            agent_module = inspect.getmodule(self.__class__)
            if agent_module is None:
                raise RuntimeError("Cannot determine agent module")
            
            agent_file = agent_module.__file__
            if agent_file is None:
                raise RuntimeError("Cannot determine agent file path")
            
            agent_dir = os.path.dirname(os.path.abspath(agent_file))
            print(f"Using agent directory path: {agent_dir}")
            
            # Execute the agent via Daytona
            if self.runtime is None:
                from agentscaffold.agent import DaytonaRuntime
                self.runtime = DaytonaRuntime()
                
            result = await self.runtime.execute(agent_dir, validated_input)
        except Exception as e:
            print(f"Error in Daytona execution: {str(e)}")
            import traceback
            traceback.print_exc()
            result = {
                "response": f"Error processing request in Daytona: {str(e)}",
                "metadata": {"error": True}
            }
        
        print(f"Daytona result: {result.get('response', '')[:50]}...")
        
        # Validate output
        try:
            output = self.output_class(**result)
            return output.model_dump()
        except Exception as output_error:
            print(f"Error validating output: {str(output_error)}")
            # Return the raw result if validation fails
            return result
            """
            Run the agent with the provided input data.
            
            Args:
                input_data: Input data for the agent
                
            Returns:
                Agent output
            """
            print(f"Agent.run called with input: {input_data.get('message', '')[:50]}...")
            
            # Handle string input
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    input_data = {"message": input_data}
            
            # Validate input using the input class
            try:
                validated_input = self.input_class(**input_data).model_dump()
            except Exception as e:
                print(f"Error validating input: {str(e)}")
                # Fallback: create a minimal valid input
                validated_input = {"message": input_data.get("message", "Error: invalid input"), "context": {}}
            
            # Try to run locally first
            try:
                # Important: Await the process method if it's async, otherwise call directly
                if inspect.iscoroutinefunction(self.process):
                    result = await self.process(validated_input)
                else:
                    result = self.process(validated_input)
                    
                # Validate output
                return self.output_class(**result).model_dump()
            except Exception as local_error:
                print(f"Error in local processing: {str(local_error)}")
                print("Falling back to Daytona execution...")
                
                try:
                    # Use the Daytona runtime to execute the agent
                    import os
                    import inspect
                    
                    # Get the module file path where the agent class is defined
                    agent_module = inspect.getmodule(self.__class__)
                    if agent_module is None:
                        raise RuntimeError("Cannot determine agent module")
                    
                    agent_file = agent_module.__file__
                    if agent_file is None:
                        raise RuntimeError("Cannot determine agent file path")
                    
                    agent_dir = os.path.dirname(os.path.abspath(agent_file))
                    print(f"Using agent directory path: {agent_dir}")
                    
                    # Execute the agent via the improved execute method
                    if self.runtime is None:
                        from agentscaffold.agent import DaytonaRuntime
                        self.runtime = DaytonaRuntime()
                        
                    result = await self.runtime.execute(agent_dir, validated_input)
                except Exception as e:
                    print(f"Error in agent.run: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    result = {
                        "response": f"Error processing request: {str(e)}",
                        "metadata": {"error": True}
                    }
            
            print(f"Process result: {result.get('response', '')[:50]}...")
            
            # Validate output
            try:
                output = self.output_class(**result)
                return output.model_dump()
            except Exception as output_error:
                print(f"Error validating output: {str(output_error)}")
                # Return the raw result if validation fails
                return result