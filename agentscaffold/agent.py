"""Base agent implementation for AgentScaffold."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional, Callable, Type, ClassVar, Union
import json
import os
import importlib.util
import sys
import inspect
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
        
    @property
    def daytona(self):
        """Lazy-load the Daytona SDK to avoid import errors if not installed."""
        if self._daytona is None:
            try:
                from daytona_sdk import Daytona, CreateWorkspaceParams, DaytonaConfig
                
                # Try to load .env file if not already loaded
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                except ImportError:
                    pass
                
                # Configure Daytona from environment variables or config
                api_key = os.environ.get("DAYTONA_API_KEY", self.config.get("api_key"))
                if not api_key:
                    raise ValueError("API key is required. Set DAYTONA_API_KEY environment variable or provide in config.")
                    
                daytona_config = DaytonaConfig(
                    api_key=api_key,
                    server_url=os.environ.get("DAYTONA_SERVER_URL", self.config.get("server_url")),
                    target=os.environ.get("DAYTONA_TARGET", self.config.get("target", "us"))
                )
                
                self._daytona = Daytona(daytona_config)
                self._CreateWorkspaceParams = CreateWorkspaceParams
            except ImportError:
                raise ImportError(
                    "The daytona-sdk package is required to use the Daytona runtime. "
                    "Install it with 'pip install daytona-sdk'."
                )
        return self._daytona
    
    def _ensure_dependency(self, package: str, workspace) -> None:
        """Ensure that a dependency exists in the Daytona workspace."""
        try:
            # Check if package is installed
            result = workspace.process.exec(f"python -c 'import {package}'")
            if result.exit_code == 0:
                return
        except Exception:
            pass
        
        # Install the package if not already installed
        try:
            # First check if pip is installed
            pip_check = workspace.process.exec("which pip || which pip3")
            if pip_check.exit_code != 0:
                # Install pip
                workspace.process.exec(
                    "sudo curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && "
                    "python3 get-pip.py --break-system-packages"
                )
            
            # Install the package
            workspace.process.exec(f"python3 -m pip install {package} --break-system-packages")
        except Exception as e:
            print(f"Error installing {package}: {e}")
    
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
                "*.py", 
                "requirements.in", 
                "requirements.txt", 
                "pyproject.toml", 
                "README.md",
                ".env"
            ]
            
            # Process each pattern
            for pattern in essential_patterns:
                # Find files matching the pattern
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
            
            # Upload package directory separately
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
                
                # Upload package files
                package_path = os.path.join(agent_dir, package_name)
                for local_path in Path(package_path).glob("*.py"):
                    if local_path.is_file():
                        try:
                            filename = os.path.basename(local_path)
                            remote_path = f"{pkg_dir}/{filename}"
                            
                            # Upload package file
                            print(f"Uploading package file {package_name}/{filename}...")
                            with open(local_path, "rb") as f:
                                content = f.read()
                            workspace.fs.upload_file(remote_path, content)
                            
                            # Add to successful uploads
                            uploaded_files.append(f"{package_name}/{filename}")
                        except Exception as file_error:
                            print(f"Error uploading package file {local_path}: {file_error}")
            
            # Print summary of uploaded files
            print(f"Successfully uploaded {len(uploaded_files)} files to Daytona workspace")
            
            # List directory contents to verify uploads
            ls_result = workspace.process.exec(f"ls -la {remote_dir}")
            print(f"Directory contents: {ls_result.result}")
            
        except Exception as e:
            print(f"Error uploading agent code: {e}")
            # Print stack trace for debugging
            import traceback
            traceback.print_exc(file=sys.stdout)
    
    def _get_required_packages(self, agent_dir: str) -> List[str]:
        """Get the list of required packages from requirements.in."""
        requirements_file = Path(agent_dir) / "requirements.in"
        if not requirements_file.exists():
            return ["pydantic", "daytona-sdk"]
        
        packages = []
        with open(requirements_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name from line (ignoring version specifiers)
                    package = line.split("#")[0].strip().split(">=")[0].split("==")[0].strip()
                    if package:
                        packages.append(package)
        
        return packages
        
    def execute(self, agent_fn: Union[Callable, str], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent function with the provided input data using Daytona.
        
        Args:
            agent_fn: Agent function to execute or path to agent directory
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        # If agent_fn is a string, it's the path to the agent directory
        if isinstance(agent_fn, str):
            try:
                from daytona_sdk import CreateWorkspaceParams
            except ImportError:
                return {
                    "response": "Daytona SDK not installed. Please install with 'pip install daytona-sdk'",
                    "metadata": {"error": True}
                }
                
            agent_dir = agent_fn
            
            # Create a new Daytona workspace
            try:
                params = CreateWorkspaceParams(language="python")
                self.workspace = self.daytona.create(params)
                
                # Try a simple command to test workspace access
                test_cmd = self.workspace.process.exec("mkdir -p /home/daytona/agent")
                if test_cmd.exit_code != 0:
                    return {
                        "response": f"Error: Cannot create directories in Daytona workspace. Please check permissions. Details: {test_cmd.result}",
                        "metadata": {"error": True}
                    }
                
                # Use home directory instead of /workspace
                print("Using /home/daytona/agent directory for code")
                
                # Ensure required packages are installed
                for package in self._get_required_packages(agent_dir):
                    self._ensure_dependency(package, self.workspace)
                
                # Upload agent code using a simpler approach
                self._simple_upload_agent_code(self.workspace, agent_dir)
                
                # Run the agent in the workspace
                code = f"""
import sys
sys.path.append('/home/daytona/agent')
import json
import os
os.chdir('/home/daytona/agent')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

print("Debug: Current directory:", os.getcwd())
print("Debug: Directory contents:", os.listdir('.'))

# Find the main agent module
try:
    print("Debug: Attempting to import main module...")
    from main import main
    import asyncio
    print("Debug: Running agent asynchronously...")
    result = asyncio.run(main())
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
            input_data = {input_json}
            result = agent.run(input_data)
            print(json.dumps(result))
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Debug: Error running agent: {{str(e)}}")
            print(f"Debug: Traceback: {{traceback_str}}")
            print(json.dumps({{"response": f"Error running agent: {{str(e)}}", "metadata": {{"error": True}}}}))
    else:
        print(f"Debug: No agent package found in directory")
        print(json.dumps({{"response": "Error: No agent package found in directory", "metadata": {{"error": True}}}}))
except Exception as e:
    import traceback
    print(f"Debug: Unexpected error: {{str(e)}}")
    print(f"Debug: Traceback: {{traceback.format_exc()}}")
    print(json.dumps({{"response": f"Error: {{str(e)}}", "metadata": {{"error": True}}}}))
"""
                encoded_input = json.dumps(input_data).replace('"', '\\"')
                code = code.replace("{input_json}", f'"{encoded_input}"')
                
                # Execute the code in the workspace
                response = self.workspace.process.code_run(code)
                
                if response.exit_code != 0:
                    return {
                        "response": f"Error executing agent in Daytona: {response.result}",
                        "metadata": {"error": True, "exit_code": response.exit_code}
                    }
                
                # Try to parse the output as JSON
                try:
                    lines = response.result.strip().split('\n')
                    for line in reversed(lines):
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            continue
                    
                    # If no JSON found, return the raw output
                    return {
                        "response": response.result.strip(),
                        "metadata": {}
                    }
                except Exception as e:
                    return {
                        "response": f"Error parsing agent output: {str(e)}",
                        "metadata": {"error": True, "output": response.result}
                    }
                
            except Exception as e:
                return {
                    "response": f"Error running agent in Daytona: {str(e)}",
                    "metadata": {"error": True}
                }
            finally:
                # Cleanup the workspace if created
                if self.workspace:
                    try:
                        self.daytona.remove(self.workspace)
                    except Exception:
                        pass
        else:
            # Use the agent function directly
            return agent_fn(input_data)


class Agent(BaseModel):
    """Base agent class for AgentScaffold."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    description: str = ""
    input_class: ClassVar[Type[AgentInput]] = AgentInput
    output_class: ClassVar[Type[AgentOutput]] = AgentOutput
    runtime: Optional[DaytonaRuntime] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.runtime is None:
            self.runtime = DaytonaRuntime()
    
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
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the provided input data.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        # Validate input
        validated_input = self.input_class(**input_data).model_dump()
        
        # Execute agent using runtime
        result = self.runtime.execute(self.process, validated_input)
        
        # Validate output
        output = self.output_class(**result)
        
        return output.model_dump()