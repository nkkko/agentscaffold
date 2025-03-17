"""Base agent implementation for AgentScaffold."""

import asyncio
import inspect
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
        print("DaytonaRuntime initialized")
        
    @property
    def daytona(self):
        """Lazy-load the Daytona SDK to avoid import errors if not installed."""
        if self._daytona is None:
            try:
                print("Attempting to load Daytona SDK...")
                from daytona_sdk import Daytona, CreateWorkspaceParams, DaytonaConfig
                
                # Try to load .env file if not already loaded
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                except ImportError:
                    print("Warning: dotenv not installed, using environment variables directly")
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
                
                print(f"Daytona config: server={daytona_config.server_url}, target={daytona_config.target}")
                self._daytona = Daytona(daytona_config)
                self._CreateWorkspaceParams = CreateWorkspaceParams
                print("Daytona SDK loaded successfully")
            except ImportError as e:
                print(f"Error loading Daytona SDK: {e}")
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
            return ["pydantic", "daytona-sdk", "pydantic-ai", "python-dotenv", "agentscaffold"]
        
        # Always include these essential packages
        packages = ["pydantic", "daytona-sdk", "pydantic-ai", "python-dotenv", "agentscaffold"]
        
        with open(requirements_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name from line (ignoring version specifiers)
                    package = line.split("#")[0].strip().split(">=")[0].split("==")[0].strip()
                    if package and package not in packages:  # Avoid duplicates
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
        print("DaytonaRuntime.execute called")
        # Always try to use Daytona if available, regardless of agent_fn type
        try:
            # Check if Daytona SDK is available first
            from daytona_sdk import CreateWorkspaceParams
            
            print("Daytona SDK is available")
            
            # Get the agent directory if agent_fn is a callable
            if not isinstance(agent_fn, str):
                # Try to get the directory from the module
                try:
                    module = inspect.getmodule(agent_fn)
                    if module:
                        agent_dir = os.path.dirname(os.path.dirname(module.__file__))
                    else:
                        # Fallback to current directory
                        agent_dir = os.getcwd()
                except Exception as e:
                    print(f"Error getting agent directory: {e}")
                    agent_dir = os.getcwd()
            else:
                agent_dir = agent_fn
            
            print(f"Using agent directory: {agent_dir}")
            
            # Create a new Daytona workspace
            try:
                print("Creating Daytona workspace...")
                params = CreateWorkspaceParams(language="python")
                self.workspace = self.daytona.create(params)
                print(f"Workspace created: {self.workspace.id}")
                
                # Try a simple command to test workspace access
                test_cmd = self.workspace.process.exec("mkdir -p /home/daytona/agent")
                if test_cmd.exit_code != 0:
                    print(f"Error: Cannot create directories in Daytona workspace: {test_cmd.result}")
                    return {
                        "response": f"Error: Cannot create directories in Daytona workspace. Please check permissions. Details: {test_cmd.result}",
                        "metadata": {"error": True}
                    }
                
                # Use home directory instead of /workspace
                print("Using /home/daytona/agent directory for code")
                
                # Ensure required packages are installed
                required_packages = self._get_required_packages(agent_dir)
                print(f"Installing required packages: {required_packages}")
                for package in required_packages:
                    self._ensure_dependency(package, self.workspace)
                
                # Upload agent code using a simpler approach
                print("Uploading agent code...")
                self._simple_upload_agent_code(self.workspace, agent_dir)
                
                # Run the agent in the workspace
                print("Running agent in Daytona workspace...")
                # Use a template string instead of f-string to avoid issues with placeholders
                template = """
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
    print(f"Debug: ImportError or AttributeError: {str(e)}")
    # Try to import the agent directly
    import os
    from importlib import import_module
    
    print("Debug: Looking for agent package...")
    for item in os.listdir('.'):
        print(f"Debug: Found item: {item} (is_dir: {os.path.isdir(item)})")
        if os.path.isdir(item):
            init_path = os.path.join(item, '__init__.py')
            print(f"Debug: Checking for __init__.py in {item}: {os.path.exists(init_path)}")
    
    # Find the package directory (the first directory with __init__.py)
    agent_package = None
    for item in os.listdir('.'):
        if os.path.isdir(item) and os.path.exists(os.path.join(item, '__init__.py')):
            agent_package = item
            print(f"Debug: Found agent package: {agent_package}")
            break
    
    if agent_package:
        try:
            print(f"Debug: Importing {agent_package}.agent")
            agent_module = import_module(f"{agent_package}.agent")
            print(f"Debug: Creating agent instance")
            agent = agent_module.Agent()
            print(f"Debug: Running agent with input data")
            input_data = {input_json_placeholder}
            result = agent.run(input_data)
            print(json.dumps(result))
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Debug: Error running agent: {str(e)}")
            print(f"Debug: Traceback: {traceback_str}")
            print(json.dumps({"response": f"Error running agent: {str(e)}", "metadata": {"error": True}}))
    else:
        print(f"Debug: No agent package found in directory")
        print(json.dumps({"response": "Error: No agent package found in directory", "metadata": {"error": True}}))
except Exception as e:
    import traceback
    print(f"Debug: Unexpected error: {str(e)}")
    print(f"Debug: Traceback: {traceback.format_exc()}")
    print(json.dumps({"response": f"Error: {str(e)}", "metadata": {"error": True}}))
"""
                # Replace the placeholder with the actual data
                encoded_input = json.dumps(input_data).replace('"', '\\"')
                code = template.replace("{input_json_placeholder}", f'json.loads("{encoded_input}")')
                
                # Execute the code in the workspace
                print("Executing code in Daytona workspace...")
                response = self.workspace.process.code_run(code)
                
                if response.exit_code != 0:
                    print(f"Error executing agent in Daytona: {response.exit_code}, {response.result}")
                    return {
                        "response": f"Error executing agent in Daytona: {response.result}",
                        "metadata": {"error": True, "exit_code": response.exit_code}
                    }
                
                # Try to parse the output as JSON
                try:
                    print("Parsing response from Daytona workspace")
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
                        "metadata": {}
                    }
                except Exception as e:
                    print(f"Error parsing agent output: {str(e)}")
                    return {
                        "response": f"Error parsing agent output: {str(e)}",
                        "metadata": {"error": True, "output": response.result}
                    }
                
            except Exception as e:
                print(f"Error running agent in Daytona: {str(e)}")
                # Try to execute the function directly as fallback
                if not isinstance(agent_fn, str):
                    print("Falling back to direct execution")
                    return agent_fn(input_data)
                else:
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
        except ImportError as e:
            print(f"Daytona SDK not available: {e}")
            # If Daytona is not available, use the function directly
            if not isinstance(agent_fn, str):
                print("Daytona not available, using direct execution")
                return agent_fn(input_data)
            else:
                return {
                    "response": "Daytona SDK not installed. Please install with 'pip install daytona-sdk'",
                    "metadata": {"error": True}
                }
        except Exception as e:
            print(f"Unexpected error in DaytonaRuntime.execute: {e}")
            # Fallback to direct execution if possible
            if not isinstance(agent_fn, str):
                print("Falling back to direct execution due to error")
                return agent_fn(input_data)
            else:
                return {
                    "response": f"Error: {str(e)}",
                    "metadata": {"error": True}
                }


class Agent(BaseModel):
    """Base agent class for AgentScaffold."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str
    description: str = ""
    input_class: ClassVar[Type[AgentInput]] = AgentInput
    output_class: ClassVar[Type[AgentOutput]] = AgentOutput
    runtime: Optional[DaytonaRuntime] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Always create a DaytonaRuntime instance
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
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        print(f"Agent.run called with input: {input_data.get('message', '')[:50]}...")
        
        # Validate input
        validated_input = self.input_class(**input_data).model_dump()
        
        # Always run through the Daytona runtime regardless of whether process is sync or async
        # Get the current working directory to use for Daytona execution
        cwd = os.getcwd()
        
        try:
            # Check if process is an asynchronous function
            if inspect.iscoroutinefunction(self.process):
                print("Using async process method with Daytona runtime")
                if self.runtime:
                    # For async methods, we need to pass the current directory path
                    result = await asyncio.to_thread(self.runtime.execute, cwd, validated_input)
                else:
                    # Fallback to direct execution if runtime is not available
                    result = await self.process(validated_input)
            else:
                # For sync methods, we can pass the method directly
                print("Using sync process method with Daytona runtime")
                if self.runtime:
                    result = await asyncio.to_thread(self.runtime.execute, self.process, validated_input)
                else:
                    # Fallback to direct execution if runtime is not available
                    result = self.process(validated_input)
        except Exception as e:
            print(f"Error in agent.run: {str(e)}")
            # Create a fallback response in case of errors
            result = {
                "response": f"Error processing request: {str(e)}",
                "metadata": {"error": True}
            }
        
        print(f"Process result: {result.get('response', '')[:50]}...")
        
        # Validate output
        output = self.output_class(**result)
        
        return output.model_dump()