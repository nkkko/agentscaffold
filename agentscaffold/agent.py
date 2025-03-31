"""Base agent implementation for AgentScaffold."""

import asyncio
import inspect
import logging
import json
import os
from typing import Dict, Any, List, Optional, Union, ClassVar, Type
from pathlib import Path

# Import Pydantic
from pydantic import BaseModel, Field, ConfigDict


class AgentInput(BaseModel):
    """Base class for agent inputs."""
    message: str = Field(..., description="Input message for the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class AgentOutput(BaseModel):
    """Base class for agent outputs."""
    response: str = Field(..., description="Response from the agent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseAgent:
    """Base agent class that all scaffolded agents inherit from."""
    
    def __init__(
        self,
        name: str = "Base Agent",
        template_dir: str = "flask",
        description: str = "A base agent with minimal functionality",
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            **kwargs: Additional configuration options
        """
        self.name = name
        self.description = description
        self.config = kwargs
        self.template_dir = template_dir  # Store the template_dir as instance variable
        
        # Setup logging
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"agent.{self.__class__.__name__}")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        
        # Initialize providers
        self.llm_provider = self._init_llm()
        self.search_provider = self._init_search()
        self.memory_provider = self._init_memory()
        self.logging_provider = self._init_logging()
        self.mcp_provider = self._init_mcp()
        
        # Set input and output classes
        self.input_class = AgentInput
        self.output_class = AgentOutput
        
        # Set runtime configuration
        self.silent_mode = False
        
        # Initialize DaytonaRuntime
        self._runtime = DaytonaRuntime()
        
        self.logger.info(f"Initialized {self.__class__.__name__} agent")
    
    def set_silent_mode(self, silent: bool = True):
        """Set the agent to silent mode."""
        self.silent_mode = silent
        if hasattr(self, "_runtime") and hasattr(self._runtime, "set_silent_mode"):
            self._runtime.set_silent_mode(silent)
        
    def _init_llm(self):
        """Initialize LLM provider."""
        self.logger.info("No default LLM provider")
        return None
    
    def _init_search(self):
        """Initialize search provider."""
        self.logger.info("No default search provider")
        return None
    
    def _init_memory(self):
        """Initialize memory provider."""
        self.logger.info("No default memory provider")
        return None
    
    def _init_logging(self):
        """Initialize logging provider."""
        self.logger.info("No default logging provider")
        return None
    
    def _init_mcp(self):
        """Initialize MCP provider."""
        self.logger.info("No default MCP provider")
        return None
    
    @property
    def runtime(self):
        """Get the DaytonaRuntime instance."""
        return self._runtime

    
    def get_flask_templates_path(self, template_name=None):
        """
        Get path to Flask template files.
        
        Args:
            template_name: Optional specific template name
            
        Returns:
            Path to template directory or specific template file
        """
        from pathlib import Path
        
        template_dir = Path(__file__).parent / "templates" / "flask"
        
        if template_name:
            return template_dir / template_name
        
        return template_dir  
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the given input.
        
        Args:
            input_data: Input data (message and optional context)
            
        Returns:
            Agent response with metadata
        """
        if isinstance(input_data, str):
            input_data = {"message": input_data}
            
        if isinstance(input_data, dict) and input_data.get('message', '').lower().strip() in ['exit', 'quit', 'bye']:
            self.logger.info("Received exit command, cleaning up resources...")
            if hasattr(self, "_runtime") and hasattr(self._runtime, "_cleanup_workspace"):
                self._runtime._cleanup_workspace()
            return {"response": "Session ended. All resources have been cleaned up.", "metadata": {"exited": True}}
        
        if not isinstance(input_data, dict):
            self.logger.error(f"Invalid input type: {type(input_data)}")
            return {"response": "Error: Invalid input type", "metadata": {"error": True}}
        
        if "message" not in input_data:
            self.logger.error("No message in input")
            return {"response": "Error: No message provided", "metadata": {"error": True}}
        
        try:
            import time
            start_time = time.time()
            self.logger.info("Running agent in Daytona environment")
            try:
                agent_module = inspect.getmodule(self.__class__)
                if agent_module is None:
                    raise RuntimeError("Cannot determine agent module")
                agent_file = agent_module.__file__
                if agent_file is None:
                    raise RuntimeError("Cannot determine agent file path")
                agent_dir = os.path.dirname(os.path.abspath(agent_file))
                self.logger.info(f"Using agent directory: {agent_dir}")
                result = await self._runtime.execute(agent_dir, input_data)
                processing_time = time.time() - start_time
                if isinstance(result, dict) and "metadata" in result:
                    result["metadata"]["processing_time"] = processing_time
                return result
            except Exception as e:
                self.logger.error(f"❌ Daytona execution error: {e}")
                self.logger.info("Falling back to local execution due to Daytona error")
                result = await self.process(input_data)
                processing_time = time.time() - start_time
                if isinstance(result, dict) and "metadata" in result:
                    result["metadata"]["processing_time"] = processing_time
                return result
        except Exception as e:
            self.logger.error(f"Error in run: {e}", exc_info=True)
            return {"response": f"Error: {str(e)}", "metadata": {"error": True}}
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback local processing."""
        message = input_data.get("message", "")
        return {"response": f"Received: {message}", "metadata": {"default": True}}
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Default search for: {query}")
        return []
    
    async def remember(self, query: str = "") -> str:
        self.logger.info(f"Default memory retrieval for: {query}")
        return ""
    
    async def store_memory(self, data: Dict[str, Any]) -> bool:
        self.logger.info("Default memory storage")
        return False
    
    async def log_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        self.logger.info(f"Default logging for event: {event_type}")
        return False


class DaytonaRuntime:
    """Daytona runtime for agent execution with persistent workspace and search capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.workspace = None
        self._daytona = None
        self._CreateWorkspaceParams = None
        self._is_initialized = False
        self._agent_dir = None
        self._conversation_id = None
        self._api_keys = {}
        self.silent_mode = False
        
        self._load_env_vars()
        if not os.environ.get("DAYTONA_API_KEY"):
            print("⚠️ DAYTONA_API_KEY environment variable is required")
        else:
            print(f"✅ Loaded .env (DAYTONA_API_KEY=***{os.environ.get('DAYTONA_API_KEY')[-4:] if len(os.environ.get('DAYTONA_API_KEY', '')) > 4 else ''})")
        if not os.environ.get("DAYTONA_API_URL"):
            print("⚠️ DAYTONA_API_URL environment variable is required")
        self._collect_api_keys()
        self._load_daytona_sdk()
        
    def set_silent_mode(self, silent: bool = True):
        self.silent_mode = silent

    def _print(self, message: str, end="\n", flush=False):
        if not self.silent_mode:
            print(message, end=end, flush=flush)
    
    def _load_env_vars(self):
        try:
            from dotenv import load_dotenv
            for env_path in ['.env', '../.env']:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    break
        except ImportError:
            for env_path in ['.env', '../.env']:
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#') and '=' in line:
                                key, value = line.strip().split('=', 1)
                                os.environ[key.strip()] = value.strip()
    
    def _collect_api_keys(self):
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            self._api_keys["OPENAI_API_KEY"] = openai_api_key
            print(f"✅ Found OpenAI API key (length: {len(openai_api_key)})")
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self._api_keys["ANTHROPIC_API_KEY"] = anthropic_api_key
            print(f"✅ Found Anthropic API key (length: {len(anthropic_api_key)})")
        brave_api_key = os.environ.get("BRAVE_API_KEY")
        if brave_api_key:
            self._api_keys["BRAVE_API_KEY"] = brave_api_key
            print(f"✅ Found Brave Search API key (length: {len(brave_api_key)})")
        logfire_api_key = os.environ.get("LOGFIRE_API_KEY")
        if logfire_api_key:
            self._api_keys["LOGFIRE_API_KEY"] = logfire_api_key
            print(f"✅ Found LogFire API key (length: {len(logfire_api_key)})")
    
    def _load_daytona_sdk(self):
        try:
            from daytona_sdk import Daytona, CreateWorkspaceParams, DaytonaConfig
            api_key = os.environ.get("DAYTONA_API_KEY", self.config.get("api_key"))
            server_url = os.environ.get("DAYTONA_API_URL", self.config.get("server_url"))
            target = os.environ.get("DAYTONA_TARGET", self.config.get("target", "us"))
            if not api_key:
                print("❌ No Daytona API key found")
                return
            daytona_config = DaytonaConfig(api_key=api_key, server_url=server_url, target=target)
            self._daytona = Daytona(daytona_config)
            self._CreateWorkspaceParams = CreateWorkspaceParams
        except ImportError as e:
            print(f"❌ daytona-sdk package not installed: {e}")
        except Exception as e:
            print(f"❌ Daytona initialization failed: {e}")
    
    def _init_workspace(self, agent_dir: str) -> bool:
        if self.workspace is not None:
            print("🔄 Reusing existing workspace")
            return True
        if not self._daytona:
            print("❌ Daytona SDK not initialized")
            return False
        try:
            params = self._CreateWorkspaceParams(language="python")
            self.workspace = self._daytona.create(params)
            print(f"🔧 Workspace ID: {self.workspace.id}")
            if not self._conversation_id:
                import uuid
                self._conversation_id = str(uuid.uuid4())
                print(f"🆕 New conversation ID: {self._conversation_id}")
            return True
        except Exception as e:
            print(f"❌ Error creating workspace: {e}")
            return False
    
    def _upload_agent_code(self, agent_dir: str) -> None:
        if self._is_initialized and self._agent_dir == agent_dir:
            print("🔄 Reusing existing code upload")
            return
        self._agent_dir = agent_dir
        try:
            self.workspace.process.exec("mkdir -p /home/daytona/agent")
            print("Creating .env file with API keys...")
            env_content = ""
            for key, value in self._api_keys.items():
                env_content += f"{key}={value}\n"
            self.workspace.fs.upload_file("/home/daytona/agent/.env", env_content.encode('utf-8'))
            print("Successfully uploaded .env file with API keys")
            self._is_initialized = True
        except Exception as e:
            print(f"Error uploading agent code: {e}")
    
    def _cleanup_workspace(self):
        if not self.workspace:
            return
        try:
            print("🧹 Cleaning up workspace...")
            self._daytona.remove(self.workspace)
            self.workspace = None
            self._is_initialized = False
        except Exception as e:
            print(f"❌ Error cleaning up workspace: {e}")
            
    def _get_persistent_conversation_id(self, agent_dir: str) -> str:
        """Get a persistent conversation ID across sessions."""
        # Use a file to store the conversation ID
        conversation_file = os.path.join(agent_dir, ".conversation_id")
        
        if os.path.exists(conversation_file):
            try:
                with open(conversation_file, 'r') as f:
                    conversation_id = f.read().strip()
                    if conversation_id:
                        print(f"🔄 Using existing conversation ID: {conversation_id}")
                        return conversation_id
            except Exception as e:
                print(f"Error reading conversation ID: {e}")
        
        # If no existing conversation ID, generate a new one
        import uuid
        conversation_id = str(uuid.uuid4())
        try:
            with open(conversation_file, 'w') as f:
                f.write(conversation_id)
            print(f"🆕 Created new conversation ID: {conversation_id}")
        except Exception as e:
            print(f"Error saving conversation ID: {e}")
        
        return conversation_id
    
    def _prepare_execution_code(self, input_base64, conversation_id):
        """
        Generate optimized execution code with memory, logging, and MCP capabilities.
        This code is uploaded and executed in the Daytona workspace.
        """
        return f"""
import sys, os, json, base64, traceback, asyncio
from typing import Dict, Any, Optional, List
import time, uuid, urllib.request, urllib.parse

# Ensure agent directory exists
if not os.path.exists('/home/daytona/agent'):
    print("Creating agent directory...")
    os.makedirs('/home/daytona/agent', exist_ok=True)

sys.path.append('/home/daytona/agent')
sys.path.append('/home/daytona')
os.chdir('/home/daytona/agent')

# Load environment variables
try:
    from dotenv import load_dotenv
    if os.path.exists('/home/daytona/agent/.env'):
        load_dotenv('/home/daytona/agent/.env')
        print("✅ Loaded environment variables in Daytona")
    else:
        print("No .env file found in agent directory")
except ImportError:
    if os.path.exists('/home/daytona/agent/.env'):
        with open('/home/daytona/agent/.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Manually loaded environment variables in Daytona")
    else:
        print("No .env file found in agent directory")

print("Current directory contents:")
try:
    print(os.listdir('.'))
except Exception as e:
    print(f"Error listing directory: {{e}}")

# Check for API keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    print(f"Found OpenAI API key in Daytona (length: {{len(openai_api_key)}})")
brave_api_key = os.environ.get("BRAVE_API_KEY")
if brave_api_key:
    print(f"Found Brave API key in Daytona (length: {{len(brave_api_key)}})")
logfire_api_key = os.environ.get("LOGFIRE_API_KEY")
if logfire_api_key:
    print(f"Found LogFire API key in Daytona (length: {{len(logfire_api_key)}})")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
if anthropic_api_key:
    print(f"Found Anthropic API key in Daytona (length: {{len(anthropic_api_key)}})")

CONVERSATION_ID = "{conversation_id}"
print(f"🔄 Processing message in conversation: {{CONVERSATION_ID}}")

input_base64 = "{input_base64}"
input_json = base64.b64decode(input_base64).decode('utf-8')
input_data = json.loads(input_json)
message = input_data.get('message', '')
print(f"📩 Received message: '{{message}}'")

# --- MEMORY IMPLEMENTATION ---
conversation_history = []

def read_conversation_history():
    history_file = '/home/daytona/agent/conversation_history.json'
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading conversation history: {{e}}")
    return []

def save_conversation_history(history):
    history_file = '/home/daytona/agent/conversation_history.json'
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f)
        print(f"Saved {{len(history)}} conversation entries to file")
    except Exception as e:
        print(f"Error saving conversation history: {{e}}")

conversation_history = read_conversation_history()
print(f"Loaded {{len(conversation_history)}} conversation entries from history")

def add_to_memory(text, metadata=None):
    metadata = metadata or {{}}
    entry = {{
        "id": str(uuid.uuid4()),
        "text": text,
        "timestamp": time.time(),
        "metadata": metadata
    }}
    conversation_history.append(entry)
    save_conversation_history(conversation_history)
    return entry["id"]

def get_context(query, n_results=3):
    if not conversation_history:
        return ""
    relevant = []
    for entry in conversation_history:
        if query.lower() in entry['text'].lower():
            relevant.append(entry)
    if not relevant:
        relevant = sorted(conversation_history, key=lambda x: x.get('timestamp', 0), reverse=True)[:n_results]
    else:
        relevant = relevant[:n_results]
    if relevant:
        return "\\n\\n".join([f"Memory {{i+1}}:\\n{{m['text']}}" for i, m in enumerate(relevant)])
    return ""

# --- MCP IMPLEMENTATION ---
def invoke_mcp(server_id, input_data):
    print(f"Invoking MCP server: {{server_id}}")
    
    if server_id == "brave-search" or "search" in server_id:
        return invoke_brave_search(input_data.get("query", ""))
    elif server_id == "chroma-memory" or "memory" in server_id:
        operation = input_data.get("operation", "retrieve")
        if operation == "store":
            data = input_data.get("data", {{}})
            return store_in_memory(data)
        else:
            query = input_data.get("query", "")
            return retrieve_from_memory(query)
    elif server_id == "logfire-logging" or "logging" in server_id:
        event = input_data.get("event", "custom_event")
        data = input_data.get("data", {{}})
        return log_to_logfire(event, data)
    else:
        return {{"status": "error", "message": f"Unknown MCP server: {{server_id}}"}}

def invoke_brave_search(query):
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
         return {{"status": "error", "message": "No Brave API key provided"}}
         
    params = urllib.parse.urlencode({{"q": query, "count": 3}})
    url = f"https://api.search.brave.com/res/v1/web/search?{{params}}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("X-Subscription-Token", api_key)
    
    try:
         with urllib.request.urlopen(req, timeout=10) as response:
             data = json.loads(response.read().decode('utf-8'))
             results = []
             if "web" in data and "results" in data["web"]:
                 for result in data["web"]["results"][:3]:
                     results.append({{
                         "title": result.get("title", ""),
                         "url": result.get("url", ""),
                         "snippet": result.get("description", "")
                     }})
             return {{"status": "success", "results": results}}
    except Exception as e:
         return {{"status": "error", "message": str(e), "results": []}}

def store_in_memory(data):
    if not isinstance(data, dict):
        data = {{"content": str(data)}}
    
    memory_id = data.get("id", f"mem_{{int(time.time())}}")
    content = data.get("content", str(data))
    
    # Store in conversation history
    memory_entry = f"Memory: {{content}}"
    add_to_memory(memory_entry, {{"type": "stored_memory", "data": data}})
    
    return {{"status": "success", "message": "Stored in memory", "id": memory_id}}

def retrieve_from_memory(query):
    context = get_context(query)
    
    if context:
        memories = []
        for entry in conversation_history:
            if query.lower() in json.dumps(entry).lower():
                memories.append(entry.get("metadata", {{}}).get("data", {{"content": entry.get("text", "")}}))
        
        return {{"status": "success", "results": memories, "context": context}}
    else:
        return {{"status": "error", "message": "No relevant memories found", "results": []}}"""



    async def execute(self, agent_dir: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent code in the Daytona workspace.
        
        Args:
            agent_dir: Directory containing agent code
            input_data: Input data for the agent
            
        Returns:
            Agent response
        """
        if not self._daytona:
            print("❌ Daytona SDK not initialized")
            return {"response": "Error: Daytona SDK not initialized", "metadata": {"error": True}}
        
        try:
            # Initialize workspace if needed
            if not self._init_workspace(agent_dir):
                return {"response": "Error: Failed to initialize workspace", "metadata": {"error": True}}
            
            # Upload agent code
            self._upload_agent_code(agent_dir)
            
            # Ensure conversation ID is set
            if not self._conversation_id:
                self._conversation_id = self._get_persistent_conversation_id(agent_dir)
            
            # Prepare input data
            import base64, json
            input_json = json.dumps(input_data)
            input_base64 = base64.b64encode(input_json.encode('utf-8')).decode('utf-8')
            
            # Generate execution code
            execution_code = self._prepare_execution_code(input_base64, self._conversation_id)
            # 
            # Execute agent code
            self._print("🚀 Executing agent in Daytona workspace...")
            response = self.workspace.process.exec("cd /home/daytona/agent && python3 -c 'import asyncio; from agent import BaseAgent; asyncio.run(BaseAgent().process(input_data))'", 
                                                  environment=self._api_keys,
                                                  stdout=True, stderr=True)
            
            # # Process response
            # try:
            #     response_data = json.loads(response)
            #     return response_data
            # except json.JSONDecodeError:
            #     # If not JSON, return as plain text
            #     return {"response": response, "metadata": {"raw_response": True}}
            return response
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self._print(f"❌ Error executing agent: {e}")
            self._print(error_traceback)
            return {"response": f"Error executing agent: {str(e)}", "metadata": {"error": True, "traceback": error_traceback}}
    
    async def search(self, query: str, provider: str = "brave") -> List[Dict[str, Any]]:
        """
        Perform a search using the specified provider.
        
        Args:
            query: Search query
            provider: Search provider to use (brave, google, etc.)
            
        Returns:
            List of search results
        """
        if not self.workspace:
            self._print("❌ Workspace not initialized")
            return []
        
        try:
            # Use MCP implementation in workspace
            search_code = f"""
import json, os, urllib.request, urllib.parse

def search_with_{provider}(query):
    api_key = os.environ.get("{provider.upper()}_API_KEY")
    if not api_key:
        return {{"status": "error", "message": "No {provider} API key provided"}}
        
    if "{provider}" == "brave":
        params = urllib.parse.urlencode({{"q": query, "count": 5}})
        url = f"https://api.search.brave.com/res/v1/web/search?{{params}}"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("X-Subscription-Token", api_key)
    else:
        return {{"status": "error", "message": "Unsupported search provider"}}
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = []
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"][:5]:
                    results.append({{
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("description", "")
                    }})
            return {{"status": "success", "results": results}}
    except Exception as e:
        return {{"status": "error", "message": str(e), "results": []}}

result = search_with_{provider}("{query}")
print(json.dumps(result))
            """
            
            response = self.workspace.process.exec(f"python3 -c '{search_code}'", 
                                                  environment=self._api_keys,
                                                  stdout=True, stderr=True)
            
            try:
                response_data = json.loads(response)
                if response_data.get("status") == "success":
                    return response_data.get("results", [])
                else:
                    self._print(f"❌ Search error: {response_data.get('message', 'Unknown error')}")
                    return []
            except json.JSONDecodeError:
                self._print(f"❌ Invalid search response: {response}")
                return []
                
        except Exception as e:
            self._print(f"❌ Error performing search: {e}")
            return []
    
    async def store_memory(self, data: Dict[str, Any]) -> bool:
        """
        Store data in agent memory.
        
        Args:
            data: Data to store
            
        Returns:
            True if successful, False otherwise
        """
        if not self.workspace:
            self._print("❌ Workspace not initialized")
            return False
        
        try:
            import json
            # Convert to JSON string
            data_json = json.dumps(data)
            
            # Use memory implementation in workspace
            memory_code = f"""
import json, os, time, uuid

def store_in_memory(data):
    try:
        memory_file = '/home/daytona/agent/memory.json'
        memory_data = []
        
        # Load existing memory if present
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                try:
                    memory_data = json.load(f)
                except:
                    memory_data = []
        
        # Prepare entry
        memory_id = data.get("id", str(uuid.uuid4()))
        entry = {{
            "id": memory_id,
            "timestamp": time.time(),
            "data": data
        }}
        
        # Add to memory
        memory_data.append(entry)
        
        # Save memory
        with open(memory_file, 'w') as f:
            json.dump(memory_data, f)
            
        return {{"status": "success", "id": memory_id}}
    except Exception as e:
        return {{"status": "error", "message": str(e)}}

# Parse data
try:
    data = json.loads('{data_json}')
except Exception as e:
    print(json.dumps({{"status": "error", "message": f"Invalid JSON: {{str(e)}}"}}))
    exit(1)
    
result = store_in_memory(data)
print(json.dumps(result))
            """
            
            response = self.workspace.process.exec(f"python3 -c '{memory_code}'", 
                                                  environment=self._api_keys,
                                                  stdout=True, stderr=True)
            
            try:
                response_data = json.loads(response)
                if response_data.get("status") == "success":
                    return True
                else:
                    self._print(f"❌ Memory storage error: {response_data.get('message', 'Unknown error')}")
                    return False
            except json.JSONDecodeError:
                self._print(f"❌ Invalid memory response: {response}")
                return False
                
        except Exception as e:
            self._print(f"❌ Error storing memory: {e}")
            return False
    
    async def retrieve_memory(self, query: str = "") -> List[Dict[str, Any]]:
        """
        Retrieve data from agent memory.
        
        Args:
            query: Optional query to filter memory
            
        Returns:
            List of memory entries
        """
        if not self.workspace:
            self._print("❌ Workspace not initialized")
            return []
        
        try:
            # Use memory implementation in workspace
            memory_code = f"""
import json, os

def retrieve_from_memory(query=""):
    try:
        memory_file = '/home/daytona/agent/memory.json'
        
        # Check if memory exists
        if not os.path.exists(memory_file):
            return {{"status": "error", "message": "No memory found", "results": []}}
        
        # Load memory
        with open(memory_file, 'r') as f:
            try:
                memory_data = json.load(f)
            except:
                return {{"status": "error", "message": "Invalid memory format", "results": []}}
        
        # Filter by query if provided
        if query:
            results = []
            query_lower = query.lower()
            for entry in memory_data:
                entry_json = json.dumps(entry).lower()
                if query_lower in entry_json:
                    results.append(entry)
        else:
            # Return all memory entries
            results = memory_data
            
        return {{"status": "success", "results": results}}
    except Exception as e:
        return {{"status": "error", "message": str(e), "results": []}}

result = retrieve_from_memory("{query}")
print(json.dumps(result))
            """
            
            response = self.workspace.process.exec(f"python3 -c '{memory_code}'", 
                                                  stdout=True, stderr=True)
            
            try:
                response_data = json.loads(response)
                if response_data.get("status") == "success":
                    return response_data.get("results", [])
                else:
                    self._print(f"❌ Memory retrieval error: {response_data.get('message', 'Unknown error')}")
                    return []
            except json.JSONDecodeError:
                self._print(f"❌ Invalid memory response: {response}")
                return []
                
        except Exception as e:
            self._print(f"❌ Error retrieving memory: {e}")
            return []
    
    async def log_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Log an event using logging provider.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.workspace:
            self._print("❌ Workspace not initialized")
            return False
        
        try:
            import json
            # Convert to JSON string
            data_json = json.dumps(data)
            
            # Use logging implementation in workspace
            logging_code = f"""
import json, os, time, urllib.request, urllib.parse

def log_to_logfire(event_type, data):
    api_key = os.environ.get("LOGFIRE_API_KEY")
    if not api_key:
        return {{"status": "error", "message": "No LogFire API key provided"}}
    
    try:
        # Prepare log entry
        log_entry = {{
            "timestamp": time.time(),
            "event": event_type,
            "data": data
        }}
        
        # Log to LogFire API
        url = "https://in.logfire.dev/ingest"
        req = urllib.request.Request(url)
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {{api_key}}")
        
        log_data = json.dumps([log_entry]).encode('utf-8')
        
        with urllib.request.urlopen(req, log_data, timeout=10) as response:
            if response.status == 200:
                return {{"status": "success"}}
            else:
                return {{"status": "error", "message": f"LogFire API returned status {{response.status}}"}}
    except Exception as e:
        return {{"status": "error", "message": str(e)}}

# Parse data
try:
    data = json.loads('{data_json}')
except Exception as e:
    print(json.dumps({{"status": "error", "message": f"Invalid JSON: {{str(e)}}"}}))
    exit(1)
    
result = log_to_logfire("{event_type}", data)
print(json.dumps(result))
            """
            
            response = self.workspace.process.exec(f"python3 -c '{logging_code}'", 
                                                  environment=self._api_keys,
                                                  stdout=True, stderr=True)
            
            try:
                response_data = json.loads(response)
                if response_data.get("status") == "success":
                    return True
                else:
                    self._print(f"❌ Logging error: {response_data.get('message', 'Unknown error')}")
                    return False
            except json.JSONDecodeError:
                self._print(f"❌ Invalid logging response: {response}")
                return False
                
        except Exception as e:
            self._print(f"❌ Error logging event: {e}")
            return False
    
    async def invoke_mcp(self, server_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke an MCP server.
        
        Args:
            server_id: Server ID
            input_data: Input data for the server
            
        Returns:
            Server response
        """
        if not self.workspace:
            self._print("❌ Workspace not initialized")
            return {"status": "error", "message": "Workspace not initialized"}
        
        try:
            import json
            # Convert to JSON string
            data_json = json.dumps(input_data)
            
            # Use MCP implementation in workspace
            mcp_code = f"""
import json, os, time, urllib.request, urllib.parse

def invoke_mcp_server(server_id, input_data):
    if server_id == "brave-search" or "search" in server_id:
        return invoke_brave_search(input_data.get("query", ""))
    elif server_id == "chroma-memory" or "memory" in server_id:
        operation = input_data.get("operation", "retrieve")
        if operation == "store":
            data = input_data.get("data", {{}})
            return store_in_memory(data)
        else:
            query = input_data.get("query", "")
            return retrieve_from_memory(query)
    elif server_id == "logfire-logging" or "logging" in server_id:
        event = input_data.get("event", "custom_event")
        data = input_data.get("data", {{}})
        return log_to_logfire(event, data)
    else:
        return {{"status": "error", "message": f"Unknown MCP server: {{server_id}}"}}

def invoke_brave_search(query):
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return {{"status": "error", "message": "No Brave API key provided"}}
        
    params = urllib.parse.urlencode({{"q": query, "count": 3}})
    url = f"https://api.search.brave.com/res/v1/web/search?{{params}}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("X-Subscription-Token", api_key)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = []
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"][:3]:
                    results.append({{
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("description", "")
                    }})
            return {{"status": "success", "results": results}}
    except Exception as e:
        return {{"status": "error", "message": str(e), "results": []}}

def store_in_memory(data):
    try:
        memory_file = '/home/daytona/agent/memory.json'
        memory_data = []
        
        # Load existing memory if present
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                try:
                    memory_data = json.load(f)
                except:
                    memory_data = []
        
        # Prepare entry
        memory_id = data.get("id", f"mem_{{int(time.time())}}")
        entry = {{
            "id": memory_id,
            "timestamp": time.time(),
            "data": data
        }}
        
        # Add to memory
        memory_data.append(entry)
        
        # Save memory
        with open(memory_file, 'w') as f:
            json.dump(memory_data, f)
            
        return {{"status": "success", "message": "Stored in memory", "id": memory_id}}
    except Exception as e:
        return {{"status": "error", "message": str(e)}}

def retrieve_from_memory(query=""):
    try:
        memory_file = '/home/daytona/agent/memory.json'
        
        # Check if memory exists
        if not os.path.exists(memory_file):
            return {{"status": "error", "message": "No memory found", "results": []}}
        
        # Load memory
        with open(memory_file, 'r') as f:
            try:
                memory_data = json.load(f)
            except:
                return {{"status": "error", "message": "Invalid memory format", "results": []}}
        
        # Filter by query if provided
        if query:
            results = []
            query_lower = query.lower()
            for entry in memory_data:
                entry_json = json.dumps(entry).lower()
                if query_lower in entry_json:
                    results.append(entry)
        else:
            # Return all memory entries
            results = memory_data
            
        return {{"status": "success", "results": results}}
    except Exception as e:
        return {{"status": "error", "message": str(e), "results": []}}

def log_to_logfire(event_type, data):
    api_key = os.environ.get("LOGFIRE_API_KEY")
    if not api_key:
        return {{"status": "error", "message": "No LogFire API key provided"}}
    
    try:
        # Prepare log entry
        log_entry = {{
            "timestamp": time.time(),
            "event": event_type,
            "data": data
        }}
        
        # Log to LogFire API or to local file if API not available
        try:
            url = "https://in.logfire.dev/ingest"
            req = urllib.request.Request(url)
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {{api_key}}")
            
            log_data = json.dumps([log_entry]).encode('utf-8')
            
            with urllib.request.urlopen(req, log_data, timeout=10) as response:
                if response.status == 200:
                    return {{"status": "success"}}
                else:
                    raise Exception(f"LogFire API returned status {{response.status}}")
        except Exception as api_error:
            # Fallback to local logging
            log_file = '/home/daytona/agent/agent_logs.jsonl'
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\\n')
            return {{"status": "success", "message": "Logged locally (API error: {{str(api_error)}})"}}
    except Exception as e:
        return {{"status": "error", "message": str(e)}}

# Parse data
try:
    data = json.loads('{data_json}')
except Exception as e:
    print(json.dumps({{"status": "error", "message": f"Invalid JSON: {{str(e)}}"}}))
    exit(1)
    
result = invoke_mcp_server("{server_id}", data)
print(json.dumps(result))
            """
            
            response = self.workspace.process.exec(f"python3 -c '{mcp_code}'", 
                                                  environment=self._api_keys,
                                                  stdout=True, stderr=True)
            
            try:
                response_data = json.loads(response)
                return response_data
            except json.JSONDecodeError:
                self._print(f"❌ Invalid MCP response: {response}")
                return {"status": "error", "message": "Invalid MCP response"}
                
        except Exception as e:
            self._print(f"❌ Error invoking MCP: {e}")
            return {"status": "error", "message": str(e)}