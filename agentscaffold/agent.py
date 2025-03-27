"""Base agent implementation for AgentScaffold."""

import asyncio
import inspect
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional, Callable, Type, ClassVar, Union
import json
import os
import importlib.util
import sys
import re
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
        
        # Load environment variables first
        self._load_env_vars()
        
        # Verify Daytona configuration with better error handling
        if not os.environ.get("DAYTONA_API_KEY"):
            print("⚠️ DAYTONA_API_KEY environment variable is required")
        else:
            print(f"✅ Loaded .env (DAYTONA_API_KEY=***{os.environ.get('DAYTONA_API_KEY')[-4:] if len(os.environ.get('DAYTONA_API_KEY', '')) > 4 else ''})")
        
        if not os.environ.get("DAYTONA_SERVER_URL"):
            print("⚠️ DAYTONA_SERVER_URL environment variable is required")
        
        # Collect API keys for services
        self._collect_api_keys()
        
        # Initialize Daytona SDK
        self._load_daytona_sdk()
    
    def _load_env_vars(self):
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            # Try loading from multiple locations
            for env_path in ['.env', '../.env']:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    break
        except ImportError:
            # Try manual loading if dotenv is not available
            for env_path in ['.env', '../.env']:
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#') and '=' in line:
                                key, value = line.strip().split('=', 1)
                                os.environ[key.strip()] = value.strip()
    
    def _collect_api_keys(self):
        """Collect API keys for services."""
        # OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            self._api_keys["OPENAI_API_KEY"] = openai_api_key
            print(f"✅ Found OpenAI API key (length: {len(openai_api_key)})")
        
        # Anthropic API key
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self._api_keys["ANTHROPIC_API_KEY"] = anthropic_api_key
            print(f"✅ Found Anthropic API key (length: {len(anthropic_api_key)})")
        
        # Brave Search API key
        brave_api_key = os.environ.get("BRAVE_API_KEY")
        if brave_api_key:
            self._api_keys["BRAVE_API_KEY"] = brave_api_key
            print(f"✅ Found Brave Search API key (length: {len(brave_api_key)})")
        
        # LogFire API key
        logfire_api_key = os.environ.get("LOGFIRE_API_KEY")
        if logfire_api_key:
            self._api_keys["LOGFIRE_API_KEY"] = logfire_api_key
            print(f"✅ Found LogFire API key (length: {len(logfire_api_key)})")
    
    def _load_daytona_sdk(self):
        """Load and verify Daytona SDK with better error handling."""
        try:
            from daytona_sdk import Daytona, CreateWorkspaceParams, DaytonaConfig
            
            # Configure from environment with fallbacks
            api_key = os.environ.get("DAYTONA_API_KEY", self.config.get("api_key"))
            server_url = os.environ.get("DAYTONA_SERVER_URL", self.config.get("server_url"))
            target = os.environ.get("DAYTONA_TARGET", self.config.get("target", "us"))
            
            if not api_key:
                print("❌ No Daytona API key found")
                return
                
            daytona_config = DaytonaConfig(
                api_key=api_key,
                server_url=server_url,
                target=target
            )
            
            self._daytona = Daytona(daytona_config)
            self._CreateWorkspaceParams = CreateWorkspaceParams
            
        except ImportError as e:
            print(f"❌ daytona-sdk package not installed: {e}")
        except Exception as e:
            print(f"❌ Daytona initialization failed: {e}")
    
    def _init_workspace(self, agent_dir: str) -> bool:
        """Initialize workspace if not already initialized."""
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
            
            # Generate a conversation ID if not already set
            if not self._conversation_id:
                import uuid
                self._conversation_id = str(uuid.uuid4())
                print(f"🆕 New conversation ID: {self._conversation_id}")
            
            return True
        except Exception as e:
            print(f"❌ Error creating workspace: {e}")
            return False
    
    def _upload_agent_code(self, agent_dir: str) -> None:
        """Upload agent code to the Daytona workspace if not already uploaded."""
        if self._is_initialized and self._agent_dir == agent_dir:
            print("🔄 Reusing existing code upload")
            return

        self._agent_dir = agent_dir
        
        try:
            # Create the agent directory first
            self.workspace.process.exec("mkdir -p /home/daytona/agent")
            
            # Upload the code
            from pathlib import Path
            
            # Upload .env file with API keys
            print("Creating .env file with API keys...")
            env_content = ""
            for key, value in self._api_keys.items():
                env_content += f"{key}={value}\n"
            
            self.workspace.fs.upload_file("/home/daytona/agent/.env", env_content.encode('utf-8'))
            print("Successfully uploaded .env file with API keys")
            
            # We don't need to upload extra files, the execute method will
            # run our self-contained script
            
            self._is_initialized = True
            
        except Exception as e:
            print(f"Error uploading agent code: {e}")
   
    
    def _cleanup_workspace(self):
        """Clean up the workspace when done."""
        if not self.workspace:
            return
            
        try:
            print("🧹 Cleaning up workspace...")
            self._daytona.remove(self.workspace)
            self.workspace = None
            self._is_initialized = False
        except Exception as e:
            print(f"❌ Error cleaning up workspace: {e}")
    def _prepare_execution_code(self, input_base64, conversation_id):
        """Generate optimized execution code with memory and logging capabilities."""
        return f"""
import sys, os, json, base64, traceback, asyncio
from typing import Dict, Any, Optional, List
import time
import uuid

# Check if directory exists before changing to it
if not os.path.exists('/home/daytona/agent'):
    print("Creating agent directory...")
    os.makedirs('/home/daytona/agent', exist_ok=True)

# Setup paths
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
    # Manual env loading
    if os.path.exists('/home/daytona/agent/.env'):
        with open('/home/daytona/agent/.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Manually loaded environment variables in Daytona")
    else:
        print("No .env file found in agent directory")

# List directory contents for debugging
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

# Conversation ID for persistent memory
CONVERSATION_ID = "{conversation_id}"
print(f"🔄 Processing message in conversation: {{CONVERSATION_ID}}")

# Parse the input data
input_base64 = "{input_base64}"
input_json = base64.b64decode(input_base64).decode('utf-8')
input_data = json.loads(input_json)
message = input_data.get('message', '')
print(f"📩 Received message: '{{message}}'")


# --- MEMORY IMPLEMENTATION ---
# Simple in-memory storage for this session
conversation_history = []

# Function to read conversation history from a file
def read_conversation_history():
    history_file = '/home/daytona/agent/conversation_history.json'
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading conversation history: {{e}}")
    return []

# Function to save conversation history to a file
def save_conversation_history(history):
    history_file = '/home/daytona/agent/conversation_history.json'
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f)
        print(f"Saved {{len(history)}} conversation entries to file")
    except Exception as e:
        print(f"Error saving conversation history: {{e}}")

# Load conversation history
conversation_history = read_conversation_history()
print(f"Loaded {{len(conversation_history)}} conversation entries from history")

# Function to add a memory entry
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

# Function to retrieve context from memory
def get_context(query, n_results=3):
    if not conversation_history:
        return ""
    
    # Simple keyword matching (in a real implementation, use embeddings)
    relevant = []
    for entry in conversation_history:
        if query.lower() in entry['text'].lower():
            relevant.append(entry)
    
    # Sort by recency if there are no keyword matches
    if not relevant:
        # Get most recent entries
        relevant = sorted(conversation_history, key=lambda x: x.get('timestamp', 0), reverse=True)[:n_results]
    else:
        # Limit to n_results
        relevant = relevant[:n_results]
    
    # Format as string
    if relevant:
        return "\\n\\n".join([f"Memory {{i+1}}:\\n{{m['text']}}" for i, m in enumerate(relevant)])
    return ""

# --- LOGGING IMPLEMENTATION ---
def log_event(event_type, data):
    "\"\"\"Log an event to a file and try LogFire if available.\"\"\"
    log_entry = {{
        "event_type": event_type,
        "timestamp": time.time(),
        "conversation_id": CONVERSATION_ID,
        "data": data
    }}
    
    # Log to file
    log_file = '/home/daytona/agent/agent_log.jsonl'
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\\n')
    except Exception as e:
        print(f"Error writing to log file: {{e}}")
    
    # Try LogFire if available
    if logfire_api_key:
        try:
            import urllib.request
            import urllib.parse
            import urllib.error
            
            url = "https://api.logfire.dev/api/v1/log"
            headers = {{
                "Content-Type": "application/json",
                "Authorization": f"Bearer {{logfire_api_key}}"
            }}
            
            log_data = {{
                "event": event_type,
                "conversation_id": CONVERSATION_ID,
                **data
            }}
            
            req = urllib.request.Request(
                url, 
                data=json.dumps(log_data).encode('utf-8'),
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                response_data = response.read().decode('utf-8')
                print(f"LogFire log successful: {{response_data}}")
                
        except Exception as e:
            print(f"Error logging to LogFire: {{e}}")

# Log user message
log_event("user_message", {{"message": message}})

# --- SEARCH IMPLEMENTATION ---
def perform_search(query, api_key=None):
    \"\"\"Simple search implementation using only standard library.\"\"\"
    print(f"Performing search for: {{query}}")
    
    api_key = api_key or os.environ.get("BRAVE_API_KEY")
    if not api_key:
        print("No Brave API key available")
        return [
            {{"title": f"Result 1 for {{query}}", "url": "https://example.com/1", "snippet": f"This is a sample result for {{query}}"}},
            {{"title": f"Result 2 for {{query}}", "url": "https://example.com/2", "snippet": f"Another sample result for {{query}}"}},
        ]
    
    try:
        import urllib.request
        import urllib.parse
        import urllib.error
        
        # URL encode parameters
        params = urllib.parse.urlencode({{"q": query, "count": 3}})
        url = f"https://api.search.brave.com/res/v1/web/search?{{params}}"
        
        # Create request with headers
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("X-Subscription-Token", api_key)
        
        # Make request
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
            
            print(f"Got {{len(results)}} search results for '{{query}}'")
            return results
    except Exception as e:
        print(f"Error performing search: {{e}}")
        return [
            {{"title": f"Result 1 for {{query}}", "url": "https://example.com/1", "snippet": f"This is a sample result for {{query}}"}},
            {{"title": f"Result 2 for {{query}}", "url": "https://example.com/2", "snippet": f"Another sample result for {{query}}"}},
        ]

# --- MAIN EXECUTION LOGIC ---
# Check if this is a search request
search_keywords = ["search", "look up", "find", "google", "information about", "what is", "tell me about"]
is_search_request = any(keyword in message.lower() for keyword in search_keywords)

# Check if this is a memory request
memory_keywords = ["remember", "memory", "previous", "recall", "earlier", "before", "last time"]
is_memory_request = any(keyword in message.lower() for keyword in memory_keywords)

# Try to handle the request
try:
    # If it's a search request, perform search directly
    if is_search_request:
        search_query = message
        
        # Try to extract search query from message
        for prefix in ["search for", "search", "look up", "find", "tell me about"]:
            if prefix in message.lower():
                search_query = message.lower().split(prefix, 1)[1].strip()
                break
        
        print(f"Detected search request: {{search_query}}")
        results = perform_search(search_query, brave_api_key)
        
        # Format the results
        formatted_results = "\\n".join([f"{{i+1}}. {{r['title']}}\\n   {{r['url']}}\\n   {{r['snippet']}}" for i, r in enumerate(results)])
        
        response_text = f"Here's what I found for '{{search_query}}':\\n\\n{{formatted_results}}"
        
        # Store in memory
        add_to_memory(f"User: {{message}}\\nAgent: {{response_text}}", {{"type": "search", "query": search_query}})
        
        # Log the search and response
        log_event("search", {{"query": search_query, "results": results}})
        log_event("agent_response", {{"response": response_text, "type": "search"}})
        
        response = {{
            "response": response_text,
            "metadata": {{"search_results": results}}
        }}
        
        print("✅ Direct search successful")
        print(json.dumps(response))
        sys.exit(0)  # Exit successfully after processing
    
    # If it's a memory request, retrieve and provide context
    elif is_memory_request:
        context = get_context(message)
        
        if context:
            response_text = f"Here's what I remember from our previous conversation:\\n\\n{{context}}"
        else:
            response_text = "I don't have any relevant memories from our previous conversations yet."
        
        # Store this interaction
        add_to_memory(f"User: {{message}}\\nAgent: {{response_text}}", {{"type": "memory_retrieval"}})
        
        # Log the memory retrieval
        log_event("memory_retrieval", {{"query": message, "found": bool(context)}})
        log_event("agent_response", {{"response": response_text, "type": "memory"}})
        
        response = {{
            "response": response_text,
            "metadata": {{"memory_retrieval": True, "conversation_id": CONVERSATION_ID}}
        }}
        
        print("✅ Memory retrieval successful")
        print(json.dumps(response))
        sys.exit(0)
    
    # For regular messages, provide a response with conversation context
    else:
        # Get recent context
        context = get_context("")  # Empty query gets most recent items
        
        if context:
            response_text = f"I received your message: '{{message}}'. I'm running in a persistent Daytona environment with search and memory capabilities. You can ask me to search for information by saying 'search for [topic]' or ask about our previous conversation.\\n\\nBased on our conversation history:\\n{{context}}"
        else:
            response_text = f"I received your message: '{{message}}'. I'm running in a persistent Daytona environment with search and memory capabilities. You can ask me to search for information by saying 'search for [topic]' or ask me to remember our conversation later."
        
        # Store this interaction
        add_to_memory(f"User: {{message}}\\nAgent: {{response_text}}", {{"type": "conversation"}})
        
        # Log the regular message
        log_event("regular_message", {{"message": message}})
        log_event("agent_response", {{"response": response_text, "type": "regular"}})
        
        response = {{
            "response": response_text,
            "metadata": {{"conversation_id": CONVERSATION_ID}}
        }}
        
        print(json.dumps(response))
    
except Exception as e:
    # Ultimate fallback
    print(f"❌ Critical error: {{e}}")
    traceback.print_exc()
    
    # Log the error
    log_event("error", {{"error": str(e), "traceback": traceback.format_exc()}})
    
    print(json.dumps({{
        "response": "I encountered a system error. Please try again later.",
        "metadata": {{"error": True, "critical": True}}
    }}))
"""
    async def execute(self, agent_dir: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent in Daytona workspace with proper directory handling."""
        # Exit command handling
        if isinstance(input_data, dict) and input_data.get('message', '').lower().strip() in ['exit', 'quit', 'bye']:
            print("👋 Received exit command")
            self._cleanup_workspace()
            return {
                "response": "Session ended. All resources have been cleaned up.",
                "metadata": {"exited": True}
            }
            
        if not self._daytona:
            return {
                "response": "Error: Daytona SDK not initialized. Check your API keys and server URL.",
                "metadata": {"error": True}
            }
            
        print("🚀 Starting Daytona agent...")
        
        try:
            # Initialize or reuse workspace
            if not self._init_workspace(agent_dir):
                return {
                    "response": "Error: Failed to initialize Daytona workspace.",
                    "metadata": {"error": True}
                }
            
            # Create the agent directory in the workspace
            try:
                self.workspace.process.exec("mkdir -p /home/daytona/agent")
                print("Created agent directory in workspace")
            except Exception as e:
                print(f"Error creating agent directory: {e}")
            
            # Upload agent code if not already uploaded
            print("📤 Uploading code to Daytona...")
            self._upload_agent_code(agent_dir)
            
            # Prepare input data
            import base64
            input_json = json.dumps(input_data)
            input_base64 = base64.b64encode(input_json.encode('utf-8')).decode('utf-8')
            
            # Prepare and run the execution code
            print("⚡ Running in Daytona...")
            exec_code = self._prepare_execution_code(input_base64, self._conversation_id)
            response = self.workspace.process.code_run(exec_code)
            
            # Parse the response
            if response.exit_code != 0:
                print(f"❌ Execution failed (exit code {response.exit_code})")
                return {
                    "response": f"Error executing agent in Daytona: {response.result}",
                    "metadata": {"error": True}
                }
            
            # Find JSON in the output
            lines = response.result.strip().split('\n')
            for line in reversed(lines):
                try:
                    result = json.loads(line)
                    print("✅ Execution completed successfully")
                    return result
                except json.JSONDecodeError:
                    continue
            
            # Fallback if no JSON found
            return {
                "response": "Execution completed but no valid response was returned.",
                "metadata": {"raw_output": response.result}
            }
            
        except Exception as e:
            print(f"❌ Daytona execution error: {e}")
            return {
                "response": f"Error: {str(e)}",
                "metadata": {"error": True}
            }
        finally:
            # We don't cleanup the workspace here to maintain state between calls
            pass
    
    def cleanup_resources(self):
        """Clean up all resources when conversation ends."""
        self._cleanup_workspace()

class BaseAgent(BaseModel):
    """Minimal base agent implementation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = "MinimalAgent"
    description: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        self._init_daytona_runtime()
        # Always force Daytona by default
        self.force_daytona = os.environ.get("FORCE_DAYTONA", "true").lower() in ["true", "1", "yes"]
    
    def _init_daytona_runtime(self):
        """Initialize Daytona runtime."""
        self._runtime = DaytonaRuntime()
            
    @property
    def runtime(self):
        """Get the DaytonaRuntime instance."""
        return self._runtime
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and generate output."""
        return {"response": f"Processed: {input_data['message']}", "metadata": {}}
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent with the provided input data."""
        if isinstance(input_data, str):
            input_data = {"message": input_data}
            
        # Check for exit command
        if isinstance(input_data, dict) and input_data.get('message', '').lower().strip() in ['exit', 'quit', 'bye']:
            print("Received exit command, cleaning up resources...")
            return {
                "response": "Session ended. All resources have been cleaned up.",
                "metadata": {"exited": True}
            }
            
        # Check if we should force Daytona execution (from env var or input data)
        context_force_daytona = input_data.get("context", {}).get("force_daytona")
        force_daytona = context_force_daytona if context_force_daytona is not None else self.force_daytona
        
        # Process the input
        try:
            if force_daytona:
                print("🚀 Starting Daytona execution...")
                    
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
                    print(f"Using agent directory: {agent_dir}")
                    
                    # Execute the agent via Daytona runtime
                    result = await self._runtime.execute(agent_dir, input_data)
                    return result
                except Exception as e:
                    print(f"❌ Daytona execution error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Fall back to local execution
                    print("Falling back to local execution due to Daytona error")
                    return await self.process(input_data)
            else:
                return await self.process(input_data)
        except Exception as e:
            print(f"Error running agent: {e}")
            return {"response": f"Error: {e}", "metadata": {"error": True}}
class Agent(BaseModel):
    """Base agent class for AgentScaffold."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "BaseAgent"  # Provide a default value for name
    description: str = ""
    input_class: ClassVar[Type[AgentInput]] = AgentInput
    output_class: ClassVar[Type[AgentOutput]] = AgentOutput
    pydantic_agent: Optional[Any] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        print("🔒 Initializing with mandatory Daytona execution")
        self._runtime = DaytonaRuntime()


        # Handle missing name field by providing a default
        if "name" not in data:
            data["name"] = "BaseAgent"
            # Ensure environment variables are loaded before anything else
    try:
        from dotenv import load_dotenv
        # Load from multiple possible locations - both current dir and parent dir
        load_dotenv()
        load_dotenv('../.env')  # Try parent directory too
        
        # Check if crucial API keys are loaded
        if os.environ.get('OPENAI_API_KEY'):
            print(f"Found OPENAI_API_KEY in environment (length: {len(os.environ.get('OPENAI_API_KEY'))})")
        else:
            print("WARNING: OPENAI_API_KEY not found in environment variables")
            
            # Try to load from .env file directly as a last resort
            env_files = ['.env', '../.env']
            for env_file in env_files:
                if os.path.exists(env_file):
                    print(f"Found .env file at {env_file}, trying direct loading")
                    with open(env_file, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                if '=' in line:
                                    key, value = line.strip().split('=', 1)
                                    if key.strip() == 'OPENAI_API_KEY' and value.strip():
                                        os.environ['OPENAI_API_KEY'] = value.strip()
                                        print(f"Loaded OPENAI_API_KEY directly from {env_file} (length: {len(value.strip())})")
                                        break
        
        if os.environ.get('ANTHROPIC_API_KEY'):
            print(f"Found ANTHROPIC_API_KEY in environment (length: {len(os.environ.get('ANTHROPIC_API_KEY'))})")
    except ImportError:
        print("Warning: python-dotenv not installed, trying direct env file parsing")
        # Try to load API keys manually if dotenv isn't available
        try:
            env_files = ['.env', '../.env']
            for env_file in env_files:
                if os.path.exists(env_file):
                    print(f"Found .env file at {env_file}, trying direct loading")
                    with open(env_file, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                if '=' in line:
                                    key, value = line.strip().split('=', 1)
                                    if key.strip() in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY'] and value.strip():
                                        os.environ[key.strip()] = value.strip()
                                        print(f"Loaded {key.strip()} directly from {env_file} (length: {len(value.strip())})")
        except Exception as e:
            print(f"Error loading environment variables directly: {e}")
            
        super().__init__(**data)
        self._runtime = DaytonaRuntime()
        self.force_daytona = os.environ.get('FORCE_DAYTONA', 'false').lower() == 'true'
        print(f"Agent initialized: {self.name}")

        # Initialize PydanticAgent with robust error handling
        self.pydantic_agent = None
        try:
            # First check if we have an API key before trying to initialize
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                print("WARNING: No OPENAI_API_KEY in environment, skipping PydanticAgent initialization")
            else:
                from pydantic_ai import PydanticAgent
                self.pydantic_agent = PydanticAgent(
                    "openai:gpt-4",
                    result_type=self.output_class,
                    system_prompt=(
                        f"You are {self.name}, an AI assistant designed to help with "
                        f"{self.description}. Be helpful, concise, and accurate in your responses."
                    )
                )
                print("Successfully initialized PydanticAgent")
        except Exception as e:
            print(f"Warning: Error initializing PydanticAgent: {e}")
            try:
                # Try alternative approach if original fails
                from pydantic_ai import PydanticAgent
                api_key = os.environ.get('OPENAI_API_KEY')
                if api_key:
                    print("Trying alternative PydanticAgent initialization with explicit API key")
                    # Import provider directly to set API key manually
                    from openai import OpenAI, AsyncOpenAI
                    client = AsyncOpenAI(api_key=api_key)
                    
                    self.pydantic_agent = PydanticAgent(
                        "gpt-3.5-turbo",  # Fallback to a simpler model
                        result_type=self.output_class
                    )
                    print("Successfully initialized fallback PydanticAgent")
                else:
                    print("No OpenAI API key available, agent will work in limited capacity")
            except Exception as fallback_error:
                print(f"Warning: Error initializing fallback PydanticAgent: {fallback_error}")
                self.pydantic_agent = None



        # Initialize providers
        # Search provider
        self.search_provider = None
        if hasattr(self, '_init_search'):
            self.search_provider = self._init_search()
        
        # Memory provider
        self.memory_provider = None
        if hasattr(self, '_init_memory'):
            self.memory_provider = self._init_memory()
        
        # Logging provider
        self.logging_provider = None
        if hasattr(self, '_init_logging'):
            self.logging_provider = self._init_logging()

    @property
    def runtime(self):
        """Get the DaytonaRuntime instance."""
        return self._runtime
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
        """Run agent in Daytona."""
        print("🚀 Starting Daytona execution...")
        
        try:
            agent_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
            result = await self._runtime.execute(agent_dir, input_data)
            print("✅ Daytona execution completed")
            return result
        except Exception as e:
            print(f"❌ Daytona execution failed: {e}")
            return {
                "response": f"Daytona execution error: {str(e)}",
                "metadata": {"error": True}
            }
        








       