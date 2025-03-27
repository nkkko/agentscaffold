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
    """Daytona runtime for agent execution with persistent workspace and full API integration."""
    
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
    
    def _upload_agent_code(self, agent_dir: str) -> None:
        """Upload agent code to the Daytona workspace if not already uploaded."""
        if self._is_initialized and self._agent_dir == agent_dir:
            print("🔄 Reusing existing code upload")
            return

        self._agent_dir = agent_dir
        self._simple_upload_agent_code(self.workspace, agent_dir)
        self._is_initialized = True
    
    def _simple_upload_agent_code(self, workspace, agent_dir: str) -> None:
        """Upload agent code to the Daytona workspace with improved search capability."""
        from pathlib import Path
        import sys
        import os
        
        # Create the workspace directory structure
        remote_dir = "/home/daytona/agent"
    try:
            # Create directories
        workspace.process.exec(f"mkdir -p {remote_dir}")
            
            # Upload improved brave_search.py module first
        brave_search_code = """
import os
import json
import urllib.request
import urllib.parse
import urllib.error

class BraveSearch:
    \"\"\"Simple Brave Search implementation using only standard library.\"\"\"
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            print("WARNING: No Brave API key found")
        else:
            print(f"Initialized Brave Search with API key: {self.api_key[:4]}...{self.api_key[-4:]}")
    
    def search(self, query, num_results=3):
        \"\"\"Perform a search using Brave Search API with only standard library.\"\"\"
        if not self.api_key:
            print("No Brave API key available")
            return self._mock_results(query, num_results)
        
        try:
            # URL encode parameters
            params = urllib.parse.urlencode({"q": query, "count": num_results})
            url = f"https://api.search.brave.com/res/v1/web/search?{params}"
            
            # Create request with headers
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            req.add_header("X-Subscription-Token", self.api_key)
            
            # Make request
            print(f"Making request to Brave Search API for: {query}")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                results = []
                if "web" in data and "results" in data["web"]:
                    for result in data["web"]["results"][:num_results]:
                        results.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "snippet": result.get("description", "")
                        })
                
                print(f"Got {len(results)} search results for '{query}'")
                return results
        except Exception as e:
            print(f"Error performing search: {e}")
            return self._mock_results(query, num_results)
    
    def _mock_results(self, query, num_results=3):
        \"\"\"Fallback mock results when API is unavailable.\"\"\"
        print("Using mock search results")
        return [
            {"title": f"Result 1 for {query}", "url": "https://example.com/1", "snippet": f"This is a sample result for {query}"},
            {"title": f"Result 2 for {query}", "url": "https://example.com/2", "snippet": f"Another sample result for {query}"},
            {"title": f"Result 3 for {query}", "url": "https://example.com/3", "snippet": f"Yet another sample result for {query}"}
        ][:num_results]
    
    def search_with_snippets(self, query, num_results=3):
        \"\"\"Generate formatted search results.\"\"\"
        results = self.search(query, num_results)
        formatted = "\\n".join([f"{i+1}. {r['title']}\\n   {r['url']}\\n   {r['snippet']}" for i, r in enumerate(results)])
        return formatted
"""
        workspace.fs.upload_file(f"{remote_dir}/brave_search.py", brave_search_code.encode('utf-8'))
        print("Uploaded brave_search.py with standard library implementation")
        
        # Upload .env file with API keys
        print("Creating .env file with API keys...")
        env_content = ""
        for key, value in self._api_keys.items():
            env_content += f"{key}={value}\n"
        
        workspace.fs.upload_file(f"{remote_dir}/.env", env_content.encode('utf-8'))
        print("Successfully uploaded .env file with API keys")
        
        # Upload simplified_agent.py for better integration
        simplified_agent_code = """
from typing import Dict, Any, Optional, List
import json
import asyncio
import os
import time
import sys

# Import our streamlined search implementation
try:
    from brave_search import BraveSearch
    print("Successfully imported BraveSearch")
    search_available = True
except ImportError:
    print("Warning: BraveSearch module not available")
    search_available = False

# Basic in-memory storage for the conversation
conversation_memory = []

class MemoryProvider:
    \"\"\"Memory provider implementation.\"\"\"
    
    def __init__(self):
        print("Initialized memory provider")
        self.memories = []
        
        # Load from conversation memory if available
        global conversation_memory
        for entry in conversation_memory:
            if entry.get("type") == "memory":
                self.add(entry.get("text"), entry.get("metadata"))
        
    def add(self, text, metadata=None):
        \"\"\"Add a memory entry.\"\"\"
        metadata = metadata or {}
        entry_id = f"memory-{len(self.memories)}"
        self.memories.append({"id": entry_id, "text": text, "metadata": metadata})
        
        # Update global memory
        global conversation_memory
        conversation_memory.append({
            "type": "memory",
            "text": text,
            "metadata": metadata
        })
        
        return entry_id
    
    def get_context(self, query, n_results=3, as_string=True):
        \"\"\"Get context based on the query.\"\"\"
        if not self.memories:
            return "" if as_string else []
            
        # Simple keyword matching
        relevant = [m for m in self.memories if query.lower() in m['text'].lower()]
        relevant = sorted(relevant, key=lambda x: len(x['text']), reverse=True)[:n_results]
        
        if as_string:
            return "\\n\\n".join([f"Memory {i+1}:\\n{m['text']}" for i, m in enumerate(relevant)])
        return relevant

class LoggingProvider:
    \"\"\"Simple logging provider implementation.\"\"\"
    
    def __init__(self):
        print("Initialized logging provider")
        self.conversation_id = None
        self.logs = []
    
    def start_conversation(self, user_id=None):
        \"\"\"Start a conversation.\"\"\"
        import uuid
        self.conversation_id = str(uuid.uuid4())
        self.log_system_message(f"Conversation started: {self.conversation_id}")
        return self.conversation_id
    
    def log_system_message(self, message):
        \"\"\"Log a system message.\"\"\"
        entry = {"type": "system", "message": message, "timestamp": time.time()}
        self.logs.append(entry)
        print(f"System: {message}")
    
    def log_user_message(self, message, metadata=None):
        \"\"\"Log a user message.\"\"\"
        entry = {"type": "user", "message": message, "timestamp": time.time(), "metadata": metadata or {}}
        self.logs.append(entry)
        print(f"User: {message}")
    
    def log_agent_message(self, message, metadata=None):
        \"\"\"Log an agent message.\"\"\"
        entry = {"type": "agent", "message": message, "timestamp": time.time(), "metadata": metadata or {}}
        self.logs.append(entry)
        print(f"Agent: {message[:100]}...")

    def log_error(self, error, context=None):
        \"\"\"Log an error.\"\"\"
        entry = {"type": "error", "error": str(error), "context": context or {}, "timestamp": time.time()}
        self.logs.append(entry)
        print(f"Error: {error}")

class Agent:
    \"\"\"Simple agent implementation with search and memory.\"\"\"
    
    def __init__(self):
        print("Initialized agent with search capability")
        
        # Initialize OpenAI client
        self.openai_client = None
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                print(f"Initialized OpenAI client with API key")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
        
        # Initialize Brave Search
        self.search_provider = None
        if search_available:
            try:
                brave_api_key = os.environ.get("BRAVE_API_KEY")
                self.search_provider = BraveSearch(api_key=brave_api_key)
                print("Initialized Brave Search provider")
            except Exception as e:
                print(f"Error initializing search provider: {e}")
        
        # Initialize other providers
        self.memory_provider = MemoryProvider()
        self.logging_provider = LoggingProvider()
        self.logging_provider.start_conversation()
        
    async def _generate_with_openai(self, prompt, system_prompt=None):
        \"\"\"Generate a response using OpenAI.\"\"\"
        if not self.openai_client:
            return f"I cannot access the OpenAI API at the moment. {prompt}"
            
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use latest available model
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}")
            return f"Error generating response with OpenAI: {str(e)}"
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Process a message and return a response.\"\"\"
        try:
            message = input_data.get("message", "")
            context = input_data.get("context", {})
            metadata = {}
            
            # Log user message
            self.logging_provider.log_user_message(message)
            
            # Check for search command
            is_search_request = False
            search_query = None
            
            # Check for explicit search_query parameter
            if input_data.get("search_query") or context.get("search_query"):
                is_search_request = True
                search_query = input_data.get("search_query") or context.get("search_query")
            
            # Look for search keywords in the message
            search_keywords = ["search", "look up", "find information", "google", "lookup"]
            if any(keyword in message.lower() for keyword in search_keywords):
                is_search_request = True
                # Try to extract search query
                words = message.lower().split()
                if "for" in words and words.index("for") < len(words) - 1:
                    idx = words.index("for")
                    search_query = " ".join(message.split()[idx+1:])
                else:
                    # Use the whole message as query
                    search_query = message
            
            # Perform search if requested and provider is available
            if is_search_request and search_query and self.search_provider:
                print(f"Executing search for: {search_query}")
                search_results = self.search_provider.search(search_query)
                metadata["search_results"] = search_results
                context["search_results"] = self.search_provider.search_with_snippets(search_query)
                print(f"Search results: {context['search_results']}")
            
            # Check for memory retrieval
            retrieve_context = input_data.get("retrieve_context", context.get("retrieve_context", False))
            context_query = input_data.get("context_query", context.get("context_query", message))
            if retrieve_context or any(w in message.lower() for w in ["remember", "previous", "before"]):
                print(f"Retrieving memory context for: {context_query}")
                memory_context = self.memory_provider.get_context(context_query)
                context["memory_context"] = memory_context
                metadata["memory_retrieval"] = True
            
            # Build enhanced prompt with context
            system_prompt = "You are an AI assistant with access to search and memory capabilities. Be helpful, concise, and informative."
            
            enhanced_message = message
            if context.get("memory_context"):
                enhanced_message = f"Context from memory:\\n{context['memory_context']}\\n\\nUser message: {message}"
                system_prompt += " Use the provided memory context to inform your response."
            
            if context.get("search_results"):
                enhanced_message = f"Search results:\\n{context['search_results']}\\n\\nUser message: {message}"
                system_prompt += " Use the provided search results to inform your response when appropriate."
            
            # Generate response
            response = None
            
            # If search was performed, craft a response that includes the search results
            if is_search_request and search_query and self.search_provider and context.get("search_results"):
                response = f"Here's what I found for '{search_query}':\\n\\n{context['search_results']}\\n\\nWould you like me to explain any of these results in more detail?"
            
            # Otherwise try to generate with OpenAI
            if not response and self.openai_client:
                print("Generating response with OpenAI...")
                response = await self._generate_with_openai(enhanced_message, system_prompt)
            
            # If we still don't have a response, use a fallback
            if not response:
                if is_search_request and search_query and self.search_provider:
                    if context.get("search_results"):
                        response = f"I searched for '{search_query}' and found:\\n\\n{context['search_results']}"
                    else:
                        response = f"I tried to search for '{search_query}', but couldn't get any results."
                elif "help" in message.lower():
                    response = "I'm an AI assistant running in the Daytona environment. I can help with various tasks including searching for information and remembering context from our conversation."
                else:
                    response = f"I received your message: '{message}'. I'm running in an enhanced Daytona environment with search and memory capabilities."
            
            # Store in memory
            entry_text = f"User: {message}\\nAgent: {response}"
            memory_metadata = {
                "timestamp": time.time(),
                "source": "conversation"
            }
            entry_id = self.memory_provider.add(entry_text, memory_metadata)
            metadata["memory_entry_id"] = entry_id
            
            # Log agent response
            self.logging_provider.log_agent_message(response, metadata)
            
            # Prepare output
            output = {
                "response": response,
                "metadata": metadata
            }
            
            # Add search_results to output if available
            if "search_results" in metadata:
                output["search_results"] = metadata["search_results"]
            
            # Add memory_context to output if available
            if "memory_context" in context:
                output["memory_context"] = context["memory_context"]
                
            return output
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            self.logging_provider.log_error(e, context=input_data)
            return {"response": error_msg, "metadata": {"error": True}}
"""
        workspace.fs.upload_file(f"{remote_dir}/simplified_agent.py", simplified_agent_code.encode('utf-8'))
        print("Uploaded simplified_agent.py with streamlined search integration")
        
        # Upload other essential files
        essential_patterns = ["requirements.in", "requirements.txt", "pyproject.toml", "README.md"]
        for pattern in essential_patterns:
            for path in Path(agent_dir).glob(pattern):
                if path.is_file():
                    filename = os.path.basename(path)
                    with open(path, "rb") as f:
                        content = f.read()
                    workspace.fs.upload_file(f"{remote_dir}/{filename}", content)
                    print(f"Uploaded {filename}")
        
        # Create __init__.py
        workspace.fs.upload_file(f"{remote_dir}/__init__.py", b"# Package initialization")
        
        # Upload agent.py
        agent_path = Path(agent_dir) / "agent.py"
        if agent_path.exists():
            with open(agent_path, "rb") as f:
                content = f.read()
            workspace.fs.upload_file(f"{remote_dir}/agent.py", content)
            
        # Upload main.py
        main_path = Path(agent_dir) / "main.py"
        if main_path.exists():
            with open(main_path, "rb") as f:
                content = f.read()
            workspace.fs.upload_file(f"{remote_dir}/main.py", content)
        
        # Check for package directory
        package_name = None
        for item in os.listdir(agent_dir):
            pkg_path = os.path.join(agent_dir, item)
            if os.path.isdir(pkg_path) and os.path.exists(os.path.join(pkg_path, "__init__.py")):
                package_name = item
                pkg_dir = f"{remote_dir}/{package_name}"
                workspace.process.exec(f"mkdir -p {pkg_dir}")
                
                # Upload package files
                for root, dirs, files in os.walk(pkg_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, pkg_path)
                            remote_file_path = f"{pkg_dir}/{rel_path}"
                            remote_file_dir = os.path.dirname(remote_file_path)
                            workspace.process.exec(f"mkdir -p {remote_file_dir}")
                            
                            with open(file_path, "rb") as f:
                                content = f.read()
                            workspace.fs.upload_file(remote_file_path, content)
                break
        
        # List directory contents to verify uploads
        ls_result = workspace.process.exec(f"ls -la {remote_dir}")
        print(f"Directory contents: {ls_result.result}")
    
    except Exception as e:
        print(f"Error uploading agent code: {e}")



    def _install_dependencies(self, workspace) -> None:
        """Install required dependencies in the workspace."""
        if self._is_initialized:
            return
            
        try:
            print("📦 Installing required dependencies...")
            result = workspace.process.exec("cd /home/daytona/agent && python dependencies.py")
            if result.exit_code != 0:
                print(f"⚠️ Warning: Failed to install dependencies: {result.result}")
            else:
                print("✅ Successfully installed dependencies")
        except Exception as e:
            print(f"⚠️ Warning: Error installing dependencies: {e}")
    
    def _prepare_execution_code(self, input_base64, conversation_id):
        """Generate optimized execution code with persistent memory."""
        return f"""
import sys, os, json, base64, traceback, asyncio
from typing import Dict, Any, Optional, List

# Setup paths
sys.path.append('/home/daytona/agent')
sys.path.append('/home/daytona')
os.chdir('/home/daytona/agent')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('/home/daytona/agent/.env')
    print("✅ Loaded environment variables in Daytona")
except ImportError:
    # Manual env loading
    if os.path.exists('/home/daytona/agent/.env'):
        with open('/home/daytona/agent/.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Manually loaded environment variables in Daytona")

# Check for API keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    print(f"Found OpenAI API key in Daytona (length: {{len(openai_api_key)}})")

anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
if anthropic_api_key:
    print(f"Found Anthropic API key in Daytona (length: {{len(anthropic_api_key)}})")

brave_api_key = os.environ.get("BRAVE_API_KEY")
if brave_api_key:
    print(f"Found Brave API key in Daytona (length: {{len(brave_api_key)}})")

# Conversation ID for persistent memory
CONVERSATION_ID = "{conversation_id}"
print(f"🔄 Processing message in conversation: {{CONVERSATION_ID}}")

# Parse the input data
input_base64 = "{input_base64}"
input_json = base64.b64decode(input_base64).decode('utf-8')
input_data = json.loads(input_json)
message = input_data.get('message', '')
print(f"📩 Received message: '{{message}}'")

# Try different import approaches
try:
    # First try full API integration agent
    try:
        # Install dependencies first
        try:
            print("Installing required packages...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "--break-system-packages"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.0.0", "--break-system-packages"])
            print("✅ Installed required packages")
        except Exception as install_error:
            print(f"⚠️ Warning: Could not install packages: {{install_error}}")
        
        from full_agent import Agent
        print("✅ Successfully imported Agent with full API integration")
        agent = Agent()
        print("✅ Created Agent instance with API integration")
        result = asyncio.run(agent.process(input_data))
        print("✅ Processed message with full Agent")
        print(json.dumps(result))
        sys.exit(0)  # Exit successfully after processing
    except Exception as e:
        print(f"⚠️ Full API agent failed: {{e}}")
        traceback.print_exc()
    
    # Try direct import of Agent
    try:
        from agent import Agent
        print("✅ Successfully imported Agent from agent.py")
        agent = Agent()
        print("✅ Created Agent instance")
        result = asyncio.run(agent.process(input_data))
        print("✅ Processed message with agent")
        print(json.dumps(result))
        sys.exit(0)  # Exit successfully after processing
    except Exception as e:
        print(f"⚠️ Direct import failed: {{e}}")
        traceback.print_exc()
    
    # Try importing from main.py
    try:
        print("🔍 Trying to import from main.py...")
        from main import main
        print("✅ Successfully imported main function")
        result = asyncio.run(main(input_data))
        print("✅ Processed message with main function")
        print(json.dumps(result))
        sys.exit(0)  # Exit successfully after processing
    except Exception as e:
        print(f"⚠️ Main import failed: {{e}}")
        traceback.print_exc()
        
    # Try finding a package
    try:
        print("🔍 Looking for packages...")
        found_package = False
        for item in os.listdir('.'):
            if os.path.isdir(item) and os.path.exists(os.path.join(item, '__init__.py')):
                print(f"📦 Found package: {{item}}")
                try:
                    agent_module = __import__(f"{{item}}.agent", fromlist=['Agent'])
                    agent = agent_module.Agent()
                    print(f"✅ Successfully imported Agent from {{item}}.agent")
                    result = asyncio.run(agent.process(input_data))
                    print("✅ Processed message with package agent")
                    print(json.dumps(result))
                    found_package = True
                    sys.exit(0)  # Exit successfully after processing
                except Exception as pkg_error:
                    print(f"⚠️ Error with package {{item}}: {{pkg_error}}")
                    continue
                    
        if not found_package:
            print("⚠️ No working package found")
            raise ImportError("No working package found")
    except Exception as e:
        print(f"⚠️ Package search failed: {{e}}")
        traceback.print_exc()
        
    # Fall back to bare-bones processing if nothing else worked
    print("🔄 Using fallback response mechanism")
    response = {{
        "response": f"I received your message: '{{message}}'. I'm running in a persistent Daytona environment (Conversation ID: {{CONVERSATION_ID}}). Your message is recorded and I'll maintain context throughout our conversation.",
        "metadata": {{"fallback": True, "daytona": True, "conversation_id": CONVERSATION_ID}}
    }}
    print(json.dumps(response))
    
except Exception as e:
    # Ultimate fallback
    print(f"❌ Critical error: {{e}}")
    traceback.print_exc()
    print(json.dumps({{
        "response": "I encountered a system error. Please try again later.",
        "metadata": {{"error": True, "critical": True}}
    }}))
"""
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
    async def execute(self, agent_dir: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent in Daytona workspace, maintaining state between executions."""
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
                
            # Upload agent code if not already uploaded
            print("📤 Uploading code to Daytona...")
            self._upload_agent_code(agent_dir)
            
            # Install required packages if not already installed
            if not self._is_initialized:
                required_packages = self._get_required_packages(agent_dir)
                for package in required_packages:
                    self._ensure_dependency(package, self.workspace)
            
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
        








       