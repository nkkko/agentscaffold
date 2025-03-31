"""MCP (Model Context Protocol) server integration for AgentScaffold."""

import os
import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Constants for capability types
CAPABILITY_SEARCH = "search"
CAPABILITY_MEMORY = "memory"
CAPABILITY_LOGGING = "logging"
CAPABILITY_CODE = "code"
CAPABILITY_DATABASE = "database"
CAPABILITY_LOCATION = "location"

# MCP configuration is per-agent, stored in the agent's directory
MCP_CONFIG_FILE = ".mcp.json"

# Define built-in MCP types
BUILTIN_MCPS = {
    "brave-search": {
        "id": "brave-search",
        "name": "Brave Search",
        "capability": CAPABILITY_SEARCH,
        "description": "Web search using Brave API",
        "type": "http",
        "url": "https://api.search.brave.com",
        "api_key_env": "BRAVE_API_KEY"
    },
    "chroma-memory": {
        "id": "chroma-memory",
        "name": "Chroma Memory",
        "capability": CAPABILITY_MEMORY,
        "description": "Vector database for memories",
        "type": "http",
        "url": "http://localhost:8000",
        "config": {
            "client_type": "persistent",
            "data_dir": "~/.agentscaffold/chroma"
        }
    },
    "logfire-logging": {
        "id": "logfire-logging",
        "name": "LogFire Logging",
        "capability": CAPABILITY_LOGGING,
        "description": "Observability and logging",
        "type": "http",
        "url": "https://api.logfire.dev",
        "api_key_env": "LOGFIRE_READ_TOKEN"
    },
    "sqlite": {
        "id": "sqlite",
        "name": "SQLite",
        "capability": CAPABILITY_DATABASE,
        "description": "SQLite database access",
        "type": "stdio",
        "command": "uvx",
        "args": ["mcp-server-sqlite"],
        "config": {
            "database_path": "~/.agentscaffold/sqlite/data.db"
        }
    }
}

logger = logging.getLogger(__name__)

def get_config_path(location: Optional[str] = None) -> str:
    """
    Get the path to the MCP configuration file.
    
    Args:
        location: Directory to check (default: current directory)
        
    Returns:
        Absolute path to the configuration file
    """
    if location is None:
        location = os.getcwd()
        
    return os.path.join(os.path.abspath(location), MCP_CONFIG_FILE)

def load_mcp_servers(location: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load MCP servers configuration from the specified location.
    
    Args:
        location: Directory containing the configuration (default: current directory)
        
    Returns:
        List of MCP server configurations
    """
    config_path = get_config_path(location)
    
    if not os.path.exists(config_path):
        return []
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
            # Convert dictionary to list format for consistency
            servers = []
            for server_id, server_config in config.get("mcpServers", {}).items():
                server = server_config.copy()
                server["id"] = server_id
                servers.append(server)
                
            return servers
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading MCP configuration: {e}")
        return []

def save_mcp_servers(servers: List[Dict[str, Any]], location: Optional[str] = None) -> bool:
    """
    Save MCP servers configuration to the specified location.
    
    Args:
        servers: List of server configurations
        location: Directory to save to (default: current directory)
        
    Returns:
        Success or failure
    """
    config_path = get_config_path(location)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Convert list to dictionary format for storage
    mcp_servers = {}
    for server in servers:
        server_id = server.get("id")
        if not server_id:
            continue
            
        server_copy = server.copy()
        if "id" in server_copy:
            del server_copy["id"]
            
        mcp_servers[server_id] = server_copy
    
    # Create the configuration object
    config = {"mcpServers": mcp_servers}
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except (IOError, OSError) as e:
        logger.error(f"Error saving MCP configuration: {e}")
        return False

def get_available_mcps() -> List[Dict[str, Any]]:
    """
    Get list of available built-in MCP servers.
    
    Returns:
        List of available MCP configurations
    """
    return list(BUILTIN_MCPS.values())

def get_mcp_by_id(mcp_id: str) -> Optional[Dict[str, Any]]:
    """
    Get built-in MCP configuration by ID.
    
    Args:
        mcp_id: MCP server ID
        
    Returns:
        MCP configuration or None if not found
    """
    return BUILTIN_MCPS.get(mcp_id)

def register_built_in_mcps(location: Optional[str] = None) -> List[str]:
    """
    Register built-in MCP servers based on available API keys.
    
    Args:
        location: Directory to save configuration (default: current directory)
        
    Returns:
        List of registered MCP server IDs
    """
    registered_mcps = []
    servers = load_mcp_servers(location)
    server_dict = {server.get("id"): server for server in servers}
    
    for mcp_id, mcp_config in BUILTIN_MCPS.items():
        # Check if API key is required and available
        api_key_env = mcp_config.get("api_key_env")
        config = None
        
        if api_key_env:
            # This MCP requires an API key
            api_key = os.environ.get(api_key_env)
            if not api_key:
                logger.info(f"Skipping built-in MCP {mcp_id} due to missing {api_key_env}")
                continue
                
            # Create a copy of the configuration with the API key
            config = mcp_config.copy()
            
            # Add env dictionary if not present
            if "env" not in config:
                config["env"] = {}
                
            # Add API key to env
            config["env"][api_key_env] = api_key
        else:
            # No API key required, check for any config requirements
            config = mcp_config.copy()
            
            # Special handling for memory providers with data directory
            if mcp_id == "chroma-memory" and "config" in config and "data_dir" in config["config"]:
                # Expand data directory path
                data_dir = os.path.expanduser(config["config"]["data_dir"])
                os.makedirs(data_dir, exist_ok=True)
                config["config"]["data_dir"] = data_dir
                
                # Also update args if present
                if "args" in config:
                    for i, arg in enumerate(config["args"]):
                        if arg == "--data-dir" and i + 1 < len(config["args"]):
                            config["args"][i + 1] = data_dir
                            break
                    else:
                        config["args"].extend(["--data-dir", data_dir])
        
        if config:
            # Register this MCP
            server_dict[mcp_id] = config
            registered_mcps.append(mcp_id)
    
    # Save the updated configuration
    if registered_mcps:
        save_mcp_servers(list(server_dict.values()), location)
    
    return registered_mcps


def get_capability_mcps(capability: str, servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get MCP servers with the specified capability.
    
    Args:
        capability: Capability to filter by
        servers: List of server configurations
        
    Returns:
        List of MCP servers with the specified capability
    """
    result = []
    
    for server in servers:
        # Check direct capability field
        server_capability = server.get("capability")
        if server_capability == capability:
            result.append(server)
            continue
            
        # Check capabilities list if present
        capabilities = server.get("capabilities", [])
        if isinstance(capabilities, list) and capability in capabilities:
            result.append(server)
            continue
    
    return result


class MCPProvider:
    """Provider for MCP (Model Context Protocol) servers."""
    
    def __init__(self):
        """Initialize MCP provider with configured servers."""
        self.servers = load_mcp_servers()
        self.logger = logging.getLogger("agentscaffold.mcp")
        self.memory_cache = {}  # Local cache for memory operations
    
    async def execute(self, server_id: str, input_data: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        """
        Execute an MCP command on the specified server.
        
        Args:
            server_id: ID of the MCP server to use
            input_data: Input data for the command
            timeout: Timeout in seconds
            
        Returns:
            Result of the command execution
        """
        server = next((srv for srv in self.servers if srv.get("id") == server_id), None)
        if not server:
            raise ValueError(f"No MCP server found with ID: {server_id}")
        
        server_type = server.get("type")
        if server_type == "http":
            return await self._execute_http(server, input_data, timeout)
        elif server_type == "stdio":
            return await self._execute_stdio(server, input_data, timeout)
        else:
            raise ValueError(f"Unsupported MCP server type: {server_type}")
    
    async def _execute_http(self, server: Dict[str, Any], input_data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """
        Execute an MCP command on an HTTP server.
        
        Args:
            server: Server configuration
            input_data: Input data for the command
            timeout: Timeout in seconds
            
        Returns:
            Result of the command execution
        """
        url = server.get("url")
        server_id = server.get("id", "unknown")
        
        # Handle specific services directly
        if ("brave" in url.lower() or "brave" in server_id.lower()) and ("search" in server_id.lower() or CAPABILITY_SEARCH == server.get("capability")):
            return await self._execute_brave_search(server, input_data, timeout)
        elif ("chroma" in url.lower() or "chroma" in server_id.lower() or "memory" in server_id.lower() or 
              CAPABILITY_MEMORY == server.get("capability")):
            return await self._execute_chroma_memory(server, input_data, timeout)
        elif ("logfire" in url.lower() or "logfire" in server_id.lower() or "logging" in server_id.lower() or 
              CAPABILITY_LOGGING == server.get("capability")):
            return await self._execute_logfire_logging(server, input_data, timeout)
        
        # Try importing httpx
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx package is required for HTTP MCP")
        
        # Generic HTTP MCP handling
        endpoint = f"{url.rstrip('/')}/execute"
        headers = {"Content-Type": "application/json"}
        
        # Add authentication if available
        if server.get("apiKey"):
            headers["Authorization"] = f"Bearer {server['apiKey']}"
        elif server.get("bearer"):
            headers["Authorization"] = f"Bearer {server['bearer']}"
        elif server.get("env"):
            # Add env vars as headers or auth
            for key, value in server.get("env", {}).items():
                if "API_KEY" in key or "TOKEN" in key:
                    headers["Authorization"] = f"Bearer {value}"
        
        # Send request to MCP server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint, 
                json=input_data, 
                headers=headers, 
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"HTTP error {response.status_code}: {response.text}")
    
    async def _execute_brave_search(self, server: Dict[str, Any], input_data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """
        Execute a Brave Search query.
        
        Args:
            server: Server configuration
            input_data: Input data with query
            timeout: Timeout in seconds
            
        Returns:
            Search results
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx package is required for Brave Search")
        
        url = server.get("url")
        query = input_data.get("query", "")
        count = input_data.get("count", 3)
        
        search_url = f"{url.rstrip('/')}/res/v1/web/search?q={query}&count={count}"
        
        # Find the API key
        api_key = None
        if server.get("env") and server.get("env").get("BRAVE_API_KEY"):
            api_key = server.get("env").get("BRAVE_API_KEY")
        
        # If not in server config, try environment variables
        if not api_key:
            import os
            api_key = os.environ.get("BRAVE_API_KEY")
        
        if not api_key:
            # Attempt with no API key - might work for some queries
            self.logger.warning("No Brave API key found, search might be limited")
        
        headers = {"Accept": "application/json"}
        if api_key:
            headers["X-Subscription-Token"] = api_key
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    search_url, 
                    headers=headers, 
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    if "web" in data and "results" in data["web"]:
                        for result in data["web"]["results"][:count]:
                            results.append({
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "snippet": result.get("description", "")
                            })
                    return {"status": "success", "results": results}
                else:
                    # Log error details
                    self.logger.error(f"Brave Search error: {response.status_code}")
                    try:
                        error_body = response.json()
                        self.logger.error(f"Error details: {error_body}")
                    except:
                        self.logger.error(f"Error body: {response.text}")
                    
                    # Return error
                    return {
                        "status": "error", 
                        "code": response.status_code,
                        "message": f"Brave Search error: {response.status_code}",
                        "results": []
                    }
        except Exception as e:
            self.logger.error(f"Error calling Brave Search: {e}")
            return {"status": "error", "message": str(e), "results": []}
    
    async def _execute_chroma_memory(self, server: Dict[str, Any], input_data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """
        Execute a ChromaDB memory operation.
        
        Args:
            server: Server configuration
            input_data: Input data with operation and data
            timeout: Timeout in seconds
            
        Returns:
            Operation result
        """
        # Handle both direct dictionary format or wrap within a query/data
        operation = input_data.get("operation", "retrieve")
        if "data" in input_data:
            data = input_data.get("data", {})
        else:
            # Input data might BE the data directly
            data = input_data
        
        query = input_data.get("query", "")
        
        # Always maintain a local memory cache as backup
        if operation == "store" and data:
            memory_id = data.get("id", f"mem_{int(time.time())}")
            self.memory_cache[memory_id] = data
            
        # Try ChromaDB integration first
        try:
            # Try to import specialized ChromaDB client
            try:
                import chromadb
                from chromadb.config import Settings
                from chromadb.utils import embedding_functions
                
                # Configure ChromaDB client
                client_type = server.get("config", {}).get("client_type", "persistent")
                data_dir = server.get("config", {}).get("data_dir", "~/.agentscaffold/chroma")
                
                # Expand data directory path
                data_dir = os.path.expanduser(data_dir)
                os.makedirs(data_dir, exist_ok=True)
                
                client_settings = Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=data_dir
                )
                
                client = chromadb.Client(client_settings)
                
                # Get or create collection
                collection_name = input_data.get("collection", "default")
                try:
                    collection = client.get_collection(name=collection_name)
                except:
                    collection = client.create_collection(name=collection_name)
                
                # Common embedding function
                ef = embedding_functions.DefaultEmbeddingFunction()
                
                if operation == "store":
                    # Store data in ChromaDB
                    memory_id = data.get("id", f"mem_{int(time.time())}")
                    content = str(data.get("content", json.dumps(data)))
                    metadata = data
                    
                    # Add to collection
                    collection.add(
                        ids=[memory_id],
                        documents=[content],
                        metadatas=[metadata]
                    )
                    
                    return {"status": "success", "message": "Stored in ChromaDB", "id": memory_id}
                else:  # retrieve
                    # Query ChromaDB
                    query_text = query if query else "latest memories"
                    results = collection.query(
                        query_texts=[query_text],
                        n_results=input_data.get("n_results", 5)
                    )
                    
                    # Format results
                    metadatas = results.get("metadatas", [[]])
                    if metadatas and len(metadatas) > 0:
                        return {"status": "success", "results": metadatas[0]}
                    return {"status": "success", "results": []}
                    
            except ImportError:
                # ChromaDB not available locally, try HTTP API
                return await self._execute_chroma_http(server, input_data, timeout)
                
        except Exception as e:
            self.logger.error(f"Error with ChromaDB: {e}")
            # Fall back to local memory
            return self._use_local_memory(operation, query, data)
    
    async def _execute_chroma_http(self, server: Dict[str, Any], input_data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Use ChromaDB HTTP API"""
        try:
            import httpx
        except ImportError:
            # Fall back to local memory cache
            return self._use_local_memory(input_data.get("operation", "retrieve"), 
                                         input_data.get("query", ""), 
                                         input_data.get("data", {}))
        
        url = server.get("url")
        operation = input_data.get("operation", "retrieve")
        
        # Try to use the ChromaDB API
        try:
            # First check if collection exists
            collection_name = input_data.get("collection", "default")
            collection_url = f"{url.rstrip('/')}/api/v1/collections/{collection_name}"
            
            async with httpx.AsyncClient() as client:
                try:
                    collection_response = await client.get(collection_url, timeout=timeout)
                    if collection_response.status_code == 404:
                        # Create collection if it doesn't exist
                        create_url = f"{url.rstrip('/')}/api/v1/collections"
                        create_data = {"name": collection_name}
                        await client.post(create_url, json=create_data, timeout=timeout)
                except Exception as e:
                    self.logger.warning(f"Error checking/creating ChromaDB collection: {e}")
                    # Continue anyway - operation might still work
            
            if operation == "store":
                # Store data in ChromaDB
                data = input_data.get("data", {})
                endpoint = f"{url.rstrip('/')}/api/v1/collections/{collection_name}/add"
                
                # Generate ID if not provided
                memory_id = data.get("id", f"mem_{int(time.time())}")
                
                # Format for ChromaDB API
                payload = {
                    "ids": [memory_id],
                    "metadatas": [data],
                    "documents": [str(data.get("content", json.dumps(data)))]
                }
                
                # Also store in local cache as backup
                self.memory_cache[memory_id] = data
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(endpoint, json=payload, timeout=timeout)
                    
                    if response.status_code < 300:
                        return {"status": "success", "message": "Stored in ChromaDB", "id": memory_id}
                    else:
                        self.logger.warning(f"ChromaDB store error {response.status_code}: {response.text}")
                        return {
                            "status": "warning", 
                            "message": f"ChromaDB error: {response.status_code}. Using local memory.",
                            "id": memory_id
                        }
            else:  # retrieve
                query = input_data.get("query", "")
                endpoint = f"{url.rstrip('/')}/api/v1/collections/{collection_name}/query"
                
                # Format for ChromaDB API
                payload = {
                    "query_texts": [query if query else "latest memory"],
                    "n_results": input_data.get("n_results", 5)
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(endpoint, json=payload, timeout=timeout)
                    
                    if response.status_code < 300:
                        result = response.json()
                        # Extract metadatas from response
                        metadatas = result.get("metadatas", [[]])
                        if metadatas and len(metadatas) > 0:
                            return {"status": "success", "results": metadatas[0]}
                        return {"status": "success", "results": []}
                    else:
                        self.logger.warning(f"ChromaDB retrieve error {response.status_code}: {response.text}")
                        # Fall back to local memory
                        return self._use_local_memory(operation, query, None)
                        
        except Exception as e:
            self.logger.error(f"Error with ChromaDB: {e}")
            # Fall back to local memory
            return self._use_local_memory(operation, input_data.get("query", ""), input_data.get("data", {}))
    
    async def _execute_logfire_logging(self, server: Dict[str, Any], input_data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """
        Execute a LogFire logging operation.
        
        Args:
            server: Server configuration
            input_data: Input data with event and data
            timeout: Timeout in seconds
            
        Returns:
            Operation result
        """
        try:
            import httpx
        except ImportError:
            return {"status": "error", "message": "httpx package is required for LogFire logging"}
        
        url = server.get("url", "https://api.logfire.dev")
        event_type = input_data.get("event", input_data.get("event_type", "agent_event"))
        data = input_data.get("data", input_data)
        
        # Find API key
        api_key = None
        if server.get("env") and server.get("env").get("LOGFIRE_API_KEY"):
            api_key = server.get("env").get("LOGFIRE_API_KEY")
        
        if not api_key:
            # Try environment
            import os
            api_key = os.environ.get("LOGFIRE_API_KEY")
            
        if not api_key:
            return {"status": "error", "message": "No LogFire API key available"}
        
        # Prepare request
        endpoint = f"{url.rstrip('/')}/api/v1/log"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Prepare log data
        log_data = {"event": event_type}
        log_data.update(data)
        
        # Send request
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    json=log_data,
                    headers=headers,
                    timeout=timeout
                )
                
                if response.status_code < 300:
                    return {"status": "success", "message": "Event logged to LogFire"}
                else:
                    self.logger.error(f"LogFire error: {response.status_code}")
                    return {"status": "error", "message": f"LogFire API error: {response.status_code}"}
        except Exception as e:
            self.logger.error(f"Error with LogFire: {e}")
            return {"status": "error", "message": str(e)}
    
    def _use_local_memory(self, operation: str, query: str = "", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use local memory cache as fallback.
        
        Args:
            operation: Operation type (store or retrieve)
            query: Query string for retrieval
            data: Data to store
            
        Returns:
            Operation result
        """
        if operation == "store" and data:
            # Store in local memory cache
            memory_id = data.get("id", f"mem_{int(time.time())}")
            self.memory_cache[memory_id] = data
            return {"status": "success", "message": "Stored in local memory", "id": memory_id}
        else:  # retrieve
            # Query local memory cache
            query = query.lower() if query else ""
            
            if not query:
                # Return most recent memories
                memories = list(self.memory_cache.values())[-5:]
                return {"status": "success", "results": memories}
            else:
                # Search in memory cache
                matching = []
                for mem in self.memory_cache.values():
                    # Convert memory to string for searching
                    mem_str = json.dumps(mem).lower()
                    if query in mem_str:
                        matching.append(mem)
                return {"status": "success", "results": matching}
    
    async def _execute_stdio(self, server: Dict[str, Any], input_data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """
        Execute an MCP command using stdio communication.
        
        Args:
            server: Server configuration
            input_data: Input data for the command
            timeout: Timeout in seconds
            
        Returns:
            Result of the command execution
        """
        command = server.get("command")
        if not command:
            raise ValueError("Missing command for stdio MCP server")
        
        args = server.get("args", [])
        env = os.environ.copy()
        
        # Add environment variables
        if server.get("env"):
            for key, value in server.get("env", {}).items():
                env[key] = value
        
        # Create process
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # Encode input as JSON
        input_json = json.dumps(input_data) + "\n"
        
        try:
            # Write to stdin
            process.stdin.write(input_json.encode())
            await process.stdin.drain()
            
            # Read from stdout with timeout
            result = await asyncio.wait_for(process.stdout.read(), timeout)
            
            try:
                # Parse result as JSON
                return json.loads(result)
            except json.JSONDecodeError:
                return {"status": "error", "message": f"Invalid JSON response: {result.decode()[:100]}..."}
        except asyncio.TimeoutError:
            # Kill process if it times out
            process.kill()
            raise TimeoutError(f"MCP stdio command timed out after {timeout} seconds")
        finally:
            # Make sure process is terminated
            if process.returncode is None:
                process.kill()