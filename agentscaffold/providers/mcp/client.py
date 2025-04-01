"""
Model Context Protocol (MCP) client implementation for AgentScaffold.

This module provides a client for interacting with MCP servers, allowing
agents to access various capabilities like search, memory, and logging.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
import aiohttp
import requests
from pydantic import BaseModel, Field

from .base import MCPProvider, MCPConfig

class MCPRequest(BaseModel):
    """Model for MCP request data."""
    server_id: str = Field(..., description="ID of the MCP server to invoke")
    operation: str = Field(..., description="Operation to perform")
    data: Dict[str, Any] = Field(default_factory=dict, description="Request data")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class MCPResponse(BaseModel):
    """Model for MCP response data."""
    status: str = Field(..., description="Status of the response (success/error)")
    data: Dict[str, Any] = Field(default_factory=dict, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if status is error")


class MCPClient(MCPProvider):
    """Client for interacting with MCP servers."""
    
    def __init__(
        self,
        config: Optional[MCPConfig] = None,
        **kwargs
    ):
        """
        Initialize the MCP client.
        
        Args:
            config: MCP configuration
            **kwargs: Additional configuration options
        """
        # Initialize base provider
        super().__init__(config=config or MCPConfig(), **kwargs)
        
        # Setup logging
        self.logger = logging.getLogger("mcp.client")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize session for async requests
        self._session = None
        
        # Track registered servers and handlers
        self._servers = {}
        self._handlers = {}
        
        # Register default servers
        self._register_default_servers()
        
        self.logger.info("Initialized MCP client")
    
    def _register_default_servers(self):
        """Register default MCP servers."""
        # Register search servers
        self.register_server("brave-search", "search", "Brave Search service")
        self.register_server("google-search", "search", "Google Search service")
        
        # Register memory servers
        self.register_server("chroma-memory", "memory", "ChromaDB memory service")
        self.register_server("pinecone-memory", "memory", "Pinecone memory service")
        self.register_server("supabase-memory", "memory", "Supabase memory service")
        
        # Register logging servers
        self.register_server("logfire-logging", "logging", "LogFire logging service")
        self.register_server("langfuse-logging", "logging", "Langfuse logging service")
        
        # Register default handlers
        self.register_handler("search", self._handle_search)
        self.register_handler("memory", self._handle_memory)
        self.register_handler("logging", self._handle_logging)
    
    def register_server(self, server_id: str, server_type: str, description: str = ""):
        """
        Register an MCP server.
        
        Args:
            server_id: Unique identifier for the server
            server_type: Type of the server (search, memory, etc.)
            description: Optional description
        """
        self._servers[server_id] = {
            "type": server_type,
            "description": description,
            "registered_at": time.time()
        }
        self.logger.info(f"Registered MCP server: {server_id} ({server_type})")
    
    def register_handler(self, server_type: str, handler: Callable):
        """
        Register a handler for a server type.
        
        Args:
            server_type: Type of server
            handler: Handler function for the server type
        """
        self._handlers[server_type] = handler
        self.logger.info(f"Registered handler for server type: {server_type}")
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """
        List all registered MCP servers.
        
        Returns:
            List of server information
        """
        return [
            {"id": server_id, **info}
            for server_id, info in self._servers.items()
        ]
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
    
    async def _close_session(self):
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def invoke(self, request: Union[MCPRequest, Dict[str, Any]]) -> MCPResponse:
        """
        Invoke an MCP server.
        
        Args:
            request: MCP request data
            
        Returns:
            MCP response
        """
        try:
            # Convert dict to MCPRequest if needed
            if isinstance(request, dict):
                request = MCPRequest(**request)
            
            server_id = request.server_id
            operation = request.operation
            data = request.data
            
            # Check if server exists
            if server_id not in self._servers:
                self.logger.error(f"Unknown MCP server: {server_id}")
                return MCPResponse(
                    status="error",
                    error=f"Unknown MCP server: {server_id}"
                )
            
            server_type = self._servers[server_id]["type"]
            
            # Check if handler exists for server type
            if server_type not in self._handlers:
                self.logger.error(f"No handler for server type: {server_type}")
                return MCPResponse(
                    status="error",
                    error=f"No handler for server type: {server_type}"
                )
            
            # Get handler and invoke
            handler = self._handlers[server_type]
            self.logger.info(f"Invoking MCP server: {server_id} (type: {server_type}, operation: {operation})")
            
            # Invoke handler
            result = await handler(server_id, operation, data)
            
            # Convert result to MCPResponse
            if isinstance(result, dict):
                status = result.pop("status", "success")
                error = result.pop("error", None)
                return MCPResponse(
                    status=status,
                    data=result,
                    error=error
                )
            elif isinstance(result, MCPResponse):
                return result
            else:
                return MCPResponse(
                    status="success",
                    data={"result": result}
                )
        
        except Exception as e:
            self.logger.error(f"Error invoking MCP server: {e}", exc_info=True)
            return MCPResponse(
                status="error",
                error=str(e)
            )
    
    async def _handle_search(self, server_id: str, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle search server requests.
        
        Args:
            server_id: Server ID
            operation: Operation to perform
            data: Request data
            
        Returns:
            Search results
        """
        query = data.get("query", "")
        if not query:
            return {"status": "error", "error": "No search query provided"}
        
        if "brave" in server_id.lower():
            return await self._brave_search(query, data.get("count", 3))
        elif "google" in server_id.lower():
            return await self._google_search(query, data.get("count", 3))
        else:
            return {"status": "error", "error": f"Unsupported search provider: {server_id}"}
    
    async def _brave_search(self, query: str, count: int = 3) -> Dict[str, Any]:
        """
        Perform a search using Brave Search.
        
        Args:
            query: Search query
            count: Number of results to return
            
        Returns:
            Search results
        """
        api_key = os.environ.get("BRAVE_API_KEY")
        if not api_key:
            return {"status": "error", "error": "No Brave API key provided"}
        
        try:
            # Ensure session is created
            await self._ensure_session()
            
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": api_key
            }
            
            params = {
                "q": query,
                "count": count
            }
            
            async with self._session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10
            ) as response:
                if response.status != 200:
                    self.logger.error(f"Brave Search API error: {response.status}")
                    return {
                        "status": "error",
                        "error": f"Brave Search API returned status {response.status}",
                        "results": []
                    }
                
                data = await response.json()
                results = []
                
                if "web" in data and "results" in data["web"]:
                    for result in data["web"]["results"][:count]:
                        results.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "snippet": result.get("description", "")
                        })
                
                return {
                    "status": "success",
                    "results": results
                }
        
        except Exception as e:
            self.logger.error(f"Error with Brave Search: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }
    
    async def _google_search(self, query: str, count: int = 3) -> Dict[str, Any]:
        """
        Perform a search using Google Search (placeholder implementation).
        
        Args:
            query: Search query
            count: Number of results to return
            
        Returns:
            Search results
        """
        # Placeholder for Google Search API
        self.logger.warning("Google Search API not implemented")
        return {
            "status": "warning",
            "message": "Google Search API not implemented",
            "results": []
        }
    
    async def _handle_memory(self, server_id: str, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle memory server requests.
        
        Args:
            server_id: Server ID
            operation: Operation to perform (store, retrieve)
            data: Request data
            
        Returns:
            Memory operation result
        """
        if operation == "store":
            content = data.get("content")
            if not content:
                return {"status": "error", "error": "No content provided for memory storage"}
            
            memory_id = data.get("id", f"mem_{int(time.time())}")
            metadata = data.get("metadata", {})
            
            if "chroma" in server_id.lower():
                return await self._chroma_store(memory_id, content, metadata)
            elif "pinecone" in server_id.lower():
                return await self._pinecone_store(memory_id, content, metadata)
            elif "supabase" in server_id.lower():
                return await self._supabase_store(memory_id, content, metadata)
            else:
                # Default local memory storage
                return await self._local_store(memory_id, content, metadata)
        
        elif operation == "retrieve":
            query = data.get("query", "")
            limit = data.get("limit", 3)
            
            if "chroma" in server_id.lower():
                return await self._chroma_retrieve(query, limit)
            elif "pinecone" in server_id.lower():
                return await self._pinecone_retrieve(query, limit)
            elif "supabase" in server_id.lower():
                return await self._supabase_retrieve(query, limit)
            else:
                # Default local memory retrieval
                return await self._local_retrieve(query, limit)
        
        else:
            return {"status": "error", "error": f"Unsupported memory operation: {operation}"}
    
    async def _local_store(self, memory_id: str, content: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store data in local memory (file-based).
        
        Args:
            memory_id: Memory ID
            content: Content to store
            metadata: Additional metadata
            
        Returns:
            Storage result
        """
        try:
            # Create memory directory if it doesn't exist
            os.makedirs("memory", exist_ok=True)
            
            # Prepare memory entry
            memory_entry = {
                "id": memory_id,
                "content": content,
                "metadata": metadata,
                "timestamp": time.time()
            }
            
            # Store memory entry
            memory_file = os.path.join("memory", f"{memory_id}.json")
            with open(memory_file, "w") as f:
                json.dump(memory_entry, f)
            
            # Update memory index
            index_file = os.path.join("memory", "index.json")
            if os.path.exists(index_file):
                with open(index_file, "r") as f:
                    try:
                        index = json.load(f)
                    except json.JSONDecodeError:
                        index = []
            else:
                index = []
            
            # Add to index
            index.append({
                "id": memory_id,
                "timestamp": memory_entry["timestamp"],
                "metadata": metadata
            })
            
            # Save index
            with open(index_file, "w") as f:
                json.dump(index, f)
            
            self.logger.info(f"Stored memory: {memory_id}")
            
            return {
                "status": "success",
                "id": memory_id,
                "message": "Stored in local memory"
            }
        
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _local_retrieve(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        Retrieve data from local memory (file-based).
        
        Args:
            query: Query to search for
            limit: Maximum number of results
            
        Returns:
            Retrieval result
        """
        try:
            # Check if memory directory exists
            if not os.path.exists("memory"):
                return {
                    "status": "warning",
                    "message": "No memory found",
                    "results": []
                }
            
            # Check if index exists
            index_file = os.path.join("memory", "index.json")
            if not os.path.exists(index_file):
                return {
                    "status": "warning",
                    "message": "No memory index found",
                    "results": []
                }
            
            # Load index
            with open(index_file, "r") as f:
                try:
                    index = json.load(f)
                except json.JSONDecodeError:
                    return {
                        "status": "error",
                        "error": "Invalid memory index format",
                        "results": []
                    }
            
            # Find matching entries
            results = []
            
            if query:
                # Search for matching entries
                query_lower = query.lower()
                for entry in index:
                    memory_id = entry.get("id")
                    if not memory_id:
                        continue
                    
                    memory_file = os.path.join("memory", f"{memory_id}.json")
                    if not os.path.exists(memory_file):
                        continue
                    
                    with open(memory_file, "r") as f:
                        try:
                            memory_data = json.load(f)
                            content = str(memory_data.get("content", "")).lower()
                            if query_lower in content:
                                results.append(memory_data)
                        except json.JSONDecodeError:
                            continue
            else:
                # No query, return most recent entries
                sorted_index = sorted(index, key=lambda x: x.get("timestamp", 0), reverse=True)
                
                for entry in sorted_index[:limit]:
                    memory_id = entry.get("id")
                    if not memory_id:
                        continue
                    
                    memory_file = os.path.join("memory", f"{memory_id}.json")
                    if not os.path.exists(memory_file):
                        continue
                    
                    with open(memory_file, "r") as f:
                        try:
                            memory_data = json.load(f)
                            results.append(memory_data)
                        except json.JSONDecodeError:
                            continue
            
            # Limit results
            results = results[:limit]
            
            if results:
                return {
                    "status": "success",
                    "results": results
                }
            else:
                return {
                    "status": "warning",
                    "message": "No matching memories found",
                    "results": []
                }
        
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }
    
    # Placeholder methods for other memory providers
    async def _chroma_store(self, memory_id: str, content: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("ChromaDB storage not implemented")
        return await self._local_store(memory_id, content, metadata)
    
    async def _chroma_retrieve(self, query: str, limit: int = 3) -> Dict[str, Any]:
        self.logger.warning("ChromaDB retrieval not implemented")
        return await self._local_retrieve(query, limit)
    
    async def _pinecone_store(self, memory_id: str, content: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("Pinecone storage not implemented")
        return await self._local_store(memory_id, content, metadata)
    
    async def _pinecone_retrieve(self, query: str, limit: int = 3) -> Dict[str, Any]:
        self.logger.warning("Pinecone retrieval not implemented")
        return await self._local_retrieve(query, limit)
    
    async def _supabase_store(self, memory_id: str, content: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("Supabase storage not implemented")
        return await self._local_store(memory_id, content, metadata)
    
    async def _supabase_retrieve(self, query: str, limit: int = 3) -> Dict[str, Any]:
        self.logger.warning("Supabase retrieval not implemented")
        return await self._local_retrieve(query, limit)
    
    async def _handle_logging(self, server_id: str, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle logging server requests.
        
        Args:
            server_id: Server ID
            operation: Operation to perform
            data: Request data
            
        Returns:
            Logging result
        """
        event_type = data.get("event", "custom_event")
        event_data = data.get("data", {})
        
        if "logfire" in server_id.lower():
            return await self._logfire_log(event_type, event_data)
        elif "langfuse" in server_id.lower():
            return await self._langfuse_log(event_type, event_data)
        else:
            # Default local logging
            return await self._local_log(event_type, event_data)
    
    async def _local_log(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log an event locally.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            Logging result
        """
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Prepare log entry
            log_entry = {
                "id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "event": event_type,
                "data": data
            }
            
            # Write to log file
            log_file = os.path.join("logs", "agent_logs.jsonl")
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            self.logger.info(f"Logged event: {event_type}")
            
            return {
                "status": "success",
                "message": "Event logged locally",
                "id": log_entry["id"]
            }
        
        except Exception as e:
            self.logger.error(f"Error logging event: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Placeholder methods for other logging providers
    async def _logfire_log(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log an event using LogFire.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            Logging result
        """
        api_key = os.environ.get("LOGFIRE_API_KEY")
        if not api_key:
            self.logger.warning("No LogFire API key provided, falling back to local logging")
            return await self._local_log(event_type, data)
        
        try:
            # Ensure session is created
            await self._ensure_session()
            
            # Prepare log entry
            log_entry = {
                "timestamp": time.time(),
                "event": event_type,
                "data": data
            }
            
            # Send to LogFire API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            async with self._session.post(
                "https://in.logfire.dev/ingest",
                headers=headers,
                json=[log_entry],
                timeout=10
            ) as response:
                if response.status != 200:
                    self.logger.error(f"LogFire API error: {response.status}")
                    # Fall back to local logging
                    self.logger.info("Falling back to local logging")
                    return await self._local_log(event_type, data)
                
                self.logger.info(f"Logged event to LogFire: {event_type}")
                
                return {
                    "status": "success",
                    "message": "Event logged to LogFire"
                }
        
        except Exception as e:
            self.logger.error(f"Error with LogFire logging: {e}", exc_info=True)
            # Fall back to local logging
            self.logger.info("Falling back to local logging due to error")
            return await self._local_log(event_type, data)
    
    async def _langfuse_log(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("Langfuse logging not implemented")
        return await self._local_log(event_type, data)
    
    async def close(self):
        """Close the client and clean up resources."""
        await self._close_session()
        self.logger.info("Closed MCP client")