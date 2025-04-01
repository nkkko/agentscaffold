"""
Base implementation for Model Context Protocol (MCP) providers in AgentScaffold.

This module defines the base classes for MCP providers, including the
configuration and abstract provider interface.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field


class MCPConfig(BaseModel):
    """Configuration for MCP providers."""
    
    api_key: Optional[str] = Field(default=None, description="API key for MCP provider")
    server_url: Optional[str] = Field(default=None, description="Server URL for MCP provider")
    enable_logging: bool = Field(default=True, description="Enable logging of MCP operations")
    timeout: int = Field(default=30, description="Default timeout for MCP operations in seconds")
    
    model_config = {"extra": "allow"}


class MCPProvider:
    """Base class for MCP providers."""
    
    def __init__(
        self,
        config: Optional[MCPConfig] = None,
        **kwargs
    ):
        """
        Initialize the MCP provider.
        
        Args:
            config: MCP configuration
            **kwargs: Additional configuration options
        """
        self.config = config or MCPConfig()
        self.options = kwargs
        
        # Setup logging
        self.logger = logging.getLogger("mcp.provider")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    async def invoke(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke an MCP operation.
        
        Args:
            request: MCP request data
            
        Returns:
            MCP response
        """
        self.logger.warning("Base MCP provider does not implement invoke")
        return {
            "status": "error",
            "error": "Base MCP provider does not implement invoke"
        }
    
    async def close(self):
        """Close the provider and clean up resources."""
        pass


# Convenience functions for working with MCP providers

async def mcp_invoke(provider: MCPProvider, server_id: str, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke an MCP operation on a provider.
    
    Args:
        provider: MCP provider
        server_id: Server ID
        operation: Operation to perform
        data: Request data
        
    Returns:
        MCP response
    """
    request = {
        "server_id": server_id,
        "operation": operation,
        "data": data
    }
    return await provider.invoke(request)


async def mcp_search(provider: MCPProvider, query: str, server_id: str = "brave-search") -> List[Dict[str, Any]]:
    """
    Perform a search using an MCP provider.
    
    Args:
        provider: MCP provider
        query: Search query
        server_id: Server ID for search provider
        
    Returns:
        Search results
    """
    response = await mcp_invoke(provider, server_id, "search", {"query": query})
    if response.get("status") == "success":
        return response.get("results", [])
    return []


async def mcp_store_memory(provider: MCPProvider, content: Any, metadata: Dict[str, Any] = None, server_id: str = "chroma-memory") -> str:
    """
    Store data in memory using an MCP provider.
    
    Args:
        provider: MCP provider
        content: Content to store
        metadata: Additional metadata
        server_id: Server ID for memory provider
        
    Returns:
        Memory ID
    """
    metadata = metadata or {}
    response = await mcp_invoke(provider, server_id, "store", {
        "content": content,
        "metadata": metadata
    })
    if response.get("status") == "success":
        return response.get("id", "")
    return ""


async def mcp_retrieve_memory(provider: MCPProvider, query: str = "", limit: int = 3, server_id: str = "chroma-memory") -> List[Dict[str, Any]]:
    """
    Retrieve data from memory using an MCP provider.
    
    Args:
        provider: MCP provider
        query: Query to search for
        limit: Maximum number of results
        server_id: Server ID for memory provider
        
    Returns:
        Memory entries
    """
    response = await mcp_invoke(provider, server_id, "retrieve", {
        "query": query,
        "limit": limit
    })
    if response.get("status") == "success":
        return response.get("results", [])
    return []


async def mcp_log_event(provider: MCPProvider, event_type: str, data: Dict[str, Any] = None, server_id: str = "logfire-logging") -> bool:
    """
    Log an event using an MCP provider.
    
    Args:
        provider: MCP provider
        event_type: Type of event
        data: Event data
        server_id: Server ID for logging provider
        
    Returns:
        True if successful, False otherwise
    """
    data = data or {}
    response = await mcp_invoke(provider, server_id, "log", {
        "event": event_type,
        "data": data
    })
    return response.get("status") == "success"