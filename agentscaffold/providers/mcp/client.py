"""Client implementation for MCP (Model Context Protocol) servers."""

import json
import asyncio
import os
from typing import Dict, Any, List, Optional, Union

# Provide fallbacks if dependencies are missing
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

def test_mcp_connection(server_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test connection to an MCP server and return capabilities.
    
    Args:
        server_config: Dictionary containing server configuration
        
    Returns:
        Dictionary with connection status and server information
    """
    url = server_config["url"]
    api_key = server_config.get("api_key")
    
    if not HAS_HTTPX:
        return {
            "success": False,
            "error": "httpx package is required for MCP connectivity"
        }
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        # Try to get server capabilities
        with httpx.Client(timeout=10.0) as client:
            # First try /server_info endpoint
            try:
                response = client.get(f"{url}/server_info", headers=headers)
                response.raise_for_status()
                data = response.json()
                
                return {
                    "success": True,
                    "capabilities": data.get("capabilities", []),
                    "tools": data.get("tools", []),
                }
            except httpx.HTTPStatusError:
                # Fallback to /tools endpoint for basic connectivity test
                response = client.get(f"{url}/tools", headers=headers)
                response.raise_for_status()
                data = response.json()
                
                return {
                    "success": True,
                    "capabilities": ["tools"],
                    "tools": data.get("tools", []),
                }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

class MCPClient:
    """Client for interacting with MCP servers."""
    
    def __init__(self, server_config: Dict[str, Any], name: str = ""):
        """
        Initialize an MCP client.
        
        Args:
            server_config: Dictionary containing server configuration
            name: Optional server name
        """
        self.config = server_config
        self.name = name
        self.server_type = server_config.get("type", "http")
        
        # HTTP server setup
        if self.server_type == "http":
            self.url = server_config.get("url", "")
            self.api_key = server_config.get("apiKey")
            self.bearer_token = server_config.get("bearer")
            self.capabilities = server_config.get("capabilities", ["tools"])
            
            # Verify httpx is available for HTTP servers
            if not HAS_HTTPX:
                print("Warning: httpx package is required for HTTP MCP connectivity")
        
        # Process-based server setup 
        elif self.server_type == "stdio":
            self.command = server_config.get("command")
            self.args = server_config.get("args", [])
            self.env = server_config.get("env", {})
            # For now, assume all stdio servers have tools capability
            self.capabilities = ["tools"]
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools from the MCP server.
        
        Returns:
            List of tool definitions
        """
        if self.server_type == "http":
            if not HAS_HTTPX:
                return []
                
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    response = await client.get(f"{self.url}/tools", headers=headers)
                    response.raise_for_status()
                    return response.json().get("tools", [])
                except Exception as e:
                    print(f"Error getting tools from MCP server {self.name}: {e}")
                    return []
        elif self.server_type == "stdio":
            # For stdio servers, we need to run the command and get the tools
            # This is more complex and would require subprocess management
            # For now, we'll return a placeholder
            return [{"name": "stdio_command", "description": "Server runs as a local process"}]
        
        return []
    
    async def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to invoke
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        if self.server_type == "http":
            if not HAS_HTTPX:
                return {
                    "error": "httpx package is required for HTTP MCP connectivity",
                    "success": False
                }
                
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"
            
            payload = {
                "name": tool_name,
                "parameters": params
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        f"{self.url}/tools/invoke", 
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    return {
                        "error": str(e),
                        "success": False
                    }
        elif self.server_type == "stdio":
            # For stdio servers, we need to run the command with the tool invocation
            # This is more complex and would require subprocess management
            # For now, return a placeholder error
            return {
                "error": "Process-based MCP servers not yet fully implemented",
                "success": False
            }
        
        return {
            "error": f"Unsupported server type: {self.server_type}",
            "success": False
        }