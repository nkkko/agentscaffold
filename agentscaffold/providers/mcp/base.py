"""Base implementation for MCP (Model Context Protocol) servers."""

from typing import Dict, Any, Optional, List, Set, Union

class MCPServer:
    """Base class for MCP server configuration."""
    
    def __init__(
        self, 
        name: str, 
        url: str, 
        api_key: Optional[str] = None, 
        auth_type: str = "api_key", 
        capabilities: Optional[List[str]] = None
    ):
        """
        Initialize an MCP server configuration.
        
        Args:
            name: Unique name for the server
            url: Server URL
            api_key: API key for authentication (optional)
            auth_type: Authentication type (api_key, oauth, none)
            capabilities: List of server capabilities (tools, resources, prompts)
        """
        self.name = name
        self.url = url
        self.api_key = api_key
        self.auth_type = auth_type
        self.capabilities = capabilities or ["tools"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert server to dictionary for storage."""
        return {
            "name": self.name,
            "url": self.url,
            "api_key": self.api_key,
            "auth_type": self.auth_type,
            "capabilities": self.capabilities
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPServer':
        """
        Create server from dictionary.
        
        Args:
            data: Dictionary containing server configuration
            
        Returns:
            MCPServer instance
        """
        return cls(
            name=data["name"],
            url=data["url"],
            api_key=data.get("api_key"),
            auth_type=data.get("auth_type", "api_key"),
            capabilities=data.get("capabilities", ["tools"])
        )