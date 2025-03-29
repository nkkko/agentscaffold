"""Tests for MCP functionality."""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentscaffold.providers.mcp import load_mcp_servers, save_mcp_servers
from agentscaffold.providers.mcp.base import MCPServer
from agentscaffold.providers.mcp.client import test_mcp_connection

# Test server configurations
HTTP_SERVER = {
    "type": "http",
    "url": "http://localhost:8080",
    "apiKey": "test-api-key",
    "capabilities": ["tools"]
}

STDIO_SERVER = {
    "type": "stdio",
    "command": "/usr/bin/python",
    "args": ["server.py"],
    "env": {
        "API_KEY": "test-key",
        "DEBUG": "true"
    }
}

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for MCP configuration."""
    return tmp_path

def test_mcp_server_creation():
    """Test creating an MCP server configuration."""
    server = MCPServer(
        name="test-server",
        url="http://localhost:8080",
        api_key="test-api-key"
    )
    
    assert server.name == "test-server"
    assert server.url == "http://localhost:8080"
    assert server.api_key == "test-api-key"
    assert server.auth_type == "api_key"
    assert server.capabilities == ["tools"]
    
    # Test to_dict and from_dict
    server_dict = server.to_dict()
    assert server_dict["name"] == "test-server"
    
    server2 = MCPServer.from_dict(server_dict)
    assert server2.name == server.name
    assert server2.url == server.url
    assert server2.api_key == server.api_key

def test_save_load_servers(temp_config_dir):
    """Test saving and loading MCP server configurations."""
    with patch('agentscaffold.providers.mcp.os.getcwd', return_value=str(temp_config_dir)):
        # Save servers
        servers = {
            "http-server": HTTP_SERVER,
            "stdio-server": STDIO_SERVER
        }
        save_mcp_servers(servers)
        
        # Check that the file was created
        config_path = temp_config_dir / ".mcp.json"
        assert config_path.exists()
        
        # Verify the file structure
        with open(config_path, 'r') as f:
            config = json.load(f)
            assert "mcpServers" in config
            assert "http-server" in config["mcpServers"]
            assert "stdio-server" in config["mcpServers"]
        
        # Load servers
        loaded_servers = load_mcp_servers()
        assert loaded_servers == servers
        assert "http-server" in loaded_servers
        assert "stdio-server" in loaded_servers
        assert loaded_servers["http-server"]["url"] == "http://localhost:8080"
        assert loaded_servers["stdio-server"]["command"] == "/usr/bin/python"

@patch('httpx.Client')
def test_http_mcp_connection(mock_client):
    """Test HTTP MCP server connection testing."""
    # Mock the response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "capabilities": ["tools"],
        "tools": [{"name": "test_tool", "description": "A test tool"}]
    }
    
    # Set up the mock client
    mock_client_instance = MagicMock()
    mock_client_instance.get.return_value = mock_response
    mock_client.return_value.__enter__.return_value = mock_client_instance
    
    # Test the connection
    result = test_mcp_connection(HTTP_SERVER)
    
    # Verify the result
    assert result["success"] is True
    assert "capabilities" in result
    assert "tools" in result
    assert result["capabilities"] == ["tools"]
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "test_tool"

def test_stdio_mcp_client():
    """Test STDIO MCP client initialization."""
    from agentscaffold.providers.mcp.client import MCPClient
    
    # Initialize a client for the STDIO server
    client = MCPClient(STDIO_SERVER, name="stdio-test")
    
    # Verify the client properties
    assert client.server_type == "stdio"
    assert client.command == "/usr/bin/python"
    assert client.args == ["server.py"]
    assert "API_KEY" in client.env
    assert client.env["API_KEY"] == "test-key"