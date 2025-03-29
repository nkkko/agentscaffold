"""Tests for MCP functionality."""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentscaffold.providers.mcp import load_mcp_servers, save_mcp_servers
from agentscaffold.providers.mcp.base import MCPServer
from agentscaffold.providers.mcp.client import test_mcp_connection

# Test server configuration
TEST_SERVER = {
    "name": "test-server",
    "url": "http://localhost:8080",
    "api_key": "test-api-key",
    "auth_type": "api_key",
    "capabilities": ["tools"]
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
        servers = {"test-server": TEST_SERVER}
        save_mcp_servers(servers)
        
        # Check that the file was created
        config_path = temp_config_dir / "mcp_servers.json"
        assert config_path.exists()
        
        # Load servers
        loaded_servers = load_mcp_servers()
        assert loaded_servers == servers
        assert "test-server" in loaded_servers
        assert loaded_servers["test-server"]["url"] == "http://localhost:8080"

@patch('httpx.Client')
def test_mcp_connection(mock_client):
    """Test MCP server connection testing."""
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
    result = test_mcp_connection(TEST_SERVER)
    
    # Verify the result
    assert result["success"] is True
    assert "capabilities" in result
    assert "tools" in result
    assert result["capabilities"] == ["tools"]
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "test_tool"