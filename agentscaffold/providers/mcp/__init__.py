"""MCP (Model Context Protocol) server integration for AgentScaffold."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# MCP configuration is per-agent, stored in the agent's directory
MCP_CONFIG_FILE = ".mcp.json"

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

def load_mcp_servers(location: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load MCP servers configuration from the specified location.
    
    Args:
        location: Directory containing the configuration (default: current directory)
        
    Returns:
        Dictionary of server configurations
    """
    config_path = get_config_path(location)
    
    if not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get("mcpServers", {})
    except (json.JSONDecodeError, IOError):
        return {}

def save_mcp_servers(servers: Dict[str, Dict[str, Any]], location: Optional[str] = None) -> None:
    """
    Save MCP servers configuration to the specified location.
    
    Args:
        servers: Dictionary of server configurations
        location: Directory to save to (default: current directory)
    """
    config_path = get_config_path(location)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Check if file exists to preserve any other settings
    config = {"mcpServers": servers}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
                # Preserve other keys, only update mcpServers
                for key, value in existing_config.items():
                    if key != "mcpServers":
                        config[key] = value
        except (json.JSONDecodeError, IOError):
            pass
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)