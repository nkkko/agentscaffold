"""MCP (Model Context Protocol) server integration for AgentScaffold."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# MCP configuration is per-agent, stored in the agent's directory
MCP_CONFIG_FILE = ".mcp.json"

def load_mcp_servers(agent_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load MCP servers configuration from the specified agent directory.
    
    Args:
        agent_dir: Directory containing the agent (default: current directory)
        
    Returns:
        Dictionary of server configurations
    """
    if agent_dir is None:
        agent_dir = os.getcwd()
        
    config_path = os.path.join(agent_dir, MCP_CONFIG_FILE)
    
    if not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get("mcpServers", {})
    except json.JSONDecodeError:
        return {}

def save_mcp_servers(servers: Dict[str, Dict[str, Any]], agent_dir: Optional[str] = None) -> None:
    """
    Save MCP servers configuration to the specified agent directory.
    
    Args:
        servers: Dictionary of server configurations
        agent_dir: Directory containing the agent (default: current directory)
    """
    if agent_dir is None:
        agent_dir = os.getcwd()
        
    config_path = os.path.join(agent_dir, MCP_CONFIG_FILE)
    
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
        except json.JSONDecodeError:
            pass
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)