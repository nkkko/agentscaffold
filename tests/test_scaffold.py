"""Tests for the scaffold module."""

import pytest
import os
import shutil
from pathlib import Path
from agentscaffold.scaffold import create_new_agent


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    yield tmp_path
    # Clean up
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)


def test_create_new_agent(temp_dir):
    """Test that a new agent can be created."""
    agent_name = "test-agent"
    create_new_agent(agent_name, "basic", temp_dir)
    
    # Check that the agent directory was created
    agent_dir = temp_dir / agent_name
    assert os.path.exists(agent_dir)
    
    # Check that the package directory was created
    package_dir = agent_dir / "test_agent"
    assert os.path.exists(package_dir)
    
    # Check that the required files were created
    assert os.path.exists(package_dir / "__init__.py")
    assert os.path.exists(package_dir / "agent.py")
    assert os.path.exists(agent_dir / "pyproject.toml")
    assert os.path.exists(agent_dir / "README.md")
    assert os.path.exists(agent_dir / "main.py")