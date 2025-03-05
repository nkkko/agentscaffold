"""Tests for the agent module."""

import pytest
import os
import unittest.mock
from agentscaffold.agent import Agent, AgentInput, AgentOutput, DaytonaRuntime


def test_agent_initialization():
    """Test that the agent initializes correctly."""
    agent = Agent(name="TestAgent")
    assert agent.name == "TestAgent"
    assert isinstance(agent.runtime, DaytonaRuntime)


def test_agent_input_validation():
    """Test that agent input is correctly validated."""
    input_data = {"message": "Hello", "context": {"user": "test"}}
    agent_input = AgentInput(**input_data)
    assert agent_input.message == "Hello"
    assert agent_input.context == {"user": "test"}


def test_agent_output_validation():
    """Test that agent output is correctly validated."""
    output_data = {"response": "Hello back", "metadata": {"processed": True}}
    agent_output = AgentOutput(**output_data)
    assert agent_output.response == "Hello back"
    assert agent_output.metadata == {"processed": True}


def test_agent_run():
    """Test that the agent runs correctly."""
    # Create a mock runtime to avoid Daytona SDK import
    mock_runtime = unittest.mock.MagicMock()
    mock_runtime.execute.return_value = {"response": "Received: Hello", "metadata": {}}
    
    # Create an agent with the mock runtime
    agent = Agent(name="TestAgent", runtime=mock_runtime)
    
    # Run the agent
    result = agent.run({"message": "Hello"})
    
    # Check that the agent returned the expected result
    assert "response" in result
    assert "metadata" in result
    assert "Received: Hello" in result["response"]
    
    # Verify that the runtime's execute method was called with the right arguments
    mock_runtime.execute.assert_called_once()
    args, _ = mock_runtime.execute.call_args
    assert args[1]["message"] == "Hello"  # Second argument is the input data