"""Tests for the agent module."""

import pytest
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
    agent = Agent(name="TestAgent")
    result = agent.run({"message": "Hello"})
    assert "response" in result
    assert "metadata" in result
    assert "Received: Hello" in result["response"]