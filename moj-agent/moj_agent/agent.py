"""Agent implementation for moj-agent."""

import os
import json
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAgent
from agentscaffold.agent import Agent as BaseAgent, AgentInput, AgentOutput


class MojAgentInput(AgentInput):
    """Input for MojAgent agent."""
    
    # Add custom input fields here
    pass


class MojAgentOutput(AgentOutput):
    """Output for MojAgent agent."""
    
    # Add custom output fields here
    pass


class MojAgentPydanticResult(BaseModel):
    """Result from Pydantic AI Agent."""
    
    message: str = Field(description="Response message")
    additional_info: Dict[str, Any] = Field(default_factory=dict, description="Additional information")


class Agent(BaseAgent):
    """MojAgent agent implementation."""
    
    name: str = "MojAgent"
    description: str = "A moj-agent agent"
    input_class = MojAgentInput
    output_class = MojAgentOutput
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize Pydantic AI agent
        self.pydantic_agent = PydanticAgent(
            "openai:gpt-4o",  # Can be configured via environment variables
            result_type=MojAgentPydanticResult,
            system_prompt=(
                "You are MojAgent, an AI assistant designed to help with "
                "moj-agent. Be helpful, concise, and accurate in your responses."
            )
        )
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and generate output.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Agent output
        """
        try:
            # Process with Pydantic AI
            result = await self.pydantic_agent.run(input_data['message'])
            
            # Return response
            return {
                "response": result.data.message,
                "metadata": result.data.additional_info
            }
        except Exception as e:
            # Handle any errors
            return {
                "response": f"Error: {str(e)}",
                "metadata": {"error": True}
            }