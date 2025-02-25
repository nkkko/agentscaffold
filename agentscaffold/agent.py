"""Base agent implementation for AgentScaffold."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional, Callable, Type, ClassVar
import json


class AgentInput(BaseModel):
    """Base class for agent inputs."""
    
    message: str = Field(..., description="Input message for the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class AgentOutput(BaseModel):
    """Base class for agent outputs."""
    
    response: str = Field(..., description="Response from the agent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DaytonaRuntime:
    """Daytona runtime for agent execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def execute(self, agent_fn: Callable, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent function with the provided input data using Daytona.
        
        Args:
            agent_fn: Agent function to execute
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        # This is a mock implementation of Daytona execution
        # In a real implementation, this would use the Daytona SDK
        return agent_fn(input_data)


class Agent(BaseModel):
    """Base agent class for AgentScaffold."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    description: str = ""
    input_class: ClassVar[Type[AgentInput]] = AgentInput
    output_class: ClassVar[Type[AgentOutput]] = AgentOutput
    runtime: Optional[DaytonaRuntime] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.runtime is None:
            self.runtime = DaytonaRuntime()
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and generate output.
        
        This method should be overridden by derived agent classes.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Agent output
        """
        # Default implementation just echoes the input
        return {"response": f"Received: {input_data['message']}", "metadata": {}}
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the provided input data.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        # Validate input
        validated_input = self.input_class(**input_data).model_dump()
        
        # Execute agent using runtime
        result = self.runtime.execute(self.process, validated_input)
        
        # Validate output
        output = self.output_class(**result)
        
        return output.model_dump()