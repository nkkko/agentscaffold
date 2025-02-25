# moj-agent

An AI agent built with AgentScaffold that uses Pydantic AI and runs in a secure Daytona sandbox environment.

## Requirements

- Python 3.10 or higher
- Daytona sandbox environment (optional, falls back to direct execution if not available)
- OpenAI API key (or other LLM provider supported by Pydantic AI)

## Installation

```bash
# Install dependencies
pip install -e .
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
# OpenAI API key (required)
OPENAI_API_KEY=your-openai-api-key

# Daytona configuration (optional)
DAYTONA_API_KEY=your-daytona-api-key
DAYTONA_SERVER_URL=your-daytona-server-url
DAYTONA_TARGET=local
```

## Usage

Run the agent with:

```bash
python main.py
```

You can also use the agent in your own Python code:

```python
from moj_agent import Agent
import asyncio

async def example():
    agent = Agent()
    result = await agent.run({"message": "Hello, what can you do?"})
    print(result["response"])

asyncio.run(example())
```

## Security Features

This agent runs in a Daytona sandbox environment when available, providing:

- Isolated execution of AI-generated code
- File system isolation
- Process isolation
- Resource constraints

## Extending

To customize this agent, edit the `moj_agent/agent.py` file to modify the `process` method.