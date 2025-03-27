# test-agent

An AI agent built with AgentScaffold that uses Pydantic AI and runs in a secure Daytona sandbox environment.

## Features


- Uses OpenAI API for language model capabilities




- Integrates Brave Search API for web search capabilities





- Uses ChromaDB local vector database for memory and context retrieval





- Includes LogFire integration for observability and logging



## Requirements

- Python 3.10 or higher
- Daytona sandbox environment (optional, falls back to direct execution if not available)

- OpenAI API key




- Brave Search API key





- Local directory for ChromaDB storage





- LogFire account and API key



## Installation

```bash
# Install dependencies
pip install -e .
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```

# OpenAI API (required)
OPENAI_API_KEY=your-openai-api-key




# Brave Search API (required for search)
BRAVE_API_KEY=your-brave-api-key





# ChromaDB (optional, defaults to ./chroma_db)
CHROMA_DB_PATH=./chroma_db





# LogFire (required for logging)
LOGFIRE_API_KEY=your-logfire-api-key



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

Run the agent in interactive mode:

```bash
python main.py --interactive
```


Include a search query:

```bash
python main.py --message "Tell me about climate change" --search "latest climate change research"
```



Retrieve context from memory:

```bash
python main.py --message "What did we discuss before?" --context
```


You can also use the agent in your own Python code:

```python
from test_agent import Agent
import asyncio

async def example():
    agent = Agent()
    result = await agent.run({
        "message": "Hello, what can you do?",
        
        "search_query": "AI agents",  # Optional
        
        
        "retrieve_context": True,  # Optional
        "context_query": "agent capabilities",  # Optional
        
    })
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

To customize this agent, edit the `test_agent/agent.py` file to modify the `process` method. 


### Adding Search Providers

To modify or enhance search capabilities, edit the `_init_search` and `search_web` methods in the agent class.



### Enhancing Memory Capabilities

To modify memory and context functionality, edit the `_init_memory` and `retrieve_from_memory` methods.



### Configuring Observability

To enhance logging and observability, modify the `_init_logging` and `log_event` methods.
