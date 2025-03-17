# AgentScaffold

A framework for scaffolding AI agents using PydanticAI, a Python agent framework, and Daytona, AI agent runtime.

## Installation

NOTE. Use development instructions.

```bash
# Install with pip
pip install agentscaffold

# Install with uv (recommended)
uv pip install agentscaffold
```

## Usage

Create a new agent:

```bash
# Using the full command
agentscaffold new my-agent
```

Run an agent:

```bash
# Run agent in Daytona remote sandbox
agentscaffold run

# Run locally without Daytona
agentscaffold run --local

# Run with a custom message
agentscaffold run --message "Hello, agent!"

# Using the alias
as run
```

## Configuring Daytona

AgentScaffold uses the Daytona SDK for remote execution of AI agents. You'll need to configure your Daytona credentials:

1. Obtain a Daytona API key from the Daytona platform or CLI:
   ```bash
   daytona api-key generate
   ```

2. Set environment variables in your `.env` file:
   ```
   DAYTONA_API_KEY=your-daytona-api-key
   DAYTONA_SERVER_URL=your-daytona-server-url
   DAYTONA_TARGET=us
   ```

## Features

- PydanticAI a Python agent framework designed to build production grade applications with Generative AI
- Daytona as generative AI agent and workflow execution runtime
- UV an extremely fast Python package and project manager
- Command-line interface for easy agent scaffolding
- Remote execution in isolated Daytona workspaces

## Project Structure

When you create a new agent, the following structure is generated:

```
my-agent/
├── my_agent/
│   ├── __init__.py
│   └── agent.py
├── main.py
├── pyproject.toml
├── README.md
├── .env
└── requirements.in
```

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/AgentScaffold.git
cd AgentScaffold

# Install development dependencies
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT