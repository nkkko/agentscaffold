# AgentScaffold

A framework for scaffolding AI agents with modularity and flexibility. It focuses on providing a standardized way to scaffold and compose agents using popular providers for LLMs, search, memory, logging, and MCP (Model Context Protocol).

## Installation

NOTE. Use development instructions.

```bash
# Install with pip
pip install agentscaffold

# Install with uv (recommended)
uv pip install agentscaffold
```

## Quick Start

Create a new agent:

```bash
# Using the full command
agentscaffold new my-agent
cd my-agent
```

Run an agent:

```bash
# Run agent in Daytona remote sandbox
agentscaffold run

# Run with a custom message
agentscaffold run --message "Hello, agent!"

# Using the alias
as run
```

## MCP Integration

AgentScaffold supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), making it easy to integrate with tools like Claude Code and other MCP servers.

Add a command-based MCP server:

```bash
# Add a stdio-based MCP server
agentscaffold mcp add daytona "python -m daytona_server" -e API_KEY=your_api_key
```

Add an HTTP-based MCP server:

```bash
# Add an HTTP-based MCP server
agentscaffold mcp add-http claude-code https://claude.ai/api/claude-code --api-key your_api_key
```

List configured servers:

```bash
# List all MCP servers across all scopes
agentscaffold mcp list
```

Get details about a specific server:

```bash
# Get details about a specific server
agentscaffold mcp get daytona
```

Test a server connection:

```bash
# Test connectivity to an MCP server
agentscaffold mcp test claude-code
```

Remove a server:

```bash
# Remove an MCP server
agentscaffold mcp remove daytona
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
   DAYTONA_API_URL=your-daytona-server-url
   DAYTONA_TARGET=us
   ```

## Features

- **PydanticAI**: Python agent framework designed to build production-grade AI applications
- **Daytona**: Secure AI agent and workflow execution runtime
- **Provider Flexibility**: Support for various LLM, search, memory, and logging providers
- **MCP Integration**: Full support for the Model Context Protocol
- **UV Integration**: Ultra-fast Python package and virtual environment management
- **Command-line interface**: Easy agent scaffolding and management
- **Remote execution**: Isolated Daytona workspaces for secure agent execution

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