# AgentScaffold

A framework for scaffolding AI agents using Pydantic and Daytona runtime.

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
# Using the full command
agentscaffold run

# Using the alias
as run
```

## Features

- Pydantic for agent schema creation and validation
- Daytona as agent execution runtime
- UV compatible for fast dependency management
- Command-line interface for easy agent scaffolding

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