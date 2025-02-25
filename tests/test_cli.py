"""Tests for the CLI module."""

from typer.testing import CliRunner
import pytest
from agentscaffold.cli import app


runner = CliRunner()


def test_version():
    """Test the version command."""
    from agentscaffold import __version__
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"AgentScaffold v{__version__}" in result.stdout