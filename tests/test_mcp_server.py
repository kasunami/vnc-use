"""Tests for MCP server functionality."""

import pytest

from vnc_use.mcp_server import mcp


def test_mcp_server_initialization():
    """Test that MCP server is properly initialized."""
    assert mcp.name == "VNC Computer Use Agent"


@pytest.mark.asyncio
async def test_execute_vnc_task_tool_exists():
    """Test that execute_vnc_task tool is registered."""
    tools = {tool.name for tool in await mcp.list_tools()}
    assert "execute_vnc_task" in tools
    assert "execute_vnc_policy_task" in tools
    assert "execute_vnc_action" in tools


@pytest.mark.asyncio
async def test_execute_vnc_task_tool_registration():
    """Test execute_vnc_task is properly registered as a FunctionTool."""
    tool = await mcp.get_tool("execute_vnc_task")

    assert tool is not None
    assert tool.name == "execute_vnc_task"
    assert "VNC desktop" in tool.description
    assert "hostname" in tool.description or "vnc_server" in tool.description

    policy_tool = await mcp.get_tool("execute_vnc_policy_task")
    assert policy_tool is not None
    assert "guardrail" in policy_tool.description.lower()

    action_tool = await mcp.get_tool("execute_vnc_action")
    assert action_tool is not None
    assert "deterministic" in action_tool.description.lower()


@pytest.mark.external
@pytest.mark.asyncio
async def test_execute_vnc_task_without_vnc():
    """Test execute_vnc_task error handling when credentials not found.

    Tests the credential lookup failure path.
    """
    tool = await mcp.get_tool("execute_vnc_task")

    # Skip if we can't get the function
    if not hasattr(tool, "fn"):
        pytest.skip("Cannot access underlying function")

    result = await tool.fn(
        hostname="nonexistent-host",
        task="Test task",
        step_limit=5,
        timeout=10,
        ctx=None,
    )

    # Should return error result (no credentials found)
    assert result["success"] is False
    assert result["error"] is not None
    assert "No credentials found" in result["error"]


@pytest.mark.external
@pytest.mark.asyncio
async def test_execute_vnc_task_parameter_validation():
    """Test that execute_vnc_task returns correct structure.

    Tests the return value structure when no credentials found.
    """
    tool = await mcp.get_tool("execute_vnc_task")

    # Skip if we can't get the function
    if not hasattr(tool, "fn"):
        pytest.skip("Cannot access underlying function")

    result = await tool.fn(
        hostname="test-host",
        task="Open browser",
        step_limit=10,
        timeout=60,
        ctx=None,
    )

    # Should return a dict with expected keys
    assert isinstance(result, dict)
    assert "success" in result
    assert "run_id" in result
    assert "run_dir" in result
    assert "steps" in result
    assert "error" in result


def test_mcp_server_name():
    """Test MCP server has correct name."""
    assert mcp.name == "VNC Computer Use Agent"


@pytest.mark.asyncio
async def test_mcp_tool_manager():
    """Test tool manager is properly configured."""
    assert await mcp.list_tools()
