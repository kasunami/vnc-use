"""Tests for MCP server functionality."""

import pytest

from vnc_use.mcp_server import mcp


def test_mcp_server_initialization():
    """Test that MCP server is properly initialized."""
    assert mcp.name == "VNC Computer Use Agent"


def test_execute_vnc_task_tool_exists():
    """Test that execute_vnc_task tool is registered."""
    # Get registered tools
    tools = mcp._tool_manager._tools
    assert "execute_vnc_task" in tools


def test_execute_vnc_task_tool_registration():
    """Test execute_vnc_task is properly registered as a FunctionTool."""
    tools = mcp._tool_manager._tools
    tool = tools.get("execute_vnc_task")

    assert tool is not None
    assert tool.name == "execute_vnc_task"
    assert "VNC desktop" in tool.description
    assert "hostname" in tool.description or "vnc_server" in tool.description


@pytest.mark.external
@pytest.mark.asyncio
async def test_execute_vnc_task_without_vnc():
    """Test execute_vnc_task error handling when VNC is unavailable.

    Requires: VNC server (or will fail to connect)
    """
    # Get the underlying function from the tool
    tools = mcp._tool_manager._tools
    tool = tools.get("execute_vnc_task")

    # Skip if we can't get the function
    if not hasattr(tool, "fn"):
        pytest.skip("Cannot access underlying function")

    result = await tool.fn(
        vnc_server="nonexistent::9999",
        task="Test task",
        vnc_password=None,
        step_limit=5,
        timeout=10,
        ctx=None,
    )

    # Should return error result
    assert result["success"] is False
    assert result["error"] is not None


@pytest.mark.external
@pytest.mark.asyncio
async def test_execute_vnc_task_parameter_validation():
    """Test parameter types are correct.

    Requires: VNC server and GOOGLE_API_KEY
    """
    # Get the underlying function from the tool
    tools = mcp._tool_manager._tools
    tool = tools.get("execute_vnc_task")

    # Skip if we can't get the function
    if not hasattr(tool, "fn"):
        pytest.skip("Cannot access underlying function")

    result = await tool.fn(
        vnc_server="localhost::5901",
        task="Open browser",
        vnc_password="test",
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


def test_mcp_tool_manager():
    """Test tool manager is properly configured."""
    assert hasattr(mcp, "_tool_manager")
    assert len(mcp._tool_manager._tools) > 0
