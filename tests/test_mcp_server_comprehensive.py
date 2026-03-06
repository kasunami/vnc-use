"""Comprehensive tests for mcp_server.py module."""

import io
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.vnc_use.mcp_server import (
    _wrap_agent_for_streaming,
    credential_store,
    mcp,
)

mcp_any = cast(Any, mcp)


def get_execute_vnc_task_fn():
    """Get the underlying execute_vnc_task function."""
    tools = mcp_any._tool_manager._tools
    tool = tools.get("execute_vnc_task")
    return tool.fn if tool else None


# Get the function at module level for use in tests
execute_vnc_task = get_execute_vnc_task_fn()


def create_test_png(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG image for testing."""
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestMCPServerInitialization:
    """Tests for MCP server initialization."""

    def test_mcp_server_name(self):
        """MCP server should have correct name."""
        assert mcp.name == "VNC Computer Use Agent"

    def test_credential_store_initialized(self):
        """Credential store should be initialized."""
        assert credential_store is not None

    def test_execute_vnc_task_registered(self):
        """execute_vnc_task should be registered as a tool."""
        tools = mcp_any._tool_manager._tools
        assert "execute_vnc_task" in tools

    def test_execute_vnc_task_fn_accessible(self):
        """Should be able to access the underlying function."""
        fn = get_execute_vnc_task_fn()
        assert fn is not None
        assert callable(fn)


class TestExecuteVncTaskNoCredentials:
    """Tests for execute_vnc_task when credentials not found."""

    @pytest.mark.asyncio
    async def test_returns_error_when_no_credentials(self):
        """Should return error when no credentials found."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        with patch.object(credential_store, "get", return_value=None):
            result = await execute_vnc_task(
                hostname="unknown-host",
                task="Click button",
                ctx=mock_ctx,
            )

        assert result["success"] is False
        assert "No credentials found" in result["error"]
        assert result["run_id"] is None

    @pytest.mark.asyncio
    async def test_logs_credential_lookup_error(self):
        """Should log info about credential lookup failure."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        with patch.object(credential_store, "get", return_value=None):
            await execute_vnc_task(
                hostname="missing-host",
                task="Task",
                ctx=mock_ctx,
            )

        # Should have called ctx.info with error message
        calls = mock_ctx.info.call_args_list
        error_messages = [str(call) for call in calls]
        assert any("Error" in msg for msg in error_messages)


class TestExecuteVncTaskWithCredentials:
    """Tests for execute_vnc_task with valid credentials."""

    @pytest.mark.asyncio
    async def test_creates_agent_with_credentials(self):
        """Should create agent with credentials from store."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "testhost::5901"
        mock_creds.password = "secret"

        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "success": True,
            "run_id": "test_run",
            "run_dir": "/tmp/runs",
            "final_state": {"step": 5},
        }

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", return_value=mock_agent):
                result = await execute_vnc_task(
                    hostname="testhost",
                    task="Click button",
                    ctx=mock_ctx,
                )

        assert result["success"] is True
        assert result["run_id"] == "test_run"

    @pytest.mark.asyncio
    async def test_uses_step_limit_and_timeout(self):
        """Should pass step_limit and timeout to agent."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        mock_agent = MagicMock()
        mock_agent.run.return_value = {"success": True, "final_state": {"step": 0}}

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent") as mock_agent_cls:
                mock_agent_cls.return_value = mock_agent

                await execute_vnc_task(
                    hostname="host",
                    task="Task",
                    step_limit=100,
                    timeout=600,
                    ctx=mock_ctx,
                )

                call_kwargs = mock_agent_cls.call_args.kwargs
                assert call_kwargs["step_limit"] == 100
                assert call_kwargs["seconds_timeout"] == 600

    @pytest.mark.asyncio
    async def test_reports_success(self):
        """Should report success via context."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "success": True,
            "final_state": {"step": 10},
            "run_id": "run123",
            "run_dir": "/tmp/runs/run123",
        }

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", return_value=mock_agent):
                await execute_vnc_task(
                    hostname="host",
                    task="Task",
                    ctx=mock_ctx,
                )

        # Check that success was reported
        calls = [str(c) for c in mock_ctx.info.call_args_list]
        assert any("completed successfully" in c for c in calls)

    @pytest.mark.asyncio
    async def test_reports_failure(self):
        """Should report failure via context."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "success": False,
            "error": "Timeout reached",
            "final_state": {"step": 5},
        }

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", return_value=mock_agent):
                result = await execute_vnc_task(
                    hostname="host",
                    task="Task",
                    ctx=mock_ctx,
                )

        assert result["success"] is False
        assert "Timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_handles_agent_exception(self):
        """Should handle exceptions during agent execution."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent") as mock_agent_cls:
                mock_agent_cls.side_effect = Exception("Agent creation failed")

                result = await execute_vnc_task(
                    hostname="host",
                    task="Task",
                    ctx=mock_ctx,
                )

        assert result["success"] is False
        assert "Agent creation failed" in result["error"]


class TestExecuteVncTaskWithoutContext:
    """Tests for execute_vnc_task without MCP context."""

    @pytest.mark.asyncio
    async def test_works_without_context(self):
        """Should work without MCP context."""
        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "success": True,
            "final_state": {"step": 0},
            "run_id": "run1",
            "run_dir": "/tmp/runs",
        }

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", return_value=mock_agent):
                result = await execute_vnc_task(
                    hostname="host",
                    task="Task",
                    ctx=None,
                )

        assert result["success"] is True


class TestExecuteVncTaskHitlCallback:
    """Tests for HITL callback in execute_vnc_task."""

    @pytest.mark.asyncio
    async def test_hitl_callback_with_elicitation_accept(self):
        """Should approve when user accepts via elicitation."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_result = MagicMock()
        mock_result.action = "accept"
        mock_ctx.elicit = AsyncMock(return_value=mock_result)

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        captured_callback = None

        def capture_agent(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("hitl_callback")
            mock_agent = MagicMock()
            mock_agent.run.return_value = {"success": True, "final_state": {}}
            return mock_agent

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", side_effect=capture_agent):
                await execute_vnc_task(hostname="host", task="Task", ctx=mock_ctx)

        # Test the captured callback
        if captured_callback:
            result = await captured_callback({"reason": "Test"}, [{"name": "click"}])
            assert result is True

    @pytest.mark.asyncio
    async def test_hitl_callback_with_elicitation_decline(self):
        """Should deny when user declines via elicitation."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_result = MagicMock()
        mock_result.action = "decline"
        mock_ctx.elicit = AsyncMock(return_value=mock_result)

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        captured_callback = None

        def capture_agent(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("hitl_callback")
            mock_agent = MagicMock()
            mock_agent.run.return_value = {"success": True, "final_state": {}}
            return mock_agent

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", side_effect=capture_agent):
                await execute_vnc_task(hostname="host", task="Task", ctx=mock_ctx)

        # Test the captured callback
        if captured_callback:
            result = await captured_callback({"reason": "Test"}, [{"name": "click"}])
            assert result is False


class TestWrapAgentForStreaming:
    """Tests for _wrap_agent_for_streaming function."""

    def test_wrap_preserves_original_agent(self):
        """Should return the same agent instance."""
        mock_agent = MagicMock()
        mock_agent._propose_node = MagicMock()
        mock_agent._act_node = MagicMock()

        mock_ctx = AsyncMock()

        result = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        assert result is mock_agent

    def test_wrap_replaces_propose_node(self):
        """Should replace _propose_node method."""
        original_propose = MagicMock()
        mock_agent = MagicMock()
        mock_agent._propose_node = original_propose
        mock_agent._act_node = MagicMock()

        mock_ctx = AsyncMock()

        result = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        # The method should have been replaced
        assert result._propose_node is not original_propose

    def test_wrap_replaces_act_node(self):
        """Should replace _act_node method."""
        original_act = MagicMock()
        mock_agent = MagicMock()
        mock_agent._propose_node = MagicMock()
        mock_agent._act_node = original_act

        mock_ctx = AsyncMock()

        result = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        # The method should have been replaced
        assert result._act_node is not original_act


class TestWrappedProposeNode:
    """Tests for the wrapped propose node."""

    def test_wrapped_propose_calls_original(self):
        """Wrapped propose should call original."""
        original_propose = MagicMock(return_value={"observation": "Test"})
        mock_agent = MagicMock()
        mock_agent._propose_node = original_propose
        mock_agent._act_node = MagicMock()

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.report_progress = AsyncMock()

        wrapped = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        state = {"step": 1}
        result = wrapped._propose_node(cast(Any, state))

        # Original should have been called
        original_propose.assert_called_once_with(state)
        assert result == {"observation": "Test"}


class TestWrappedActNode:
    """Tests for the wrapped act node."""

    def test_wrapped_act_calls_original(self):
        """Wrapped act should call original."""
        original_act = MagicMock(return_value={"step_logs": []})
        mock_agent = MagicMock()
        mock_agent._propose_node = MagicMock()
        mock_agent._act_node = original_act

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        wrapped = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        state = {"step": 1}
        result = wrapped._act_node(cast(Any, state))

        # Original should have been called
        original_act.assert_called_once_with(state)
        assert result == {"step_logs": []}


class TestModelProviderSelection:
    """Tests for model provider selection."""

    @pytest.mark.asyncio
    async def test_uses_gemini_by_default(self, monkeypatch):
        """Should use Gemini provider by default."""
        monkeypatch.delenv("MODEL_PROVIDER", raising=False)

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent") as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.run.return_value = {"success": True, "final_state": {}}
                mock_agent_cls.return_value = mock_agent

                await execute_vnc_task(hostname="host", task="Task", ctx=mock_ctx)

                call_kwargs = mock_agent_cls.call_args.kwargs
                assert call_kwargs["model_provider"] == "gemini"

    @pytest.mark.asyncio
    async def test_uses_anthropic_from_env(self, monkeypatch):
        """Should use Anthropic provider from environment."""
        monkeypatch.setenv("MODEL_PROVIDER", "anthropic")

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent") as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.run.return_value = {"success": True, "final_state": {}}
                mock_agent_cls.return_value = mock_agent

                await execute_vnc_task(hostname="host", task="Task", ctx=mock_ctx)

                call_kwargs = mock_agent_cls.call_args.kwargs
                assert call_kwargs["model_provider"] == "anthropic"


class TestExecuteVncTaskDefaultValues:
    """Tests for default parameter values."""

    @pytest.mark.asyncio
    async def test_default_step_limit(self):
        """Should use default step_limit of 40."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent") as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.run.return_value = {"success": True, "final_state": {}}
                mock_agent_cls.return_value = mock_agent

                await execute_vnc_task(hostname="host", task="Task", ctx=mock_ctx)

                call_kwargs = mock_agent_cls.call_args.kwargs
                assert call_kwargs["step_limit"] == 40

    @pytest.mark.asyncio
    async def test_default_timeout(self):
        """Should use default timeout of 300."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent") as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.run.return_value = {"success": True, "final_state": {}}
                mock_agent_cls.return_value = mock_agent

                await execute_vnc_task(hostname="host", task="Task", ctx=mock_ctx)

                call_kwargs = mock_agent_cls.call_args.kwargs
                assert call_kwargs["seconds_timeout"] == 300


class TestHitlCallbackEdgeCases:
    """Tests for HITL callback edge cases."""

    @pytest.mark.asyncio
    async def test_hitl_callback_auto_approves_without_context(self):
        """Should auto-approve when no MCP context."""
        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        captured_callback = None

        def capture_agent(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("hitl_callback")
            mock_agent = MagicMock()
            mock_agent.run.return_value = {"success": True, "final_state": {}}
            return mock_agent

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", side_effect=capture_agent):
                # Call with ctx=None to skip HITL callback setup
                await execute_vnc_task(hostname="host", task="Task", ctx=None)

        # The callback should be None when ctx is None
        assert captured_callback is None

    @pytest.mark.asyncio
    async def test_hitl_callback_with_elicitation_cancel(self):
        """Should deny when user cancels via elicitation."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_result = MagicMock()
        mock_result.action = "cancel"  # Not accept or decline
        mock_ctx.elicit = AsyncMock(return_value=mock_result)

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        captured_callback = None

        def capture_agent(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("hitl_callback")
            mock_agent = MagicMock()
            mock_agent.run.return_value = {"success": True, "final_state": {}}
            return mock_agent

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", side_effect=capture_agent):
                await execute_vnc_task(hostname="host", task="Task", ctx=mock_ctx)

        # Test the captured callback - should return False for cancel
        if captured_callback:
            result = await captured_callback({"reason": "Test"}, [{"name": "click"}])
            assert result is False

    @pytest.mark.asyncio
    async def test_hitl_callback_with_elicitation_exception(self):
        """Should deny when elicitation raises exception."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.elicit = AsyncMock(side_effect=Exception("Elicitation failed"))

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        captured_callback = None

        def capture_agent(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("hitl_callback")
            mock_agent = MagicMock()
            mock_agent.run.return_value = {"success": True, "final_state": {}}
            return mock_agent

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", side_effect=capture_agent):
                await execute_vnc_task(hostname="host", task="Task", ctx=mock_ctx)

        # Test the captured callback - should return False on exception
        if captured_callback:
            result = await captured_callback({"reason": "Test"}, [{"name": "click"}])
            assert result is False

    @pytest.mark.asyncio
    async def test_hitl_callback_with_none_safety_decision(self):
        """Should handle None safety decision."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_result = MagicMock()
        mock_result.action = "accept"
        mock_ctx.elicit = AsyncMock(return_value=mock_result)

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        captured_callback = None

        def capture_agent(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("hitl_callback")
            mock_agent = MagicMock()
            mock_agent.run.return_value = {"success": True, "final_state": {}}
            return mock_agent

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", side_effect=capture_agent):
                await execute_vnc_task(hostname="host", task="Task", ctx=mock_ctx)

        # Test the captured callback with None safety_decision
        if captured_callback:
            result = await captured_callback(
                None,  # None safety decision
                [{"name": "click"}],
            )
            assert result is True


class TestStreamingWrapperEdgeCases:
    """Tests for streaming wrapper edge cases."""

    def test_wrapped_propose_reports_progress(self):
        """Wrapped propose should report progress."""
        original_propose = MagicMock(
            return_value={
                "observation": "Test observation that is long enough to test truncation " * 10,
                "proposed_actions": [
                    {"name": "click"},
                    {"name": "type"},
                    {"name": "scroll"},
                    {"name": "key"},
                ],
            }
        )
        mock_agent = MagicMock()
        mock_agent._propose_node = original_propose
        mock_agent._act_node = MagicMock()

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.report_progress = AsyncMock()

        wrapped = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        state = {"step": 5}
        result = wrapped._propose_node(cast(Any, state))

        # Original should have been called
        original_propose.assert_called_once_with(state)
        assert "observation" in result

    def test_wrapped_propose_handles_empty_observation(self):
        """Wrapped propose should handle empty observation."""
        original_propose = MagicMock(return_value={"observation": "", "proposed_actions": []})
        mock_agent = MagicMock()
        mock_agent._propose_node = original_propose
        mock_agent._act_node = MagicMock()

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.report_progress = AsyncMock()

        wrapped = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        state = {"step": 1}
        result = wrapped._propose_node(cast(Any, state))

        # Should not raise, original should have been called
        original_propose.assert_called_once_with(state)
        assert result["observation"] == ""

    def test_wrapped_act_handles_screenshot(self):
        """Wrapped act should handle screenshot streaming."""
        test_png = create_test_png(100, 100)
        original_act = MagicMock(
            return_value={
                "step_logs": [
                    {
                        "executed_action": {"name": "click", "args": {"x": 100, "y": 200}},
                        "result": "Success",
                    }
                ],
                "last_screenshot_png": test_png,
            }
        )
        mock_agent = MagicMock()
        mock_agent._propose_node = MagicMock()
        mock_agent._act_node = original_act

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        wrapped = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        state = {"step": 1}
        result = wrapped._act_node(cast(Any, state))

        # Original should have been called
        original_act.assert_called_once_with(state)
        assert "step_logs" in result

    def test_wrapped_act_handles_no_screenshot(self):
        """Wrapped act should handle no screenshot."""
        original_act = MagicMock(return_value={"step_logs": [], "last_screenshot_png": None})
        mock_agent = MagicMock()
        mock_agent._propose_node = MagicMock()
        mock_agent._act_node = original_act

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        wrapped = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        state = {"step": 1}
        wrapped._act_node(cast(Any, state))

        # Should not raise
        original_act.assert_called_once_with(state)

    def test_wrapped_act_handles_step_logs_with_failure(self):
        """Wrapped act should handle failed action in step logs."""
        original_act = MagicMock(
            return_value={
                "step_logs": [
                    {
                        "executed_action": {"name": "click", "args": {}},
                        "result": "Failed: Element not found",
                    }
                ],
                "last_screenshot_png": None,
            }
        )
        mock_agent = MagicMock()
        mock_agent._propose_node = MagicMock()
        mock_agent._act_node = original_act

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        wrapped = _wrap_agent_for_streaming(mock_agent, mock_ctx, 40)

        state = {"step": 1}
        wrapped._act_node(cast(Any, state))

        # Original should have been called
        original_act.assert_called_once_with(state)


class TestExecuteVncTaskResultStructure:
    """Tests for result structure."""

    @pytest.mark.asyncio
    async def test_successful_result_has_all_fields(self):
        """Successful result should have all expected fields."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        mock_creds = MagicMock()
        mock_creds.server = "host::5901"
        mock_creds.password = "pass"

        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "success": True,
            "final_state": {"step": 5, "observation": "Done"},
            "run_id": "run123",
            "run_dir": "/tmp/runs/run123",
        }

        with patch.object(credential_store, "get", return_value=mock_creds):
            with patch("src.vnc_use.mcp_server.VncUseAgent", return_value=mock_agent):
                result = await execute_vnc_task(
                    hostname="host",
                    task="Task",
                    ctx=mock_ctx,
                )

        assert "success" in result
        assert "run_id" in result
        assert "run_dir" in result
        assert "steps" in result

    @pytest.mark.asyncio
    async def test_error_result_has_error_field(self):
        """Error result should have error field."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        with patch.object(credential_store, "get", return_value=None):
            result = await execute_vnc_task(
                hostname="host",
                task="Task",
                ctx=mock_ctx,
            )

        assert result["success"] is False
        assert "error" in result
