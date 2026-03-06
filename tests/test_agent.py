"""Tests for agent.py module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.vnc_use.agent import VncUseAgent
from src.vnc_use.types import CUAState


@pytest.fixture
def mock_planner():
    """Create a mock planner."""
    planner = MagicMock()
    planner.generate_stateless.return_value = MagicMock()
    planner.extract_text.return_value = "I see a button"
    planner.extract_function_calls.return_value = []
    planner.extract_safety_decision.return_value = None
    return planner


@pytest.fixture
def mock_vnc():
    """Create a mock VNC controller."""
    vnc = MagicMock()
    vnc.screenshot_png.return_value = b"\x89PNG\r\n\x1a\n"
    vnc.execute_action.return_value = MagicMock(
        success=True, error=None, screenshot_png=b"\x89PNG\r\n\x1a\n"
    )
    return vnc


class TestVncUseAgentInit:
    """Tests for VncUseAgent initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default parameters."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")

                assert agent.vnc_server == "localhost::5901"
                assert agent.vnc_password is None
                assert agent.step_limit == 40
                assert agent.hitl_mode is True

    def test_init_with_custom_vnc_server(self):
        """Should use custom VNC server."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(
                    vnc_server="remote::5902", vnc_password="secret", api_key="test_key"
                )

                assert agent.vnc_server == "remote::5902"
                assert agent.vnc_password == "secret"

    def test_init_with_custom_step_limit(self):
        """Should use custom step limit."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(step_limit=100, api_key="test_key")

                assert agent.step_limit == 100

    def test_init_with_custom_timeout(self):
        """Should use custom timeout."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(seconds_timeout=600, api_key="test_key")

                assert agent.seconds_timeout == 600

    def test_init_with_hitl_disabled(self):
        """Should disable HITL mode."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(hitl_mode=False, api_key="test_key")

                assert agent.hitl_mode is False

    def test_init_with_hitl_callback(self):
        """Should set HITL callback."""

        async def callback(safety, pending):
            return True

        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(hitl_callback=callback, api_key="test_key")

                assert agent.hitl_callback is callback

    def test_init_gemini_provider(self):
        """Should use Gemini planner by default."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(model_provider="gemini", api_key="test_key")

                assert "Gemini" in type(agent.planner).__name__

    def test_init_anthropic_provider(self):
        """Should use Anthropic planner when specified."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
                mock_llm = MagicMock()
                mock_llm.bind_tools.return_value = MagicMock()
                mock_chat.return_value = mock_llm

                agent = VncUseAgent(model_provider="anthropic", api_key="test_key")

                assert "Anthropic" in type(agent.planner).__name__

    def test_init_unknown_provider_raises(self):
        """Should raise for unknown provider."""
        with patch("src.vnc_use.agent.VNCController"):
            with pytest.raises(ValueError, match="Unknown model_provider"):
                VncUseAgent(model_provider="unknown", api_key="test_key")

    def test_init_default_excluded_actions(self):
        """Should set default excluded actions."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")

                # Check that browser-specific actions are excluded by default
                assert "open_web_browser" in agent.planner.excluded_actions
                assert "navigate" in agent.planner.excluded_actions

    def test_init_custom_excluded_actions(self):
        """Should use custom excluded actions."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(excluded_actions=["scroll_at"], api_key="test_key")

                assert agent.planner.excluded_actions == ["scroll_at"]

    def test_init_builds_graph(self):
        """Should build LangGraph state machine."""
        with patch("src.vnc_use.agent.VNCController"):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")

                assert agent.graph is not None


class TestVncUseAgentProposeNode:
    """Tests for _propose_node method."""

    def test_propose_returns_done_on_step_limit(self, mock_planner, mock_vnc):
        """Should return done when step limit reached."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(step_limit=5, api_key="test_key")
                agent.planner = mock_planner

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 5,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._propose_node(state)

                assert result["done"] is True
                assert "Step limit" in result["error"]

    def test_propose_returns_done_on_timeout(self, mock_planner, mock_vnc):
        """Should return done when timeout reached."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(seconds_timeout=10, api_key="test_key")
                agent.planner = mock_planner

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 0,
                    "done": False,
                    "safety": None,
                    "start_time": time.time() - 20,  # 20 seconds ago
                    "error": None,
                }

                result = agent._propose_node(state)

                assert result["done"] is True
                assert "Timeout" in result["error"]

    def test_propose_returns_done_no_screenshot(self, mock_planner, mock_vnc):
        """Should return done when no screenshot available."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.planner = mock_planner

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": None,
                    "last_observation": "",
                    "step": 0,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._propose_node(state)

                assert result["done"] is True
                assert "screenshot" in result["error"].lower()

    def test_propose_calls_planner(self, mock_planner, mock_vnc):
        """Should call planner to generate actions."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.planner = mock_planner

                state: CUAState = {
                    "task": "Click button",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 0,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                agent._propose_node(state)

                mock_planner.generate_stateless.assert_called_once()

    def test_propose_extracts_function_calls(self, mock_planner, mock_vnc):
        """Should extract function calls from planner response."""
        mock_planner.extract_function_calls.return_value = [
            {"name": "click_at", "args": {"x": 100, "y": 200}}
        ]

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.planner = mock_planner

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 0,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._propose_node(state)

                assert len(result["pending_calls"]) == 1
                assert result["pending_calls"][0]["name"] == "click_at"

    def test_propose_returns_done_when_no_calls(self, mock_planner, mock_vnc):
        """Should return done when no function calls."""
        mock_planner.extract_function_calls.return_value = []

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.planner = mock_planner

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 0,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._propose_node(state)

                assert result["done"] is True

    def test_propose_blocks_on_safety_decision(self, mock_planner, mock_vnc):
        """Should block when safety decision indicates block."""
        mock_planner.extract_function_calls.return_value = [{"name": "click", "args": {}}]
        mock_planner.extract_safety_decision.return_value = {
            "action": "block",
            "reason": "Dangerous operation",
        }

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.planner = mock_planner

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 0,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._propose_node(state)

                assert result["done"] is True
                assert "Blocked by safety" in result["error"]

    def test_propose_handles_planner_exception(self, mock_planner, mock_vnc):
        """Should handle planner exceptions gracefully."""
        mock_planner.generate_stateless.side_effect = Exception("API error")

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.planner = mock_planner

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 0,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._propose_node(state)

                assert result["done"] is True
                assert "API error" in result["error"]


class TestVncUseAgentActNode:
    """Tests for _act_node method."""

    def test_act_returns_done_when_no_pending(self, mock_vnc):
        """Should return done when no pending calls."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 0,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._act_node(state)

                assert result["done"] is True

    def test_act_executes_action(self, mock_vnc):
        """Should execute pending action via VNC."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.vnc = mock_vnc

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click_at", "args": {"x": 100, "y": 200}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                agent._act_node(state)

                mock_vnc.execute_action.assert_called_once_with("click_at", {"x": 100, "y": 200})

    def test_act_updates_action_history(self, mock_vnc):
        """Should update action history."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.vnc = mock_vnc

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click_at", "args": {"x": 100, "y": 200}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._act_node(state)

                assert len(result["action_history"]) == 1
                assert "click_at" in result["action_history"][0]

    def test_act_removes_executed_call(self, mock_vnc):
        """Should remove executed call from pending."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.vnc = mock_vnc

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [
                        {"name": "click_at", "args": {}},
                        {"name": "type_text_at", "args": {}},
                    ],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._act_node(state)

                assert len(result["pending_calls"]) == 1
                assert result["pending_calls"][0]["name"] == "type_text_at"

    def test_act_handles_action_error(self, mock_vnc):
        """Should handle action errors gracefully."""
        mock_vnc.execute_action.return_value = MagicMock(
            success=False, error="Click failed", screenshot_png=b"png"
        )

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.vnc = mock_vnc

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click_at", "args": {}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._act_node(state)

                # Should continue despite error
                assert "Error" in result["action_history"][0]

    def test_act_handles_exception(self, mock_vnc):
        """Should handle exceptions during action execution."""
        mock_vnc.execute_action.side_effect = Exception("VNC error")
        mock_vnc.screenshot_png.return_value = b"png"

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.vnc = mock_vnc

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click_at", "args": {}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._act_node(state)

                assert "error" in result
                assert "VNC error" in result["error"]


class TestVncUseAgentHitlGateNode:
    """Tests for _hitl_gate_node method."""

    def test_hitl_with_callback_approved(self, mock_vnc):
        """Should use callback and continue when approved."""

        async def approve_callback(safety, pending):
            return True

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key", hitl_callback=approve_callback)

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click", "args": {}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": {"action": "confirm", "reason": "test"},
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._hitl_gate_node(state)

                assert result.get("done") is not True

    def test_hitl_with_callback_denied(self, mock_vnc):
        """Should return done when callback denies."""

        async def deny_callback(safety, pending):
            return False

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key", hitl_callback=deny_callback)

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click", "args": {}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": {"action": "confirm", "reason": "test"},
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._hitl_gate_node(state)

                assert result["done"] is True
                assert "denied" in result["error"].lower()

    def test_hitl_callback_exception(self, mock_vnc):
        """Should handle callback exceptions."""

        async def error_callback(safety, pending):
            raise Exception("Callback error")

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key", hitl_callback=error_callback)

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click", "args": {}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": {"action": "confirm", "reason": "test"},
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._hitl_gate_node(state)

                assert result["done"] is True
                assert "callback failed" in result["error"].lower()


class TestVncUseAgentRouting:
    """Tests for routing methods."""

    def test_route_after_propose_ends_when_done(self, mock_vnc):
        """Should end when done is True."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": True,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._route_after_propose(state)

                from langgraph.graph import END

                assert result == END

    def test_route_after_propose_to_hitl(self, mock_vnc):
        """Should route to HITL gate when confirmation required."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key", hitl_mode=True)

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click", "args": {}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": {"action": "require_confirmation", "reason": "test"},
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._route_after_propose(state)

                assert result == "hitl_gate"

    def test_route_after_propose_to_act(self, mock_vnc):
        """Should route to act when no HITL required."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key", hitl_mode=True)

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click", "args": {}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._route_after_propose(state)

                assert result == "act"

    def test_route_after_hitl_ends_when_done(self, mock_vnc):
        """Should end when done is True after HITL."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": True,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._route_after_hitl(state)

                from langgraph.graph import END

                assert result == END

    def test_route_after_hitl_to_act(self, mock_vnc):
        """Should route to act when not done."""
        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")

                state: CUAState = {
                    "task": "test",
                    "action_history": [],
                    "step_logs": [],
                    "pending_calls": [{"name": "click", "args": {}}],
                    "last_screenshot_png": b"png",
                    "last_observation": "",
                    "step": 1,
                    "done": False,
                    "safety": None,
                    "start_time": time.time(),
                    "error": None,
                }

                result = agent._route_after_hitl(state)

                assert result == "act"


class TestVncUseAgentRun:
    """Tests for run method."""

    def test_run_returns_error_on_vnc_connection_failure(self, mock_vnc):
        """Should return error when VNC connection fails."""
        mock_vnc.connect.side_effect = Exception("Connection refused")

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.vnc = mock_vnc

                result = agent.run("Click button")

                assert "error" in result
                assert "VNC connection failed" in result["error"]

    def test_run_returns_run_artifacts(self, mock_vnc, mock_planner):
        """Should return run artifacts."""
        mock_planner.extract_function_calls.return_value = []  # Complete immediately

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.vnc = mock_vnc
                agent.planner = mock_planner

                result = agent.run("Click button")

                assert "run_id" in result
                assert "run_dir" in result

    def test_run_disconnects_vnc_on_completion(self, mock_vnc, mock_planner):
        """Should disconnect VNC when done."""
        mock_planner.extract_function_calls.return_value = []

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.vnc = mock_vnc
                agent.planner = mock_planner

                agent.run("Click button")

                mock_vnc.disconnect.assert_called()

    def test_run_disconnects_vnc_on_error(self, mock_vnc, mock_planner):
        """Should disconnect VNC even on error."""
        mock_planner.generate_stateless.side_effect = Exception("API error")

        with patch("src.vnc_use.agent.VNCController", return_value=mock_vnc):
            with patch("src.vnc_use.planners.gemini.genai"):
                agent = VncUseAgent(api_key="test_key")
                agent.vnc = mock_vnc
                agent.planner = mock_planner

                agent.run("Click button")

                mock_vnc.disconnect.assert_called()
