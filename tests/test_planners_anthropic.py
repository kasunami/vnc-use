"""Tests for planners/anthropic.py module."""

import io
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from src.vnc_use.planners.anthropic import (
    DEFAULT_MODEL,
    AnthropicPlanner,
)


def create_test_png(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG image for testing."""
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestAnthropicPlannerInit:
    """Tests for AnthropicPlanner initialization."""

    def test_init_with_api_key(self):
        """Should initialize with provided API key."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="test_key")

            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["api_key"] == "test_key"
            assert planner.excluded_actions == []

    def test_init_with_env_api_key(self, monkeypatch):
        """Should use ANTHROPIC_API_KEY env var."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env_api_key")

        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            _planner = AnthropicPlanner()  # noqa: F841 - creating to test init

            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["api_key"] == "env_api_key"

    def test_init_raises_without_api_key(self, monkeypatch):
        """Should raise ValueError when no API key available."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicPlanner()

    def test_init_with_excluded_actions(self):
        """Should store excluded actions."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(
                api_key="key", excluded_actions=["scroll_at", "drag_and_drop"]
            )

            assert planner.excluded_actions == ["scroll_at", "drag_and_drop"]

    def test_init_with_custom_model(self):
        """Should use custom model when provided."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key", model="claude-sonnet-4-20250514")

            assert planner.model == "claude-sonnet-4-20250514"
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    def test_init_uses_env_model(self, monkeypatch):
        """Should use ANTHROPIC_MODEL env var."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-custom")

        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner()

            assert planner.model == "claude-custom"

    def test_init_binds_tools(self):
        """Should bind VNC tools to LLM."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            AnthropicPlanner(api_key="key")

            mock_llm.bind_tools.assert_called_once()

    def test_init_default_model(self):
        """Should use DEFAULT_MODEL when no model specified."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")

            assert planner.model == DEFAULT_MODEL


class TestAnthropicPlannerGenerateStateless:
    """Tests for generate_stateless method."""

    def test_generate_stateless_calls_llm(self):
        """Should invoke LLM with tools."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="I see the screen", tool_calls=[])
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_chat.return_value = mock_llm
            planner = AnthropicPlanner(api_key="key")
            screenshot = create_test_png()

            result = planner.generate_stateless(
                task="Click the button",
                action_history=[],
                screenshot_png=screenshot,
            )

            mock_llm_with_tools.invoke.assert_called_once()
            assert result == mock_response

    def test_generate_stateless_includes_task_in_prompt(self):
        """Should include task in system prompt."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="", tool_calls=[])
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_chat.return_value = mock_llm
            planner = AnthropicPlanner(api_key="key")
            screenshot = create_test_png()

            planner.generate_stateless(
                task="Open the browser",
                action_history=[],
                screenshot_png=screenshot,
            )

            # Check the messages passed to invoke
            call_args = mock_llm_with_tools.invoke.call_args
            messages = call_args[0][0]
            # First message is system message
            assert "Open the browser" in messages[0].content

    def test_generate_stateless_includes_action_history(self):
        """Should include action history in prompt."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="", tool_calls=[])
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_chat.return_value = mock_llm
            planner = AnthropicPlanner(api_key="key")
            screenshot = create_test_png()

            planner.generate_stateless(
                task="Task",
                action_history=["Clicked button", "Typed text"],
                screenshot_png=screenshot,
            )

            call_args = mock_llm_with_tools.invoke.call_args
            messages = call_args[0][0]
            system_content = messages[0].content
            assert "Clicked button" in system_content
            assert "Typed text" in system_content

    def test_generate_stateless_includes_screenshot(self):
        """Should include screenshot in user message."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="", tool_calls=[])
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_chat.return_value = mock_llm
            planner = AnthropicPlanner(api_key="key")
            screenshot = create_test_png()

            planner.generate_stateless(
                task="Task",
                action_history=[],
                screenshot_png=screenshot,
            )

            call_args = mock_llm_with_tools.invoke.call_args
            messages = call_args[0][0]
            # Second message is human message with screenshot
            human_content = messages[1].content
            assert any("image" in str(item) for item in human_content)

    def test_generate_stateless_raises_on_wrong_response_type(self):
        """Should raise TypeError if response is not AIMessage."""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = "not an AIMessage"
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_chat.return_value = mock_llm
            planner = AnthropicPlanner(api_key="key")
            screenshot = create_test_png()

            with pytest.raises(TypeError, match="Expected AIMessage"):
                planner.generate_stateless(
                    task="Task",
                    action_history=[],
                    screenshot_png=screenshot,
                )


class TestAnthropicPlannerExtractText:
    """Tests for extract_text method."""

    def test_extract_text_from_string_content(self):
        """Should extract text from string content."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(content="I see a button", tool_calls=[])

            result = planner.extract_text(response)

            assert result == "I see a button"

    def test_extract_text_from_list_content(self):
        """Should extract text from list content with text blocks."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(
                content=[
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ],
                tool_calls=[],
            )

            result = planner.extract_text(response)

            assert result == "First part Second part"

    def test_extract_text_from_string_list_content(self):
        """Should extract text from list of strings."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(content=["String one", "String two"], tool_calls=[])

            result = planner.extract_text(response)

            assert result == "String one String two"

    def test_extract_text_returns_empty_for_empty_string(self):
        """Should return empty string for empty content."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(content="", tool_calls=[])

            result = planner.extract_text(response)

            assert result == ""

    def test_extract_text_skips_non_text_blocks(self):
        """Should skip non-text blocks in list content."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(
                content=[
                    {"type": "text", "text": "Has text"},
                    {"type": "image", "data": "..."},
                ],
                tool_calls=[],
            )

            result = planner.extract_text(response)

            assert result == "Has text"


class TestAnthropicPlannerExtractFunctionCalls:
    """Tests for extract_function_calls method."""

    def test_extract_function_calls(self):
        """Should extract function calls from tool_calls."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(
                content="",
                tool_calls=[{"name": "click_at", "args": {"x": 100, "y": 200}, "id": "call_1"}],
            )

            result = planner.extract_function_calls(response)

            assert len(result) == 1
            assert result[0]["name"] == "click_at"
            assert result[0]["args"] == {"x": 100, "y": 200}

    def test_extract_multiple_function_calls(self):
        """Should extract multiple function calls."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(
                content="",
                tool_calls=[
                    {"name": "click_at", "args": {"x": 100, "y": 200}, "id": "call_1"},
                    {"name": "type_text_at", "args": {"text": "hello"}, "id": "call_2"},
                ],
            )

            result = planner.extract_function_calls(response)

            assert len(result) == 2
            assert result[0]["name"] == "click_at"
            assert result[1]["name"] == "type_text_at"

    def test_extract_function_calls_empty(self):
        """Should return empty list when no tool calls."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(content="No action needed", tool_calls=[])

            result = planner.extract_function_calls(response)

            assert result == []


class TestAnthropicPlannerExtractSafetyDecision:
    """Tests for extract_safety_decision method."""

    def test_extract_safety_decision_returns_none_with_tool_calls(self):
        """Should return None when response has tool calls."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(
                content="Clicking button", tool_calls=[{"name": "click_at", "args": {}, "id": "1"}]
            )

            result = planner.extract_safety_decision(response)

            assert result is None

    def test_extract_safety_decision_detects_cannot_refusal(self):
        """Should detect 'I cannot' refusal."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(
                content="I cannot help with this request as it appears dangerous", tool_calls=[]
            )

            result = planner.extract_safety_decision(response)

            assert result is not None
            assert result["action"] == "block"
            assert "refused" in result["reason"].lower()

    def test_extract_safety_decision_detects_cant_refusal(self):
        """Should detect 'I can't' refusal."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(content="I can't perform this action", tool_calls=[])

            result = planner.extract_safety_decision(response)

            assert result is not None
            assert result["action"] == "block"

    def test_extract_safety_decision_detects_unsafe(self):
        """Should detect 'unsafe' keyword."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(content="This appears to be an unsafe operation", tool_calls=[])

            result = planner.extract_safety_decision(response)

            assert result is not None
            assert result["action"] == "block"

    def test_extract_safety_decision_detects_wont(self):
        """Should detect 'I won't' refusal."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(content="I won't do that", tool_calls=[])

            result = planner.extract_safety_decision(response)

            assert result is not None
            assert result["action"] == "block"

    def test_extract_safety_decision_returns_none_for_normal_text(self):
        """Should return None for normal response without refusal."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(content="The task is complete", tool_calls=[])

            result = planner.extract_safety_decision(response)

            assert result is None

    def test_extract_safety_decision_case_insensitive(self):
        """Should detect refusals case-insensitively."""
        with patch("src.vnc_use.planners.anthropic.ChatAnthropic") as mock_chat:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = MagicMock()
            mock_chat.return_value = mock_llm

            planner = AnthropicPlanner(api_key="key")
            response = AIMessage(content="I CANNOT help with this", tool_calls=[])

            result = planner.extract_safety_decision(response)

            assert result is not None
            assert result["action"] == "block"


class TestDefaultModel:
    """Tests for DEFAULT_MODEL constant."""

    def test_default_model_is_set(self):
        """Should have DEFAULT_MODEL constant."""
        assert DEFAULT_MODEL is not None
        assert "claude" in DEFAULT_MODEL.lower() or "haiku" in DEFAULT_MODEL.lower()
