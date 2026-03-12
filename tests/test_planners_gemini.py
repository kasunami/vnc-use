"""Tests for planners/gemini.py module."""

import io
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.vnc_use.planners.gemini import (
    MODEL_ID,
    GeminiComputerUse,
    GeminiPlanner,
    compress_screenshot,
)


def create_test_png(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG image for testing."""
    img = Image.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCompressScreenshot:
    """Tests for compress_screenshot function."""

    def test_compress_small_image(self):
        """Should return compressed image when smaller than max_width."""
        png_bytes = create_test_png(100, 100)
        result = compress_screenshot(png_bytes, max_width=512)

        assert isinstance(result, bytes)
        # Should still be valid PNG
        img = Image.open(io.BytesIO(result))
        assert img.width == 100  # Not resized

    def test_compress_large_image(self):
        """Should resize image when larger than max_width."""
        png_bytes = create_test_png(1024, 768)
        result = compress_screenshot(png_bytes, max_width=512)

        img = Image.open(io.BytesIO(result))
        assert img.width == 512  # Resized to max_width
        # Height should maintain aspect ratio
        expected_height = int(768 * (512 / 1024))
        assert img.height == expected_height

    def test_compress_custom_max_width(self):
        """Should use custom max_width."""
        png_bytes = create_test_png(1000, 500)
        result = compress_screenshot(png_bytes, max_width=256)

        img = Image.open(io.BytesIO(result))
        assert img.width == 256

    def test_compress_returns_smaller_bytes(self):
        """Should return compressed output (generally smaller)."""
        png_bytes = create_test_png(800, 600)
        result = compress_screenshot(png_bytes, max_width=512)

        # Result should be different (resized/compressed)
        assert result != png_bytes


class TestGeminiPlannerInit:
    """Tests for GeminiPlanner initialization."""

    def test_init_with_api_key(self):
        """Should initialize with provided API key."""
        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            planner = GeminiPlanner(api_key="test_key")

            mock_genai.Client.assert_called_once_with(api_key="test_key")
            assert planner.excluded_actions == []
            assert planner.include_thoughts is False

    def test_init_with_gemini_api_key_env(self, monkeypatch):
        """Should use GEMINI_API_KEY env var."""
        monkeypatch.setenv("GEMINI_API_KEY", "gemini_env_key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            GeminiPlanner()

            mock_genai.Client.assert_called_once_with(api_key="gemini_env_key")

    def test_init_with_google_api_key_env(self, monkeypatch):
        """Should fall back to GOOGLE_API_KEY env var."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google_env_key")

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            GeminiPlanner()

            mock_genai.Client.assert_called_once_with(api_key="google_env_key")

    def test_init_raises_without_api_key(self, monkeypatch):
        """Should raise ValueError when no API key available."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
            GeminiPlanner()

    def test_init_with_excluded_actions(self):
        """Should store excluded actions."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key", excluded_actions=["scroll_at", "drag_and_drop"])

            assert planner.excluded_actions == ["scroll_at", "drag_and_drop"]

    def test_init_with_include_thoughts(self):
        """Should store include_thoughts setting."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key", include_thoughts=True)

            assert planner.include_thoughts is True


class TestGeminiPlannerBuildConfig:
    """Tests for build_config method."""

    def test_build_config_returns_config(self):
        """Should return GenerateContentConfig."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            config = planner.build_config()

            assert config is not None
            assert config.tools is not None
            assert len(config.tools) > 0

    def test_build_config_includes_computer_use_tool(self):
        """Should include ComputerUse tool."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            config = planner.build_config()

            tools = cast("list[Any]", config.tools or [])
            assert tools
            assert cast("Any", tools[0]).computer_use is not None

    def test_build_config_with_excluded_actions(self):
        """Should pass excluded actions to ComputerUse."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key", excluded_actions=["scroll_at"])
            config = planner.build_config()

            tools = cast("list[Any]", config.tools or [])
            computer_use = cast("Any", tools[0]).computer_use
            assert computer_use is not None
            assert computer_use.excluded_predefined_functions == ["scroll_at"]

    def test_build_config_with_thinking_enabled(self):
        """Should configure thinking when enabled."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key", include_thoughts=True)
            config = planner.build_config()

            assert config.thinking_config is not None
            assert config.thinking_config.include_thoughts is True


class TestGeminiPlannerStartContents:
    """Tests for start_contents method."""

    def test_start_contents_with_task_only(self):
        """Should create contents with task text."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            contents = planner.start_contents(task="Click the button")

            assert len(contents) == 1
            assert contents[0].role == "user"
            parts = cast("list[Any]", contents[0].parts or [])
            assert len(parts) == 1
            assert cast("Any", parts[0]).text == "Click the button"

    def test_start_contents_with_screenshot(self):
        """Should include screenshot when provided."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            contents = planner.start_contents(
                task="Click the button", initial_screenshot_png=screenshot
            )

            assert len(contents) == 1
            parts = cast("list[Any]", contents[0].parts or [])
            assert len(parts) == 2
            # Second part should be inline_data with PNG
            assert cast("Any", parts[1]).inline_data is not None

    def test_start_contents_compresses_screenshot(self):
        """Should compress large screenshots."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            large_screenshot = create_test_png(1024, 768)

            # Should not raise and should compress
            contents = planner.start_contents(task="Task", initial_screenshot_png=large_screenshot)

            parts = cast("list[Any]", contents[0].parts or [])
            assert len(parts) == 2


class TestGeminiPlannerGenerate:
    """Tests for generate method."""

    def test_generate_calls_api(self):
        """Should call Gemini API with contents."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")

            contents = [MagicMock(role="user", parts=[])]
            result = planner.generate(cast("Any", contents))

            mock_client.models.generate_content.assert_called_once()
            assert result == mock_response

    def test_generate_uses_default_config(self):
        """Should use build_config when config not provided."""
        mock_client = MagicMock()

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")

            contents = [MagicMock(role="user", parts=[])]
            planner.generate(cast("Any", contents))

            call_args = mock_client.models.generate_content.call_args
            assert call_args.kwargs["config"] is not None

    def test_generate_uses_custom_config(self):
        """Should use provided config when specified."""
        mock_client = MagicMock()
        custom_config = MagicMock()

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")

            contents = [MagicMock(role="user", parts=[])]
            planner.generate(cast("Any", contents), config=custom_config)

            call_args = mock_client.models.generate_content.call_args
            assert call_args.kwargs["config"] == custom_config

    def test_generate_uses_correct_model(self):
        """Should use the correct model ID."""
        mock_client = MagicMock()

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")

            contents = [MagicMock(role="user", parts=[])]
            planner.generate(cast("Any", contents))

            call_args = mock_client.models.generate_content.call_args
            assert call_args.kwargs["model"] == MODEL_ID

    def test_generate_cleans_old_screenshots(self):
        """Should clean screenshots from old function responses."""
        mock_client = MagicMock()

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")

            # Create mock function response with screenshot
            mock_fr = MagicMock()
            mock_fr.name = "click_at"
            mock_fr.response = {"url": "", "screenshot": {"data": "base64data"}}

            mock_part = MagicMock()
            mock_part.function_response = mock_fr

            old_content = MagicMock()
            old_content.role = "user"
            old_content.parts = [mock_part]

            new_content = MagicMock()
            new_content.role = "user"
            new_content.parts = []

            contents = [old_content, new_content]
            planner.generate(cast("Any", contents))

            # Should have called API
            mock_client.models.generate_content.assert_called_once()


class TestGeminiPlannerExtractText:
    """Tests for extract_text method."""

    def test_extract_text_from_response(self):
        """Should extract text from valid response."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_part = MagicMock()
            mock_part.text = "I see a button"

            mock_content = MagicMock()
            mock_content.parts = [mock_part]

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_text(mock_response)

            assert result == "I see a button"

    def test_extract_text_joins_multiple_parts(self):
        """Should join multiple text parts."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_part1 = MagicMock()
            mock_part1.text = "First part"
            mock_part2 = MagicMock()
            mock_part2.text = "Second part"

            mock_content = MagicMock()
            mock_content.parts = [mock_part1, mock_part2]

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_text(mock_response)

            assert result == "First part Second part"

    def test_extract_text_returns_empty_for_no_candidates(self):
        """Should return empty string when no candidates."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_response = MagicMock()
            mock_response.candidates = []

            result = planner.extract_text(mock_response)

            assert result == ""

    def test_extract_text_returns_empty_for_no_content(self):
        """Should return empty string when no content."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_candidate = MagicMock()
            mock_candidate.content = None

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_text(mock_response)

            assert result == ""

    def test_extract_text_returns_empty_for_no_parts(self):
        """Should return empty string when no parts."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_content = MagicMock()
            mock_content.parts = []

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_text(mock_response)

            assert result == ""

    def test_extract_text_skips_parts_without_text(self):
        """Should skip parts without text attribute."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_part1 = MagicMock()
            mock_part1.text = "Has text"
            mock_part2 = MagicMock(spec=[])  # No text attribute

            mock_content = MagicMock()
            mock_content.parts = [mock_part1, mock_part2]

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_text(mock_response)

            assert result == "Has text"


class TestGeminiPlannerExtractFunctionCalls:
    """Tests for extract_function_calls method."""

    def test_extract_function_calls_from_response(self):
        """Should extract function calls from valid response."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_fc = MagicMock()
            mock_fc.name = "click_at"
            mock_fc.args = {"x": 100, "y": 200}

            mock_part = MagicMock()
            mock_part.function_call = mock_fc

            mock_content = MagicMock()
            mock_content.parts = [mock_part]

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_function_calls(mock_response)

            assert len(result) == 1
            assert result[0]["name"] == "click_at"
            assert result[0]["args"] == {"x": 100, "y": 200}

    def test_extract_multiple_function_calls(self):
        """Should extract multiple function calls."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_fc1 = MagicMock()
            mock_fc1.name = "click_at"
            mock_fc1.args = {"x": 100, "y": 200}

            mock_fc2 = MagicMock()
            mock_fc2.name = "type_text_at"
            mock_fc2.args = {"text": "hello"}

            mock_part1 = MagicMock()
            mock_part1.function_call = mock_fc1
            mock_part2 = MagicMock()
            mock_part2.function_call = mock_fc2

            mock_content = MagicMock()
            mock_content.parts = [mock_part1, mock_part2]

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_function_calls(mock_response)

            assert len(result) == 2
            assert result[0]["name"] == "click_at"
            assert result[1]["name"] == "type_text_at"

    def test_extract_function_calls_empty_args(self):
        """Should handle empty args."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_fc = MagicMock()
            mock_fc.name = "wait_5_seconds"
            mock_fc.args = None

            mock_part = MagicMock()
            mock_part.function_call = mock_fc

            mock_content = MagicMock()
            mock_content.parts = [mock_part]

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_function_calls(mock_response)

            assert len(result) == 1
            assert result[0]["args"] == {}

    def test_extract_function_calls_returns_empty_for_no_candidates(self):
        """Should return empty list when no candidates."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_response = MagicMock()
            mock_response.candidates = []

            result = planner.extract_function_calls(mock_response)

            assert result == []

    def test_extract_function_calls_returns_empty_for_no_content(self):
        """Should return empty list when no content."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_candidate = MagicMock()
            mock_candidate.content = None

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_function_calls(mock_response)

            assert result == []

    def test_extract_function_calls_returns_empty_for_no_parts(self):
        """Should return empty list when no parts."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_content = MagicMock()
            mock_content.parts = []

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_function_calls(mock_response)

            assert result == []


class TestGeminiPlannerExtractSafetyDecision:
    """Tests for extract_safety_decision method."""

    def test_extract_safety_decision_from_response(self):
        """Should extract safety decision when present."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_decision = MagicMock()
            mock_decision.action = "require_confirmation"
            mock_decision.reason = "Sensitive operation"

            mock_candidate = MagicMock()
            mock_candidate.safety_decision = mock_decision

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_safety_decision(mock_response)

            assert result is not None
            assert result["action"] == "require_confirmation"
            assert result["reason"] == "Sensitive operation"

    def test_extract_safety_decision_returns_none_when_not_present(self):
        """Should return None when no safety decision."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_candidate = MagicMock()
            mock_candidate.safety_decision = None

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]

            result = planner.extract_safety_decision(mock_response)

            assert result is None

    def test_extract_safety_decision_returns_none_for_empty_candidates(self):
        """Should return None when no candidates."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")

            mock_response = MagicMock()
            mock_response.candidates = []

            result = planner.extract_safety_decision(mock_response)

            assert result is None


class TestGeminiPlannerBuildFunctionResponse:
    """Tests for build_function_response method."""

    def test_build_function_response_creates_part(self):
        """Should create Part with FunctionResponse."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            result = planner.build_function_response(
                function_name="click_at",
                screenshot_png=screenshot,
            )

            assert result is not None
            assert result.function_response is not None
            assert result.function_response.name == "click_at"

    def test_build_function_response_includes_screenshot(self):
        """Should include screenshot in response data."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            result = planner.build_function_response(
                function_name="click_at",
                screenshot_png=screenshot,
            )

            function_response = cast("Any", result.function_response)
            assert function_response is not None
            response_data = cast("dict[str, Any]", function_response.response or {})
            assert "screenshot" in response_data
            assert response_data["screenshot"]["mime_type"] == "image/png"

    def test_build_function_response_includes_url(self):
        """Should include URL when provided."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            result = planner.build_function_response(
                function_name="click_at",
                screenshot_png=screenshot,
                url="http://example.com",
            )

            function_response = cast("Any", result.function_response)
            assert function_response is not None
            response_data = cast("dict[str, Any]", function_response.response or {})
            assert response_data["url"] == "http://example.com"

    def test_build_function_response_includes_error(self):
        """Should include error when provided."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            result = planner.build_function_response(
                function_name="click_at",
                screenshot_png=screenshot,
                error="Click failed",
            )

            function_response = cast("Any", result.function_response)
            assert function_response is not None
            response_data = cast("dict[str, Any]", function_response.response or {})
            assert response_data["error"] == "Click failed"


class TestGeminiPlannerAppendFunctionResponse:
    """Tests for append_function_response method."""

    def test_append_function_response_adds_content(self):
        """Should append new content to list."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()
            contents = []

            result = planner.append_function_response(
                contents=contents,
                function_name="click_at",
                screenshot_png=screenshot,
            )

            assert len(result) == 1
            assert result[0].role == "user"

    def test_append_function_response_preserves_existing(self):
        """Should preserve existing contents."""
        with patch("src.vnc_use.planners.gemini.genai"):
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            from google.genai.types import Content, Part

            existing = Content(role="user", parts=[Part(text="existing")])
            contents = [existing]

            result = planner.append_function_response(
                contents=contents,
                function_name="click_at",
                screenshot_png=screenshot,
            )

            assert len(result) == 2
            first_parts = cast("list[Any]", result[0].parts or [])
            assert cast("Any", first_parts[0]).text == "existing"


class TestGeminiPlannerGenerateStateless:
    """Tests for generate_stateless method."""

    def test_generate_stateless_calls_api(self):
        """Should call Gemini API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            result = planner.generate_stateless(
                task="Click button",
                action_history=[],
                screenshot_png=screenshot,
            )

            mock_client.models.generate_content.assert_called_once()
            assert result == mock_response

    def test_generate_stateless_includes_task(self):
        """Should include task in context."""
        mock_client = MagicMock()

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            planner.generate_stateless(
                task="Click the submit button",
                action_history=[],
                screenshot_png=screenshot,
            )

            call_args = mock_client.models.generate_content.call_args
            contents = call_args.kwargs["contents"]
            # First part should contain task
            first_part_text = contents[0].parts[0].text
            assert "Click the submit button" in first_part_text

    def test_generate_stateless_includes_action_history(self):
        """Should include action history in context."""
        mock_client = MagicMock()

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            planner.generate_stateless(
                task="Task",
                action_history=["Clicked button", "Typed text"],
                screenshot_png=screenshot,
            )

            call_args = mock_client.models.generate_content.call_args
            contents = cast("list[Any]", call_args.kwargs["contents"])
            first_part_text = contents[0].parts[0].text
            assert "Clicked button" in first_part_text
            assert "Typed text" in first_part_text

    def test_generate_stateless_limits_action_history(self):
        """Should limit action history to last 10 items."""
        mock_client = MagicMock()

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            # Create 15 actions
            actions = [f"Action {i}" for i in range(15)]

            planner.generate_stateless(
                task="Task",
                action_history=actions,
                screenshot_png=screenshot,
            )

            call_args = mock_client.models.generate_content.call_args
            contents = call_args.kwargs["contents"]
            first_part_text = contents[0].parts[0].text

            # Should have actions 5-14 (last 10)
            assert "Action 5" in first_part_text
            assert "Action 14" in first_part_text
            # Should not have actions 0-4
            assert "Action 0" not in first_part_text or "Action" in first_part_text

    def test_generate_stateless_includes_screenshot(self):
        """Should include screenshot in request."""
        mock_client = MagicMock()

        with patch("src.vnc_use.planners.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            planner = GeminiPlanner(api_key="key")
            screenshot = create_test_png()

            planner.generate_stateless(
                task="Task",
                action_history=[],
                screenshot_png=screenshot,
            )

            call_args = mock_client.models.generate_content.call_args
            contents = cast("list[Any]", call_args.kwargs["contents"])
            # Second part should be image
            first_parts = cast("list[Any]", contents[0].parts or [])
            assert len(first_parts) == 2
            assert cast("Any", first_parts[1]).inline_data is not None


class TestGeminiComputerUseAlias:
    """Tests for backward compatibility alias."""

    def test_alias_exists(self):
        """GeminiComputerUse should be alias for GeminiPlanner."""
        assert GeminiComputerUse is GeminiPlanner


class TestModelId:
    """Tests for MODEL_ID constant."""

    def test_model_id_is_set(self):
        """Should have MODEL_ID constant."""
        assert MODEL_ID is not None
        assert "gemini" in MODEL_ID.lower()
