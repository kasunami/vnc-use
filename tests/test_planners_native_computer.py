"""Tests for Native Anthropic Computer Use planner."""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# Create a minimal valid PNG for tests
def create_test_png(width: int = 100, height: int = 75) -> bytes:
    """Create a minimal valid PNG image for testing."""
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestNativeComputerPlannerInit:
    """Tests for NativeComputerPlanner initialization."""

    def test_init_with_explicit_params(self, monkeypatch):
        """Test initialization with explicit parameters."""
        from vnc_use.planners.native_computer import NativeComputerPlanner

        planner = NativeComputerPlanner(
            excluded_actions=["scroll"],
            model="custom-model",
            api_key="test-api-key",
            display_width=1920,
            display_height=1080,
        )

        assert planner.excluded_actions == ["scroll"]
        assert planner.model == "custom-model"
        assert planner.api_key == "test-api-key"
        assert planner.display_width == 1920
        assert planner.display_height == 1080

    def test_init_from_env_vars(self, monkeypatch):
        """Test initialization from environment variables."""
        from vnc_use.planners.native_computer import NativeComputerPlanner

        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-api-key")
        monkeypatch.setenv("COMPUTER_USE_MODEL", "env-model")

        planner = NativeComputerPlanner()

        assert planner.api_key == "env-api-key"
        assert planner.model == "env-model"

    def test_init_default_model(self, monkeypatch):
        """Test that default model is used when not specified."""
        from vnc_use.planners.native_computer import DEFAULT_MODEL, NativeComputerPlanner

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("COMPUTER_USE_MODEL", raising=False)

        planner = NativeComputerPlanner()

        assert planner.model == DEFAULT_MODEL

    def test_init_missing_api_key_raises(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        from vnc_use.planners.native_computer import NativeComputerPlanner

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable must be set"):
            NativeComputerPlanner()


class TestScaleScreenshot:
    """Tests for _scale_screenshot method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_scale_screenshot_calls_compress(self, planner):
        """Test that _scale_screenshot calls compress_screenshot."""
        png_bytes = create_test_png(2048, 1536)

        with patch("vnc_use.planners.native_computer.compress_screenshot") as mock_compress:
            mock_compress.return_value = create_test_png(1024, 768)
            result = planner._scale_screenshot(png_bytes)

        mock_compress.assert_called_once_with(png_bytes, max_width=planner.api_width)
        assert result == mock_compress.return_value


class TestGenerateStateless:
    """Tests for generate_stateless method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_generate_stateless_success(self, planner):
        """Test successful API call."""
        png_bytes = create_test_png()

        # Create mock response
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "computer"
        mock_tool_use.input = {"action": "left_click", "coordinate": [100, 200]}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_use]

        planner.client.beta.messages.create = MagicMock(return_value=mock_response)

        result = planner.generate_stateless(
            task="Click the button",
            action_history=["Moved mouse"],
            screenshot_png=png_bytes,
        )

        assert result == mock_response
        planner.client.beta.messages.create.assert_called_once()

    def test_generate_stateless_with_action_history(self, planner):
        """Test that action history is included in prompt."""
        png_bytes = create_test_png()

        mock_response = MagicMock()
        mock_response.content = []

        planner.client.beta.messages.create = MagicMock(return_value=mock_response)

        planner.generate_stateless(
            task="Test task",
            action_history=["Action 1", "Action 2", "Action 3"],
            screenshot_png=png_bytes,
        )

        # Check that the system prompt contains action history
        call_args = planner.client.beta.messages.create.call_args
        system_prompt = call_args.kwargs.get("system", "")
        assert "Actions taken so far" in system_prompt
        assert "1. Action 1" in system_prompt

    def test_generate_stateless_without_action_history(self, planner):
        """Test API call without action history."""
        png_bytes = create_test_png()

        mock_response = MagicMock()
        mock_response.content = []

        planner.client.beta.messages.create = MagicMock(return_value=mock_response)

        planner.generate_stateless(
            task="Test task",
            action_history=[],
            screenshot_png=png_bytes,
        )

        call_args = planner.client.beta.messages.create.call_args
        system_prompt = call_args.kwargs.get("system", "")
        assert "Actions taken so far" not in system_prompt

    def test_generate_stateless_api_error(self, planner, caplog):
        """Test graceful handling of API errors."""
        png_bytes = create_test_png()

        planner.client.beta.messages.create = MagicMock(side_effect=Exception("API Error"))

        result = planner.generate_stateless(
            task="Test task",
            action_history=[],
            screenshot_png=png_bytes,
        )

        assert hasattr(result, "content")
        assert hasattr(result, "stop_reason")
        assert result.stop_reason == "error"
        assert "Anthropic API call failed" in caplog.text

    def test_generate_stateless_no_tool_uses_warning(self, planner, caplog):
        """Test warning when response has no tool uses."""
        png_bytes = create_test_png()

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Just some text"

        mock_response = MagicMock()
        mock_response.content = [mock_text]

        planner.client.beta.messages.create = MagicMock(return_value=mock_response)

        planner.generate_stateless(
            task="Test task",
            action_history=[],
            screenshot_png=png_bytes,
        )

        assert "No tool uses in response" in caplog.text


class TestExtractText:
    """Tests for extract_text method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_extract_text_with_text_blocks(self, planner):
        """Test extracting text from response with text blocks."""
        mock_text1 = MagicMock()
        mock_text1.type = "text"
        mock_text1.text = "First text"

        mock_text2 = MagicMock()
        mock_text2.type = "text"
        mock_text2.text = "Second text"

        mock_response = MagicMock()
        mock_response.content = [mock_text1, mock_text2]

        result = planner.extract_text(mock_response)

        assert result == "First text Second text"

    def test_extract_text_no_text_blocks(self, planner):
        """Test extracting text when there are no text blocks."""
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"

        mock_response = MagicMock()
        mock_response.content = [mock_tool]

        result = planner.extract_text(mock_response)

        assert result == ""

    def test_extract_text_mixed_content(self, planner):
        """Test extracting text from mixed content."""
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Some reasoning"

        mock_tool = MagicMock()
        mock_tool.type = "tool_use"

        mock_response = MagicMock()
        mock_response.content = [mock_text, mock_tool]

        result = planner.extract_text(mock_response)

        assert result == "Some reasoning"


class TestExtractFunctionCalls:
    """Tests for extract_function_calls method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_extract_function_calls_with_tool_use(self, planner):
        """Test extracting function calls from tool_use blocks."""
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.name = "computer"
        mock_tool.input = {"action": "left_click", "coordinate": [100, 200]}

        mock_response = MagicMock()
        mock_response.content = [mock_tool]

        result = planner.extract_function_calls(mock_response)

        assert len(result) == 1
        assert result[0]["name"] == "click_at"
        assert result[0]["args"] == {"x": 100, "y": 200}

    def test_extract_function_calls_unexpected_tool_name(self, planner, caplog):
        """Test warning for unexpected tool name."""
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.name = "other_tool"
        mock_tool.input = {"action": "left_click"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool]

        result = planner.extract_function_calls(mock_response)

        assert len(result) == 0
        assert "Unexpected tool name" in caplog.text

    def test_extract_function_calls_missing_action(self, planner, caplog):
        """Test warning for missing action field."""
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.name = "computer"
        mock_tool.input = {}  # No action field

        mock_response = MagicMock()
        mock_response.content = [mock_tool]

        result = planner.extract_function_calls(mock_response)

        assert len(result) == 0
        assert "missing 'action' field" in caplog.text

    def test_extract_function_calls_skips_non_tool_use(self, planner):
        """Test that non-tool_use blocks are skipped."""
        mock_text = MagicMock()
        mock_text.type = "text"

        mock_response = MagicMock()
        mock_response.content = [mock_text]

        result = planner.extract_function_calls(mock_response)

        assert len(result) == 0


class TestProcessAction:
    """Tests for _process_action method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_process_action_unknown(self, planner, caplog):
        """Test handling of unknown action type."""
        result = planner._process_action("unknown_action", {})

        assert result is None
        assert "Unknown action type" in caplog.text


class TestHandleCoordinateAction:
    """Tests for _handle_coordinate_action method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    @pytest.mark.parametrize(
        "action,expected_vnc_action",
        [
            ("left_click", "click_at"),
            ("double_click", "double_click_at"),
            ("right_click", "right_click_at"),
            ("triple_click", "triple_click_at"),
            ("middle_click", "middle_click_at"),
            ("mouse_move", "hover_at"),
        ],
    )
    def test_coordinate_actions(self, planner, action, expected_vnc_action):
        """Test all coordinate-based actions."""
        input_data = {"coordinate": [100, 200]}

        result = planner._handle_coordinate_action(action, input_data)

        assert result is not None
        assert result["name"] == expected_vnc_action
        assert result["args"] == {"x": 100, "y": 200}

    def test_coordinate_action_missing_coordinate(self, planner, caplog):
        """Test handling of missing coordinate."""
        result = planner._handle_coordinate_action("left_click", {})

        assert result is None
        assert "Invalid coordinate" in caplog.text

    def test_coordinate_action_invalid_coordinate_length(self, planner, caplog):
        """Test handling of invalid coordinate length."""
        result = planner._handle_coordinate_action("left_click", {"coordinate": [100]})

        assert result is None
        assert "Invalid coordinate" in caplog.text


class TestHandleSimpleAction:
    """Tests for _handle_simple_action method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    @pytest.mark.parametrize(
        "action,expected_vnc_action",
        [
            ("left_mouse_down", "left_mouse_down"),
            ("left_mouse_up", "left_mouse_up"),
            ("cursor_position", "cursor_position"),
            ("wait", "wait_5_seconds"),
        ],
    )
    def test_simple_actions(self, planner, action, expected_vnc_action):
        """Test all simple actions."""
        result = planner._handle_simple_action(action, {})

        assert result is not None
        assert result["name"] == expected_vnc_action
        assert result["args"] == {}


class TestHandleTextAction:
    """Tests for _handle_text_action method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_type_action(self, planner):
        """Test type action."""
        result = planner._handle_text_action("type", {"text": "Hello World"})

        assert result is not None
        assert result["name"] == "type_text"
        assert result["args"] == {"text": "Hello World"}

    def test_key_action(self, planner):
        """Test key action."""
        result = planner._handle_text_action("key", {"text": "ctrl+c"})

        assert result is not None
        assert result["name"] == "key_combination"
        assert result["args"] == {"keys": "ctrl+c"}

    def test_text_action_missing_text(self, planner, caplog):
        """Test handling of missing text field."""
        result = planner._handle_text_action("type", {})

        assert result is None
        assert "missing 'text' field" in caplog.text


class TestHandleDragAction:
    """Tests for _handle_drag_action method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_drag_action_valid(self, planner):
        """Test valid drag action."""
        input_data = {
            "start_coordinate": [100, 200],
            "end_coordinate": [300, 400],
        }

        result = planner._handle_drag_action("left_click_drag", input_data)

        assert result is not None
        assert result["name"] == "drag_and_drop"
        assert result["args"] == {
            "start_x": 100,
            "start_y": 200,
            "end_x": 300,
            "end_y": 400,
        }

    def test_drag_action_missing_start(self, planner, caplog):
        """Test handling of missing start coordinate."""
        input_data = {"end_coordinate": [300, 400]}

        result = planner._handle_drag_action("left_click_drag", input_data)

        assert result is None
        assert "Invalid coordinates" in caplog.text

    def test_drag_action_missing_end(self, planner, caplog):
        """Test handling of missing end coordinate."""
        input_data = {"start_coordinate": [100, 200]}

        result = planner._handle_drag_action("left_click_drag", input_data)

        assert result is None
        assert "Invalid coordinates" in caplog.text

    def test_drag_action_invalid_coordinate_length(self, planner, caplog):
        """Test handling of invalid coordinate length."""
        input_data = {
            "start_coordinate": [100],
            "end_coordinate": [300, 400],
        }

        result = planner._handle_drag_action("left_click_drag", input_data)

        assert result is None
        assert "Invalid coordinates" in caplog.text


class TestHandleScrollAction:
    """Tests for _handle_scroll_action method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_scroll_with_coordinate(self, planner):
        """Test scroll action with coordinate."""
        input_data = {
            "scroll_direction": "down",
            "scroll_amount": 3,
            "coordinate": [100, 200],
        }

        result = planner._handle_scroll_action("scroll", input_data)

        assert result is not None
        assert result["name"] == "scroll_at"
        assert result["args"]["x"] == 100
        assert result["args"]["y"] == 200
        assert result["args"]["direction"] == "down"
        assert result["args"]["magnitude"] == 480  # 3 * 160

    def test_scroll_without_coordinate(self, planner):
        """Test scroll action without coordinate."""
        input_data = {
            "scroll_direction": "up",
            "scroll_amount": 2,
        }

        result = planner._handle_scroll_action("scroll", input_data)

        assert result is not None
        assert result["name"] == "scroll_document"
        assert result["args"]["direction"] == "up"
        assert result["args"]["magnitude"] == 320  # 2 * 160

    def test_scroll_default_amount(self, planner):
        """Test scroll action with default amount."""
        input_data = {"scroll_direction": "down"}

        result = planner._handle_scroll_action("scroll", input_data)

        assert result is not None
        assert result["args"]["magnitude"] == 800  # default 5 * 160

    def test_scroll_missing_direction(self, planner, caplog):
        """Test handling of missing scroll direction."""
        result = planner._handle_scroll_action("scroll", {})

        assert result is None
        assert "missing 'scroll_direction' field" in caplog.text


class TestHandleHoldKeyAction:
    """Tests for _handle_hold_key_action method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_hold_key_valid(self, planner):
        """Test valid hold_key action."""
        input_data = {"text": "shift", "duration": 2.5}

        result = planner._handle_hold_key_action("hold_key", input_data)

        assert result is not None
        assert result["name"] == "hold_key"
        assert result["args"] == {"key": "shift", "duration": 2.5}

    def test_hold_key_missing_text(self, planner, caplog):
        """Test handling of missing text field."""
        input_data = {"duration": 2}

        result = planner._handle_hold_key_action("hold_key", input_data)

        assert result is None
        assert "missing 'text' field" in caplog.text

    def test_hold_key_missing_duration(self, planner, caplog):
        """Test handling of missing duration."""
        input_data = {"text": "shift"}

        result = planner._handle_hold_key_action("hold_key", input_data)

        assert result is None
        assert "invalid duration" in caplog.text

    def test_hold_key_invalid_duration_type(self, planner, caplog):
        """Test handling of invalid duration type."""
        input_data = {"text": "shift", "duration": "invalid"}

        result = planner._handle_hold_key_action("hold_key", input_data)

        assert result is None
        assert "invalid duration" in caplog.text

    def test_hold_key_duration_negative(self, planner, caplog):
        """Test handling of negative duration."""
        input_data = {"text": "shift", "duration": -1}

        result = planner._handle_hold_key_action("hold_key", input_data)

        assert result is None
        assert "duration out of range" in caplog.text

    def test_hold_key_duration_too_large(self, planner, caplog):
        """Test handling of duration exceeding maximum."""
        input_data = {"text": "shift", "duration": 150}

        result = planner._handle_hold_key_action("hold_key", input_data)

        assert result is None
        assert "duration out of range" in caplog.text


class TestHandleScreenshotAction:
    """Tests for _handle_screenshot_action method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_screenshot_returns_none(self, planner):
        """Test that screenshot action returns None (ignored)."""
        result = planner._handle_screenshot_action("screenshot", {})

        assert result is None


class TestScaleCoordToNative:
    """Tests for _scale_coord_to_native method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_scale_coord_no_change(self, planner):
        """Test scaling when dimensions are equal."""
        result = planner._scale_coord_to_native(100, 1024, 1024)

        assert result == 100

    def test_scale_coord_double(self, planner):
        """Test scaling to double the resolution."""
        result = planner._scale_coord_to_native(100, 1024, 2048)

        assert result == 200

    def test_scale_coord_half(self, planner):
        """Test scaling to half the resolution."""
        result = planner._scale_coord_to_native(100, 1024, 512)

        assert result == 50

    def test_scale_coord_rounds(self, planner):
        """Test that scaling rounds to nearest integer."""
        result = planner._scale_coord_to_native(100, 1024, 1536)

        assert result == 150


class TestExtractSafetyDecision:
    """Tests for extract_safety_decision method."""

    @pytest.fixture
    def planner(self, monkeypatch):
        """Create a planner for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from vnc_use.planners.native_computer import NativeComputerPlanner

        return NativeComputerPlanner()

    def test_safety_decision_refusal(self, planner, caplog):
        """Test detection of refusal stop_reason."""
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "I cannot help with that request"

        mock_response = MagicMock()
        mock_response.content = [mock_text]
        mock_response.stop_reason = "refusal"

        result = planner.extract_safety_decision(mock_response)

        assert result is not None
        assert result["action"] == "block"
        assert "Model refused" in result["reason"]
        assert "refused via stop_reason='refusal'" in caplog.text

    def test_safety_decision_no_refusal(self, planner):
        """Test when stop_reason is not refusal."""
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"

        result = planner.extract_safety_decision(mock_response)

        assert result is None

    def test_safety_decision_no_stop_reason(self, planner):
        """Test when response has no stop_reason attribute."""
        mock_response = MagicMock(spec=[])  # No stop_reason attribute

        result = planner.extract_safety_decision(mock_response)

        assert result is None
