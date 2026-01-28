"""Tests for backends/vnc.py module."""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.vnc_use.backends.vnc import VNCController, denorm_x, denorm_y


def create_test_png(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG image for testing."""
    img = Image.new("RGB", (width, height), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDenormX:
    """Tests for denorm_x function."""

    def test_denorm_x_zero(self):
        """Should return 0 for x=0."""
        assert denorm_x(0, 1440) == 0

    def test_denorm_x_max(self):
        """Should return width for x=1000."""
        assert denorm_x(1000, 1440) == 1440

    def test_denorm_x_center(self):
        """Should return half width for x=500."""
        assert denorm_x(500, 1000) == 500

    def test_denorm_x_rounds(self):
        """Should round to nearest integer."""
        result = denorm_x(333, 1000)
        assert isinstance(result, int)
        assert result == 333


class TestDenormY:
    """Tests for denorm_y function."""

    def test_denorm_y_zero(self):
        """Should return 0 for y=0."""
        assert denorm_y(0, 900) == 0

    def test_denorm_y_max(self):
        """Should return height for y=1000."""
        assert denorm_y(1000, 900) == 900

    def test_denorm_y_center(self):
        """Should return half height for y=500."""
        assert denorm_y(500, 800) == 400

    def test_denorm_y_rounds(self):
        """Should round to nearest integer."""
        result = denorm_y(333, 900)
        assert isinstance(result, int)
        assert result == 300  # 333 * 900 / 1000 = 299.7, rounds to 300


class TestVNCControllerInit:
    """Tests for VNCController initialization."""

    def test_init_creates_controller(self):
        """Should create controller without connection."""
        controller = VNCController()
        assert controller.client is None
        assert controller._screen_size is None


class TestVNCControllerConnect:
    """Tests for connect method."""

    def test_connect_establishes_connection(self):
        """Should connect to VNC server."""
        mock_client = MagicMock()

        with patch("src.vnc_use.backends.vnc.vnc_api.connect", return_value=mock_client):
            controller = VNCController()
            result = controller.connect("localhost::5901", "password")

            assert controller.client == mock_client
            assert result is controller  # Returns self for chaining

    def test_connect_passes_server_and_password(self):
        """Should pass server and password to vnc_api."""
        with patch("src.vnc_use.backends.vnc.vnc_api.connect") as mock_connect:
            controller = VNCController()
            controller.connect("myhost::5902", "secret")

            mock_connect.assert_called_once_with("myhost::5902", password="secret")


class TestVNCControllerDisconnect:
    """Tests for disconnect method."""

    def test_disconnect_closes_connection(self):
        """Should disconnect from VNC server."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client

        controller.disconnect()

        mock_client.disconnect.assert_called_once()
        assert controller.client is None

    def test_disconnect_handles_no_connection(self):
        """Should handle disconnect when not connected."""
        controller = VNCController()
        controller.disconnect()  # Should not raise


class TestVNCControllerScreenshotPng:
    """Tests for screenshot_png method."""

    def test_screenshot_png_raises_when_not_connected(self):
        """Should raise when not connected."""
        controller = VNCController()

        with pytest.raises(RuntimeError, match="Not connected"):
            controller.screenshot_png()

    def test_screenshot_png_captures_screen(self, tmp_path):
        """Should capture screenshot and return bytes."""
        mock_client = MagicMock()
        test_png = create_test_png(640, 480)

        # Mock captureScreen to write PNG to file
        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(test_png)

        mock_client.captureScreen.side_effect = capture_side_effect

        controller = VNCController()
        controller.client = mock_client

        result = controller.screenshot_png()

        assert isinstance(result, bytes)
        mock_client.captureScreen.assert_called_once()

    def test_screenshot_png_updates_screen_size(self, tmp_path):
        """Should update cached screen size."""
        mock_client = MagicMock()
        test_png = create_test_png(1920, 1080)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(test_png)

        mock_client.captureScreen.side_effect = capture_side_effect

        controller = VNCController()
        controller.client = mock_client

        controller.screenshot_png()

        assert controller._screen_size == (1920, 1080)


class TestVNCControllerGetScreenSize:
    """Tests for get_screen_size method."""

    def test_get_screen_size_raises_when_unknown(self):
        """Should raise when screen size not captured."""
        controller = VNCController()

        with pytest.raises(RuntimeError, match="Screen size unknown"):
            controller.get_screen_size()

    def test_get_screen_size_returns_cached_size(self):
        """Should return cached screen size."""
        controller = VNCController()
        controller._screen_size = (1440, 900)

        result = controller.get_screen_size()

        assert result == (1440, 900)


class TestVNCControllerMove:
    """Tests for move method."""

    def test_move_raises_when_not_connected(self):
        """Should raise when not connected."""
        controller = VNCController()

        with pytest.raises(RuntimeError, match="Not connected"):
            controller.move(100, 200)

    def test_move_calls_mouse_move(self):
        """Should call mouseMove on client."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.move(500, 300)

        mock_client.mouseMove.assert_called_once_with(500, 300)


class TestVNCControllerClick:
    """Tests for click method."""

    def test_click_raises_when_not_connected(self):
        """Should raise when not connected."""
        controller = VNCController()

        with pytest.raises(RuntimeError, match="Not connected"):
            controller.click(100, 200)

    def test_click_moves_and_presses(self):
        """Should move mouse and press button."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.click(100, 200)

        mock_client.mouseMove.assert_called_once_with(100, 200)
        mock_client.mouseDown.assert_called_once_with(1)
        mock_client.mouseUp.assert_called_once_with(1)

    def test_click_with_custom_button(self):
        """Should use custom button."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.click(100, 200, button=3)

        mock_client.mouseDown.assert_called_once_with(3)
        mock_client.mouseUp.assert_called_once_with(3)


class TestVNCControllerDoubleClick:
    """Tests for double_click method."""

    def test_double_click_raises_when_not_connected(self):
        """Should raise when not connected."""
        controller = VNCController()

        with pytest.raises(RuntimeError, match="Not connected"):
            controller.double_click(100, 200)

    def test_double_click_presses_twice(self):
        """Should press button twice."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.double_click(100, 200)

        mock_client.mouseMove.assert_called_once()
        assert mock_client.mousePress.call_count == 2


class TestVNCControllerDragAndDrop:
    """Tests for drag_and_drop method."""

    def test_drag_and_drop_raises_when_not_connected(self):
        """Should raise when not connected."""
        controller = VNCController()

        with pytest.raises(RuntimeError, match="Not connected"):
            controller.drag_and_drop(100, 100, 500, 500)

    def test_drag_and_drop_executes_drag(self):
        """Should execute drag sequence."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.drag_and_drop(100, 100, 500, 500)

        mock_client.mouseMove.assert_called_once_with(100, 100)
        mock_client.mouseDown.assert_called_once_with(1)
        mock_client.mouseDrag.assert_called_once_with(500, 500)
        mock_client.mouseUp.assert_called_once_with(1)


class TestVNCControllerTypeText:
    """Tests for type_text method."""

    def test_type_text_raises_when_not_connected(self):
        """Should raise when not connected."""
        controller = VNCController()

        with pytest.raises(RuntimeError, match="Not connected"):
            controller.type_text("hello")

    def test_type_text_types_each_character(self):
        """Should type each character."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.type_text("hi")

        assert mock_client.keyPress.call_count == 2
        mock_client.keyPress.assert_any_call("h")
        mock_client.keyPress.assert_any_call("i")

    def test_type_text_with_press_enter(self):
        """Should press Enter when enabled."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.type_text("test", press_enter=True)

        # Last call should be enter
        calls = mock_client.keyPress.call_args_list
        assert calls[-1][0][0] == "enter"

    def test_type_text_with_clear_first(self):
        """Should clear text first when enabled."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.type_text("new", clear_first=True)

        calls = mock_client.keyPress.call_args_list
        assert calls[0][0][0] == "ctrl-a"
        assert calls[1][0][0] == "delete"


class TestVNCControllerKeyCombo:
    """Tests for key_combo method."""

    def test_key_combo_raises_when_not_connected(self):
        """Should raise when not connected."""
        controller = VNCController()

        with pytest.raises(RuntimeError, match="Not connected"):
            controller.key_combo("control+c")

    def test_key_combo_normalizes_format(self):
        """Should normalize key format."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.key_combo("control+a")

        mock_client.keyPress.assert_called_once_with("ctrl-a")

    def test_key_combo_handles_existing_format(self):
        """Should handle already normalized format."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.key_combo("alt+tab")

        mock_client.keyPress.assert_called_once_with("alt-tab")


class TestVNCControllerScroll:
    """Tests for scroll method."""

    def test_scroll_raises_when_not_connected(self):
        """Should raise when not connected."""
        controller = VNCController()

        with pytest.raises(RuntimeError, match="Not connected"):
            controller.scroll("down")

    def test_scroll_down_uses_pgdn(self):
        """Should use PageDown for scrolling down."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.scroll("down", magnitude=400)

        mock_client.keyPress.assert_called_with("pgdn")

    def test_scroll_up_uses_pgup(self):
        """Should use PageUp for scrolling up."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.scroll("up", magnitude=400)

        mock_client.keyPress.assert_called_with("pgup")

    def test_scroll_repeats_based_on_magnitude(self):
        """Should repeat based on magnitude."""
        mock_client = MagicMock()
        controller = VNCController()
        controller.client = mock_client

        controller.scroll("down", magnitude=800)

        assert mock_client.keyPress.call_count == 2  # 800 / 400 = 2


class TestVNCControllerExecuteAction:
    """Tests for execute_action method."""

    def test_execute_action_click_at(self):
        """Should execute click_at action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        # Mock screenshot
        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action("click_at", {"x": 500, "y": 500})

        assert result.success is True
        # click_at uses click() which calls mouseDown/mouseUp
        assert mock_client.mouseDown.call_count >= 1
        assert mock_client.mouseUp.call_count >= 1

    def test_execute_action_type_text_at(self):
        """Should execute type_text_at action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action("type_text_at", {"x": 100, "y": 200, "text": "hello"})

        assert result.success is True
        # Should have typed the text
        assert mock_client.keyPress.call_count >= 5  # "hello" = 5 chars

    def test_execute_action_key_combination(self):
        """Should execute key_combination action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action("key_combination", {"keys": "control+s"})

        assert result.success is True
        mock_client.keyPress.assert_called()

    def test_execute_action_scroll_document(self):
        """Should execute scroll_document action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action(
            "scroll_document", {"direction": "down", "magnitude": 400}
        )

        assert result.success is True

    def test_execute_action_wait_5_seconds(self):
        """Should execute wait_5_seconds action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        with patch("src.vnc_use.backends.vnc.time.sleep") as mock_sleep:
            result = controller.execute_action("wait_5_seconds", {})

            mock_sleep.assert_called_once_with(5)
            assert result.success is True

    def test_execute_action_unknown_action(self):
        """Should return error for unknown action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action("unknown_action", {})

        assert result.success is False
        assert "Unknown action" in result.error

    def test_execute_action_handles_exception(self):
        """Should handle exceptions gracefully."""
        mock_client = MagicMock()
        mock_client.mouseMove.side_effect = Exception("VNC error")

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action("click_at", {"x": 100, "y": 200})

        assert result.success is False
        assert "VNC error" in result.error

    def test_execute_action_hover_at(self):
        """Should execute hover_at action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action("hover_at", {"x": 300, "y": 400})

        assert result.success is True
        mock_client.mouseMove.assert_called()

    def test_execute_action_scroll_at(self):
        """Should execute scroll_at action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action("scroll_at", {"x": 500, "y": 500, "direction": "up"})

        assert result.success is True
        mock_client.mouseMove.assert_called()
        mock_client.keyPress.assert_called()

    def test_execute_action_drag_and_drop(self):
        """Should execute drag_and_drop action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action(
            "drag_and_drop", {"x": 100, "y": 100, "destination_x": 500, "destination_y": 500}
        )

        assert result.success is True
        mock_client.mouseDown.assert_called()
        mock_client.mouseDrag.assert_called()
        mock_client.mouseUp.assert_called()

    def test_execute_action_double_click_at(self):
        """Should execute double_click_at action."""
        mock_client = MagicMock()

        controller = VNCController()
        controller.client = mock_client
        controller._screen_size = (1000, 1000)

        def capture_side_effect(path):
            with open(path, "wb") as f:
                f.write(create_test_png())

        mock_client.captureScreen.side_effect = capture_side_effect

        result = controller.execute_action("double_click_at", {"x": 250, "y": 250})

        assert result.success is True
        # Double click should press twice
        assert mock_client.mousePress.call_count == 2
