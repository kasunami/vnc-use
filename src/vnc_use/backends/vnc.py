"""VNC backend controller for executing UI actions."""

import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from vncdotool import api as vnc_api

from ..types import ActionResult

logger = logging.getLogger(__name__)

# Error messages
NOT_CONNECTED_ERROR = "Not connected to VNC server"

# Map Anthropic's Computer Use API key names to vncdotool key names
# Anthropic uses capitalized names, vncdotool uses lowercase
ANTHROPIC_TO_VNCDOTOOL_KEYS = {
    "Return": "return",
    "Enter": "enter",
    "Escape": "esc",
    "Backspace": "bsp",
    "Tab": "tab",
    "Delete": "delete",
    "Insert": "ins",
    "Home": "home",
    "End": "end",
    "PageUp": "pgup",
    "PageDown": "pgdn",
    "ArrowLeft": "left",
    "ArrowUp": "up",
    "ArrowRight": "right",
    "ArrowDown": "down",
    "Space": "space",
    # Function keys
    "F1": "f1",
    "F2": "f2",
    "F3": "f3",
    "F4": "f4",
    "F5": "f5",
    "F6": "f6",
    "F7": "f7",
    "F8": "f8",
    "F9": "f9",
    "F10": "f10",
    "F11": "f11",
    "F12": "f12",
}


def denorm_x(x: int, width: int, max_coord: int = 1000) -> int:
    """Convert normalized x coordinate to pixel x.

    Args:
        x: Normalized x coordinate
        width: Screen width in pixels
        max_coord: Maximum coordinate value the model outputs (default 1000 for 0-999 range)
                   Set to actual screenshot width for models that output pixel coordinates

    Returns:
        Pixel x coordinate
    """
    return round(x * width / max_coord)


def denorm_y(y: int, height: int, max_coord: int = 1000) -> int:
    """Convert normalized y coordinate to pixel y.

    Args:
        y: Normalized y coordinate
        height: Screen height in pixels
        max_coord: Maximum coordinate value the model outputs (default 1000 for 0-999 range)
                   Set to actual screenshot height for models that output pixel coordinates

    Returns:
        Pixel y coordinate
    """
    return round(y * height / max_coord)


def norm_x(px: int, width: int, max_coord: int = 1000) -> int:
    """Convert pixel x coordinate to model coordinate space."""
    return round(px * max_coord / width)


def norm_y(py: int, height: int, max_coord: int = 1000) -> int:
    """Convert pixel y coordinate to model coordinate space."""
    return round(py * max_coord / height)


class VNCController:
    """Controller for VNC desktop interactions.

    Handles screenshot capture, mouse operations, keyboard input, and scrolling
    via vncdotool. Automatically converts normalized coordinates (0-999) to
    pixels based on current screen size.
    """

    def __init__(
        self,
        coord_max: int | None = None,
        coord_max_x: int | None = None,
        coord_max_y: int | None = None,
        vnc_host: str = "vnc-desktop",
    ) -> None:
        """Initialize VNC controller (not yet connected).

        Args:
            coord_max: Back-compat alias for coord_max_x (and coord_max_y if not provided).
            coord_max_x: Maximum x coordinate value for denormalization.
                - 1000: For 0-999 normalized coordinates (recommended)
                - 1024: For Claude "computer use" coordinate space (common)
                - screen width: If your model returns absolute pixel coordinates
            coord_max_y: Maximum y coordinate value for denormalization (same conventions as coord_max_x).
        """
        self.client: Any = None
        self._screen_size: tuple[int, int] | None = None
        self.vnc_host = vnc_host

        # Back-compat: coord_max maps to x/y when explicit values not provided.
        if coord_max_x is None:
            coord_max_x = coord_max
        if coord_max_y is None:
            coord_max_y = coord_max

        # Default to 1000 (0-999 normalized coords) unless overridden via env vars.
        if coord_max_x is None:
            coord_max_x = int(os.getenv("VNC_COORD_MAX_X") or os.getenv("VNC_COORD_MAX") or "1000")
        if coord_max_y is None:
            coord_max_y = int(os.getenv("VNC_COORD_MAX_Y") or os.getenv("VNC_COORD_MAX") or "1000")

        self.coord_max_x = coord_max_x
        self.coord_max_y = coord_max_y

    def connect(self, server: str, password: str | None = None) -> "VNCController":
        """Connect to VNC server.

        Args:
            server: VNC server address (e.g., "localhost::5901" or "host:port")
            password: Optional VNC password

        Returns:
            Self for method chaining

        Raises:
            Exception: If connection fails
        """
        logger.info(f"Connecting to VNC server: {server}")
        self.client = vnc_api.connect(server, password=password)
        logger.info("VNC connection established")
        return self

    def disconnect(self) -> None:
        """Disconnect from VNC server."""
        if self.client:
            disconnect = getattr(self.client, "disconnect", None)
            if callable(disconnect):
                disconnect()
            self.client = None
            # vncdotool uses a background Twisted reactor thread. Ensure it is
            # stopped so CLI runs return to the shell prompt.
            try:
                vnc_api.shutdown()
            except Exception:
                pass
            logger.info("VNC connection closed")

    def screenshot_png(self) -> bytes:
        """Capture current screen as PNG bytes.

        Returns:
            PNG screenshot as bytes

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        # Capture to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.client.captureScreen(str(tmp_path))
            png_bytes = tmp_path.read_bytes()

            # Update cached screen size
            img = Image.open(io.BytesIO(png_bytes))
            self._screen_size = img.size
            logger.debug(f"Screenshot captured: {img.size}")

            return png_bytes
        finally:
            tmp_path.unlink(missing_ok=True)

    def get_screen_size(self) -> tuple[int, int]:
        """Get current screen dimensions.

        Returns:
            Tuple of (width, height) in pixels

        Raises:
            RuntimeError: If screen size not yet known (capture screenshot first)
        """
        if not self._screen_size:
            raise RuntimeError("Screen size unknown; capture screenshot first")
        return self._screen_size

    def move(self, x: int, y: int) -> None:
        """Move mouse pointer to pixel coordinates.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)
        self.client.mouseMove(x, y)
        logger.debug(f"Mouse moved to ({x}, {y})")

    def click(self, x: int, y: int, button: int = 1) -> None:
        """Click at pixel coordinates.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            button: Mouse button (1=left, 2=middle, 3=right)

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        # vncdotool clicks work correctly with accurate coordinates
        # Delays are important for reliable click registration
        self.client.mouseMove(x, y)
        time.sleep(0.1)  # Delay for click registration
        self.client.mouseDown(button)
        time.sleep(0.05)
        self.client.mouseUp(button)
        logger.debug(f"Clicked button {button} at ({x}, {y})")

    def double_click(self, x: int, y: int) -> None:
        """Double-click at pixel coordinates.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        self.client.mouseMove(x, y)
        self.client.mousePress(1)
        self.client.mousePress(1)
        logger.debug(f"Double-clicked at ({x}, {y})")

    def triple_click(self, x: int, y: int) -> None:
        """Triple-click at pixel coordinates.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        self.client.mouseMove(x, y)
        self.client.mousePress(1)
        time.sleep(0.01)  # 10ms delay between clicks (matches Anthropic)
        self.client.mousePress(1)
        time.sleep(0.01)
        self.client.mousePress(1)
        logger.debug(f"Triple-clicked at ({x}, {y})")

    def middle_click(self, x: int, y: int) -> None:
        """Middle-click at pixel coordinates.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        # Use button=2 for middle mouse button
        self.click(x, y, button=2)
        logger.debug(f"Middle-clicked at ({x}, {y})")

    def mouse_down(self, button: int = 1) -> None:
        """Press and hold mouse button at current position.

        Args:
            button: Mouse button (1=left, 2=middle, 3=right)

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        self.client.mouseDown(button)
        logger.debug(f"Mouse button {button} down")

    def mouse_up(self, button: int = 1) -> None:
        """Release mouse button at current position.

        Args:
            button: Mouse button (1=left, 2=middle, 3=right)

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        self.client.mouseUp(button)
        logger.debug(f"Mouse button {button} up")

    def get_cursor_position(self) -> tuple[int, int]:
        """Get current cursor position.

        Returns:
            Tuple of (x, y) pixel coordinates

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        # vncdotool client maintains position state
        return (self.client.x, self.client.y)

    def drag_and_drop(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Drag from one point to another.

        Args:
            x0: Start pixel x
            y0: Start pixel y
            x1: End pixel x
            y1: End pixel y

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        self.client.mouseMove(x0, y0)
        self.client.mouseDown(1)
        self.client.mouseDrag(x1, y1)
        self.client.mouseUp(1)
        logger.debug(f"Dragged from ({x0}, {y0}) to ({x1}, {y1})")

    def type_text(self, text: str, press_enter: bool = False, clear_first: bool = False) -> None:
        """Type text at current cursor position.

        Args:
            text: Text to type
            press_enter: Whether to press Enter after typing
            clear_first: Whether to clear existing text first (Ctrl+A, Delete)

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        if clear_first:
            self.client.keyPress("ctrl-a")
            self.client.keyPress("delete")

        # vncdotool handles string typing
        for char in text:
            self.client.keyPress(char)

        if press_enter:
            self.client.keyPress("enter")

        logger.debug(f"Typed text: {text[:50]}{'...' if len(text) > 50 else ''}")

    def key_combo(self, keys: str) -> None:
        """Execute keyboard shortcut.

        Args:
            keys: Key combination (e.g., "Return", "ctrl+c", "alt+tab")
                  Supports Anthropic's Computer Use API key names

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        # Normalize Anthropic key names to vncdotool key names
        normalized = keys
        for anthropic_key, vnc_key in ANTHROPIC_TO_VNCDOTOOL_KEYS.items():
            normalized = normalized.replace(anthropic_key, vnc_key)

        # vncdotool accepts "ctrl-a" format; normalize separators and modifiers
        normalized = normalized.replace("+", "-").replace("control", "ctrl")

        self.client.keyPress(normalized)
        logger.debug(f"Pressed key: {keys} -> {normalized}")

    def hold_key(self, key: str, duration: float) -> None:
        """Hold a key down for a specified duration.

        Args:
            key: Key to hold (e.g., "shift", "ctrl", "a")
            duration: Time to hold key in seconds (0-100)

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        # Normalize key name (control -> ctrl)
        normalized = key.replace("control", "ctrl")

        # Press and hold key
        self.client.keyDown(normalized)
        time.sleep(duration)
        self.client.keyUp(normalized)

        logger.debug(f"Held key '{key}' for {duration} seconds")

    def scroll(
        self,
        direction: Literal["up", "down", "left", "right"],
        magnitude: int = 800,
    ) -> None:
        """Scroll in a direction.

        Uses PageUp/PageDown/Arrow keys with magnitude-based repetition.

        Args:
            direction: Scroll direction
            magnitude: Scroll distance (divided by 400 for repetitions)

        Raises:
            RuntimeError: If not connected
        """
        if not self.client:
            raise RuntimeError(NOT_CONNECTED_ERROR)

        # Map direction to key
        key_map = {
            "up": "pgup",
            "down": "pgdn",
            "left": "left",
            "right": "right",
        }
        key = key_map[direction]

        # Repeat based on magnitude (heuristic: 400 pixels per press)
        repeats = max(1, magnitude // 400)
        for _ in range(repeats):
            self.client.keyPress(key)

        logger.debug(f"Scrolled {direction} with magnitude {magnitude} ({repeats} repeats)")

    def _action_click_at(self, args: dict, width: int, height: int) -> None:
        """Handle click_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        self.click(px, py)

    def _action_double_click_at(self, args: dict, width: int, height: int) -> None:
        """Handle double_click_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        self.double_click(px, py)

    def _action_right_click_at(self, args: dict, width: int, height: int) -> None:
        """Handle right_click_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        self.click(px, py, button=3)

    def _action_triple_click_at(self, args: dict, width: int, height: int) -> None:
        """Handle triple_click_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        self.triple_click(px, py)

    def _action_middle_click_at(self, args: dict, width: int, height: int) -> None:
        """Handle middle_click_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        self.middle_click(px, py)

    def _action_left_mouse_down(self, args: dict, width: int, height: int) -> None:
        """Handle left_mouse_down action."""
        self.mouse_down(button=1)

    def _action_left_mouse_up(self, args: dict, width: int, height: int) -> None:
        """Handle left_mouse_up action."""
        self.mouse_up(button=1)

    def _action_hover_at(self, args: dict, width: int, height: int) -> None:
        """Handle hover_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        self.move(px, py)

    def _action_type_text_at(self, args: dict, width: int, height: int) -> None:
        """Handle type_text_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        self.click(px, py)  # Focus first
        self.type_text(
            args["text"],
            press_enter=args.get("press_enter", False),
            clear_first=args.get("clear_before_typing", False),
        )

    def _action_type_text(self, args: dict, width: int, height: int) -> None:
        """Handle type_text action."""
        self.type_text(
            args["text"],
            press_enter=args.get("press_enter", False),
            clear_first=args.get("clear_before_typing", False),
        )

    def _action_key_combination(self, args: dict, width: int, height: int) -> None:
        """Handle key_combination action."""
        self.key_combo(args["keys"])

    def _action_hold_key(self, args: dict, width: int, height: int) -> None:
        """Handle hold_key action."""
        self.hold_key(args["key"], args["duration"])

    def _action_scroll_document(self, args: dict, width: int, height: int) -> None:
        """Handle scroll_document action."""
        self.scroll(args["direction"], args.get("magnitude", 800))

    def _action_scroll_at(self, args: dict, width: int, height: int) -> None:
        """Handle scroll_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        self.move(px, py)
        self.scroll(args["direction"], args.get("magnitude", 800))

    def _action_drag_and_drop(self, args: dict, width: int, height: int) -> None:
        """Handle drag_and_drop action.

        Accepts both schema names (x, y, destination_x, destination_y) and
        native planner names (start_x, start_y, end_x, end_y).

        Raises:
            ValueError: If required coordinates are missing
        """
        # Support both naming conventions
        start_x = args.get("start_x") or args.get("x")
        start_y = args.get("start_y") or args.get("y")
        end_x = args.get("end_x") or args.get("destination_x")
        end_y = args.get("end_y") or args.get("destination_y")

        if start_x is None or start_y is None or end_x is None or end_y is None:
            raise ValueError(
                "drag_and_drop requires start coordinates (start_x/x, start_y/y) "
                "and end coordinates (end_x/destination_x, end_y/destination_y)"
            )

        y0 = denorm_y(int(start_y), height, self.coord_max_y)
        x0 = denorm_x(int(start_x), width, self.coord_max_x)
        x1 = denorm_x(int(end_x), width, self.coord_max_x)
        y1 = denorm_y(int(end_y), height, self.coord_max_y)
        self.drag_and_drop(x0, y0, x1, y1)

    def _action_wait_5_seconds(self, args: dict, width: int, height: int) -> None:
        """Handle wait_5_seconds action."""
        logger.info("Waiting 5 seconds...")
        time.sleep(5)

    def _get_action_handlers(self) -> dict:
        """Get mapping of action names to handler methods."""
        return {
            "click_at": self._action_click_at,
            "double_click_at": self._action_double_click_at,
            "right_click_at": self._action_right_click_at,
            "triple_click_at": self._action_triple_click_at,
            "middle_click_at": self._action_middle_click_at,
            "left_mouse_down": self._action_left_mouse_down,
            "left_mouse_up": self._action_left_mouse_up,
            "hover_at": self._action_hover_at,
            "type_text_at": self._action_type_text_at,
            "type_text": self._action_type_text,
            "key_combination": self._action_key_combination,
            "hold_key": self._action_hold_key,
            "scroll_document": self._action_scroll_document,
            "scroll_at": self._action_scroll_at,
            "drag_and_drop": self._action_drag_and_drop,
            "wait_5_seconds": self._action_wait_5_seconds,
        }

    def execute_action(
        self,
        action_name: str,
        args: dict,
    ) -> ActionResult:
        """Execute a Computer Use action and capture result.

        Args:
            action_name: Name of action to execute
            args: Action arguments (with normalized coordinates if applicable)

        Returns:
            ActionResult with screenshot and execution status
        """
        try:
            width, height = self.get_screen_size()

            # Special case: cursor_position returns early with output
            if action_name == "cursor_position":
                cursor_x, cursor_y = self.get_cursor_position()
                api_x = norm_x(cursor_x, width, self.coord_max_x)
                api_y = norm_y(cursor_y, height, self.coord_max_y)
                screenshot = self.screenshot_png()
                return ActionResult(
                    success=True,
                    error=None,
                    screenshot_png=screenshot,
                    url="",
                    output=f"X={api_x},Y={api_y}",
                )

            # Dispatch to appropriate handler
            handlers = self._get_action_handlers()
            handler = handlers.get(action_name)

            if handler is None:
                raise ValueError(f"Unknown action: {action_name}")

            handler(args, width, height)

            # Capture screenshot after action
            screenshot = self.screenshot_png()

            return ActionResult(
                success=True,
                error=None,
                screenshot_png=screenshot,
                url="",  # VNC desktop has no URL
            )

        except Exception as e:
            logger.error(f"Action {action_name} failed: {e}")
            # Still try to capture screenshot for debugging
            try:
                screenshot = self.screenshot_png()
            except Exception:
                screenshot = b""

            return ActionResult(
                success=False,
                error=str(e),
                screenshot_png=screenshot,
                url="",
            )
