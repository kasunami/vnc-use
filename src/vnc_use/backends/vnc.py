"""VNC backend controller for executing UI actions."""

import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageEnhance, ImageOps
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


def _parse_crop(value: str, full_width: int, full_height: int) -> tuple[int, int, int, int] | None:
    """Parse VNC_SCREEN_CROP as x,y,w,h and clamp it to the screenshot."""
    raw = value.strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("VNC_SCREEN_CROP must be formatted as x,y,width,height")
    x, y, width, height = [int(part) for part in parts]
    if width <= 0 or height <= 0:
        raise ValueError("VNC_SCREEN_CROP width/height must be positive")
    x = max(0, min(x, full_width - 1))
    y = max(0, min(y, full_height - 1))
    width = max(1, min(width, full_width - x))
    height = max(1, min(height, full_height - y))
    return x, y, width, height


def _normalize_ocr_text(value: str) -> str:
    """Normalize OCR text for loose UI-label matching."""
    return " ".join(value.casefold().split())


def _bbox_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Return the bounding box covering all input boxes."""
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return left, top, right - left, bottom - top


def _prepare_ocr_image(png_bytes: bytes) -> tuple[bytes, float]:
    """Prepare a screenshot for OCR and return image bytes plus scale factor."""
    image = Image.open(io.BytesIO(png_bytes)).convert("L")
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(float(os.getenv("VNC_OCR_CONTRAST", "2.0")))
    scale = float(os.getenv("VNC_OCR_SCALE", "2.0"))
    if scale > 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue(), scale


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
        self._crop_offset: tuple[int, int] = (0, 0)
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
            # Do not shut down vncdotool's global Twisted reactor by default.
            # The reactor cannot be restarted in a long-running MCP server, so
            # calling shutdown after one task breaks the next task with
            # ReactorNotRestartable. CLI users that need explicit shutdown can
            # opt in with VNC_SHUTDOWN_REACTOR_ON_DISCONNECT=1.
            if os.getenv("VNC_SHUTDOWN_REACTOR_ON_DISCONNECT", "").lower() in {"1", "true", "yes"}:
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

            # Update cached screen size. If VNC_SCREEN_CROP is set, expose only
            # that region to the model and remap future coordinate actions back
            # into full-desktop coordinates.
            img = Image.open(io.BytesIO(png_bytes))
            crop = _parse_crop(os.getenv("VNC_SCREEN_CROP", ""), img.width, img.height)
            if crop:
                x, y, width, height = crop
                img = img.crop((x, y, x + width, y + height))
                out = io.BytesIO()
                img.save(out, format="PNG")
                png_bytes = out.getvalue()
                self._crop_offset = (x, y)
                self._screen_size = (width, height)
                logger.debug(f"Screenshot captured and cropped: full={img.size}, crop={crop}")
            else:
                self._crop_offset = (0, 0)
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
        px += self._crop_offset[0]
        py += self._crop_offset[1]
        self.click(px, py)

    def _find_text_bbox(
        self,
        label: str,
        match_mode: str = "contains",
        occurrence: int = 1,
        region: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int, int, int] | None:
        """Find a visible text label with OCR and return a crop-relative bbox.

        The implementation intentionally uses the system `tesseract` binary when
        available and otherwise returns None. This keeps OCR optional while still
        making unavailable text matching explicit to callers.
        """
        if not shutil.which("tesseract"):
            return None

        target = _normalize_ocr_text(label)
        if not target:
            return None

        screenshot, scale = _prepare_ocr_image(self.screenshot_png())
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp_path.write_bytes(screenshot)

        try:
            proc = None
            for psm in [part.strip() for part in os.getenv("VNC_OCR_PSMS", "6,11").split(",") if part.strip()]:
                proc = subprocess.run(
                    ["tesseract", str(tmp_path), "stdout", "--psm", psm, "tsv"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(os.getenv("VNC_OCR_TIMEOUT_S", "5")),
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    break
        finally:
            tmp_path.unlink(missing_ok=True)

        if proc is None:
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            logger.debug("tesseract OCR failed: %s", proc.stderr.strip())
            return None

        lines: dict[tuple[str, str, str], list[tuple[str, tuple[int, int, int, int]]]] = {}
        header_seen = False
        for raw_line in proc.stdout.splitlines():
            parts = raw_line.split("\t")
            if not header_seen:
                header_seen = True
                continue
            if len(parts) < 12:
                continue
            text = parts[11].strip()
            if not text:
                continue
            try:
                conf = float(parts[10])
                if conf < float(os.getenv("VNC_OCR_MIN_CONF", "35")):
                    continue
                raw_left, raw_top, raw_width, raw_height = map(int, parts[6:10])
                left = round(raw_left / scale)
                top = round(raw_top / scale)
                box_width = round(raw_width / scale)
                box_height = round(raw_height / scale)
            except ValueError:
                continue
            key = (parts[2], parts[3], parts[4])
            lines.setdefault(key, []).append((text, (left, top, box_width, box_height)))

        matches: list[tuple[int, int, int, int]] = []
        for words in lines.values():
            normalized_words = [_normalize_ocr_text(word) for word, _box in words]
            for start in range(len(words)):
                for end in range(start + 1, len(words) + 1):
                    candidate = _normalize_ocr_text(" ".join(normalized_words[start:end]))
                    is_match = candidate == target if match_mode == "exact" else target in candidate
                    if is_match:
                        bbox = _bbox_union([box for _word, box in words[start:end]])
                        if region is not None:
                            box_center_x = bbox[0] + round(bbox[2] / 2)
                            box_center_y = bbox[1] + round(bbox[3] / 2)
                            rx, ry, rw, rh = region
                            if not (rx <= box_center_x <= rx + rw and ry <= box_center_y <= ry + rh):
                                continue
                        matches.append(bbox)
                        break

        if occurrence < 1:
            occurrence = 1
        if len(matches) < occurrence:
            return None
        return matches[occurrence - 1]

    def _normalized_region_to_pixels(
        self,
        value: Any,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        """Convert optional normalized region [x,y,w,h] to crop-relative pixels."""
        if value is None:
            return None
        if not isinstance(value, list | tuple) or len(value) != 4:
            raise ValueError("click_text_or_button region must be [x, y, width, height]")
        x, y, region_width, region_height = [int(part) for part in value]
        px = denorm_x(x, width, self.coord_max_x)
        py = denorm_y(y, height, self.coord_max_y)
        pw = denorm_x(region_width, width, self.coord_max_x)
        ph = denorm_y(region_height, height, self.coord_max_y)
        if pw <= 0 or ph <= 0:
            raise ValueError("click_text_or_button region width/height must be positive")
        return px, py, pw, ph

    def _action_click_text_or_button(self, args: dict, width: int, height: int) -> None:
        """Click the center of visible text/button label using OCR or fallback coords."""
        label = str(args.get("label") or "").strip()
        if not label:
            raise ValueError("click_text_or_button requires a non-empty label")

        bbox = self._find_text_bbox(
            label=label,
            match_mode=str(args.get("match_mode") or "contains"),
            occurrence=int(args.get("occurrence") or 1),
            region=self._normalized_region_to_pixels(args.get("region"), width, height),
        )
        if bbox is not None:
            left, top, box_width, box_height = bbox
            px = self._crop_offset[0] + left + round(box_width / 2)
            py = self._crop_offset[1] + top + round(box_height / 2)
            self.click(px, py)
            return

        if args.get("x") is not None and args.get("y") is not None:
            px = denorm_x(int(args["x"]), width, self.coord_max_x) + self._crop_offset[0]
            py = denorm_y(int(args["y"]), height, self.coord_max_y) + self._crop_offset[1]
            logger.info("OCR target %r not found; using fallback coordinates", label)
            self.click(px, py)
            return

        raise ValueError(
            f"Could not find visible text/button label {label!r}. "
            "Install tesseract OCR or provide fallback x/y coordinates."
        )

    def _action_double_click_at(self, args: dict, width: int, height: int) -> None:
        """Handle double_click_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        px += self._crop_offset[0]
        py += self._crop_offset[1]
        self.double_click(px, py)

    def _action_right_click_at(self, args: dict, width: int, height: int) -> None:
        """Handle right_click_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        px += self._crop_offset[0]
        py += self._crop_offset[1]
        self.click(px, py, button=3)

    def _action_triple_click_at(self, args: dict, width: int, height: int) -> None:
        """Handle triple_click_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        px += self._crop_offset[0]
        py += self._crop_offset[1]
        self.triple_click(px, py)

    def _action_middle_click_at(self, args: dict, width: int, height: int) -> None:
        """Handle middle_click_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        px += self._crop_offset[0]
        py += self._crop_offset[1]
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
        px += self._crop_offset[0]
        py += self._crop_offset[1]
        self.move(px, py)

    def _action_type_text_at(self, args: dict, width: int, height: int) -> None:
        """Handle type_text_at action."""
        px = denorm_x(int(args["x"]), width, self.coord_max_x)
        py = denorm_y(int(args["y"]), height, self.coord_max_y)
        px += self._crop_offset[0]
        py += self._crop_offset[1]
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
        keys = args.get("keys")
        if isinstance(keys, list):
            keys = "+".join(str(k).strip() for k in keys if str(k).strip())
        self.key_combo(str(keys or ""))

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
        px += self._crop_offset[0]
        py += self._crop_offset[1]
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
        x_offset, y_offset = self._crop_offset
        x0 += x_offset
        x1 += x_offset
        y0 += y_offset
        y1 += y_offset
        self.drag_and_drop(x0, y0, x1, y1)

    def _action_wait_5_seconds(self, args: dict, width: int, height: int) -> None:
        """Handle wait_5_seconds action."""
        logger.info("Waiting 5 seconds...")
        time.sleep(5)

    def _action_open_web_browser(self, args: dict, width: int, height: int) -> None:
        """Open Chromium using the desktop run dialog instead of fragile icon coordinates."""
        _ = args, width, height
        self.key_combo("alt+f2")
        time.sleep(0.5)
        self.type_text(os.getenv("VNC_BROWSER_COMMAND", "chromium"), press_enter=True, clear_first=True)
        time.sleep(3)

    def _action_navigate(self, args: dict, width: int, height: int) -> None:
        """Navigate the focused browser by using the address bar."""
        _ = width, height
        url = str(args.get("url") or "").strip()
        if not url:
            raise ValueError("navigate requires a non-empty url")
        self.key_combo("ctrl+l")
        time.sleep(0.2)
        self.type_text(url, press_enter=True, clear_first=True)
        time.sleep(3)

    def _get_action_handlers(self) -> dict:
        """Get mapping of action names to handler methods."""
        return {
            "open_web_browser": self._action_open_web_browser,
            "navigate": self._action_navigate,
            "click_at": self._action_click_at,
            "click_text_or_button": self._action_click_text_or_button,
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
