"""Native Anthropic Computer Use API planner for VNC desktop control.

Uses Anthropic's native computer_20250124 tool type for better accuracy,
leveraging Claude's specialized training for computer control tasks.
"""

import base64
import io
import logging
import os
from typing import Any

from anthropic import Anthropic
from PIL import Image

from .base import BasePlanner
from .utils import compress_screenshot

logger = logging.getLogger(__name__)

# Default model for native computer use
DEFAULT_MODEL = "claude-haiku-4-5"


class NativeComputerPlanner(BasePlanner):
    """Native Anthropic Computer Use API planner for VNC desktop control.

    Uses Anthropic's computer_20250124 tool type which provides:
    - Specialized training for computer control tasks
    - Better coordinate accuracy from vision
    - Single unified tool interface
    - Transparent coordinate scaling
    """

    def __init__(
        self,
        excluded_actions: list[str] | None = None,
        model: str | None = None,
        api_key: str | None = None,
        display_width: int = 1024,
        display_height: int = 768,
    ) -> None:
        """Initialize native computer use planner.

        Args:
            excluded_actions: List of action names to exclude (not used with native API)
            model: Model name (defaults to claude-haiku-4-5)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            display_width: Native VNC display width in pixels
            display_height: Native VNC display height in pixels
        """
        self.excluded_actions = excluded_actions or []
        self.model: str = model or os.getenv("COMPUTER_USE_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable must be set for native computer use"
            )

        # Native display dimensions
        self.display_width = display_width
        self.display_height = display_height

        # API dimensions (scaled for Claude)
        # Following Anthropic's recommendation: scale to XGA (1024x768) or similar
        self.api_width = 1024
        self.api_height = 768

        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.api_key)

        logger.info(f"Initialized native computer use planner with model: {self.model}")
        logger.info(f"Display dimensions: {self.display_width}x{self.display_height} (native)")
        logger.info(f"API dimensions: {self.api_width}x{self.api_height} (for Claude)")

    def _scale_screenshot(self, screenshot_png: bytes) -> bytes:
        """Scale screenshot from native resolution to API resolution.

        Args:
            screenshot_png: Screenshot at native resolution

        Returns:
            Screenshot scaled to API resolution (1024x768)
        """
        # Use existing compression logic
        return compress_screenshot(screenshot_png, max_width=self.api_width)

    def generate_stateless(
        self,
        task: str,
        action_history: list[str],
        screenshot_png: bytes,
    ) -> Any:
        """Generate model response with action proposals.

        Args:
            task: User's task description
            action_history: List of text descriptions of past actions
            screenshot_png: Current screenshot as PNG bytes

        Returns:
            Anthropic Message object with tool_use blocks
        """
        # Scale screenshot to API resolution
        scaled_screenshot = self._scale_screenshot(screenshot_png)
        screenshot_b64 = base64.b64encode(scaled_screenshot).decode("utf-8")

        # Get scaled dimensions
        img = Image.open(io.BytesIO(scaled_screenshot))
        scaled_width, scaled_height = img.size

        logger.info(
            f"Screenshot scaled: {self.display_width}x{self.display_height} -> {scaled_width}x{scaled_height}"
        )

        # Build system prompt
        system_prompt = f"""You are controlling a computer via VNC (Virtual Network Computing).

<IMPORTANT>
You are using the native computer control tool which has been specifically designed for desktop automation.
The display resolution is {self.display_width}x{self.display_height} pixels.
</IMPORTANT>

Current task: {task}

REASONING GUIDELINES:
1. Review your action history - are you repeating the same action without progress?
2. If you've tried the same action 3+ times with identical observations, that approach is NOT working
3. Compare the current screenshot to your last observation - did your action work?
4. If the task appears complete, you can stop (avoid unnecessary actions)
5. When stuck, try alternative methods (keyboard shortcuts, different click locations, etc.)
"""

        if action_history:
            system_prompt += "\n\nActions taken so far:\n" + "\n".join(
                f"{i + 1}. {action}" for i, action in enumerate(action_history[-10:])
            )

        # Build user message
        user_content = [
            {
                "type": "text",
                "text": """Here is the current desktop screenshot.

Analyze:
- Did my last action achieve its intended effect?
- Am I repeating an action that isn't working?
- Is the task already complete?

What action should I take next?""",
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            },
        ]

        # Construct API call
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=1024,
                betas=["computer-use-2025-01-24"],
                tools=[
                    {
                        "type": "computer_20250124",
                        "name": "computer",
                        "display_width_px": scaled_width,
                        "display_height_px": scaled_height,
                        "display_number": 1,
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
                system=system_prompt,
            )

            logger.info(f"Received response with {len(response.content)} content block(s)")

            # Count tool uses
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            logger.info(f"Found {len(tool_uses)} tool use block(s)")

            if len(tool_uses) == 0:
                logger.warning(f"No tool uses in response. Content: {response.content[:500]}")

            return response

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")

            # Return a minimal response object to allow graceful degradation
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.content = [{"type": "text", "text": f"Error: {error_msg}"}]
                    self.stop_reason = "error"

            return ErrorResponse(str(e))

    def extract_text(self, response: Any) -> str:
        """Extract text observations/reasoning from model response.

        Args:
            response: Anthropic Message object

        Returns:
            Text content from model (empty string if none)
        """
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return " ".join(text_parts).strip()

    def extract_function_calls(self, response: Any) -> list[dict[str, Any]]:
        """Extract function calls from model response.

        Maps Anthropic's computer tool actions to our VNC action format.

        Args:
            response: Anthropic Message object

        Returns:
            List of dicts with 'name' and 'args' keys compatible with VNCController
        """
        function_calls = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name != "computer":
                logger.warning(f"Unexpected tool name: {block.name}")
                continue

            action = block.input.get("action")
            if not action:
                logger.warning(f"Tool use block missing 'action' field: {block.input}")
                continue

            result = self._process_action(action, block.input)
            if result:
                function_calls.append(result)

        return function_calls

    def _process_action(self, action: str, input_data: dict) -> dict[str, Any] | None:
        """Process a single action and return the corresponding VNC function call.

        Args:
            action: Action type from Anthropic API
            input_data: Input data for the action

        Returns:
            Function call dict or None if action should be skipped
        """
        # Action handlers dispatch table
        handlers = {
            "left_click": self._handle_coordinate_action,
            "double_click": self._handle_coordinate_action,
            "right_click": self._handle_coordinate_action,
            "triple_click": self._handle_coordinate_action,
            "middle_click": self._handle_coordinate_action,
            "mouse_move": self._handle_coordinate_action,
            "left_mouse_down": self._handle_simple_action,
            "left_mouse_up": self._handle_simple_action,
            "cursor_position": self._handle_simple_action,
            "wait": self._handle_simple_action,
            "type": self._handle_text_action,
            "key": self._handle_text_action,
            "left_click_drag": self._handle_drag_action,
            "scroll": self._handle_scroll_action,
            "hold_key": self._handle_hold_key_action,
            "screenshot": self._handle_screenshot_action,
        }

        handler = handlers.get(action)
        if handler:
            return handler(action, input_data)

        logger.warning(f"Unknown action type: {action}")
        return None

    def _handle_coordinate_action(self, action: str, input_data: dict) -> dict[str, Any] | None:
        """Handle actions that require coordinates (clicks, moves)."""
        action_map = {
            "left_click": "click_at",
            "double_click": "double_click_at",
            "right_click": "right_click_at",
            "triple_click": "triple_click_at",
            "middle_click": "middle_click_at",
            "mouse_move": "hover_at",
        }

        coordinate = input_data.get("coordinate")
        if not coordinate or len(coordinate) != 2:
            logger.warning(f"Invalid coordinate for {action}: {coordinate}")
            return None

        x, y = coordinate
        vnc_action = action_map[action]
        logger.debug(f"{action}({x},{y}) -> {vnc_action}({x},{y})")
        return {"name": vnc_action, "args": {"x": x, "y": y}}

    def _handle_simple_action(self, action: str, _input_data: dict) -> dict[str, Any] | None:
        """Handle simple actions with no parameters."""
        action_map = {
            "left_mouse_down": "left_mouse_down",
            "left_mouse_up": "left_mouse_up",
            "cursor_position": "cursor_position",
            "wait": "wait_5_seconds",
        }

        vnc_action = action_map[action]
        logger.debug(f"{action} -> {vnc_action}")
        return {"name": vnc_action, "args": {}}

    def _handle_text_action(self, action: str, input_data: dict) -> dict[str, Any] | None:
        """Handle actions that require text input (type, key)."""
        text = input_data.get("text")
        if not text:
            logger.warning(f"{action} action missing 'text' field")
            return None

        if action == "type":
            logger.debug(f"type -> type_text('{text[:50]}...')")
            return {"name": "type_text", "args": {"text": text}}

        # action == "key"
        logger.debug(f"key -> key_combination('{text}')")
        return {"name": "key_combination", "args": {"keys": text}}

    def _handle_drag_action(self, _action: str, input_data: dict) -> dict[str, Any] | None:
        """Handle left_click_drag action."""
        start = input_data.get("start_coordinate")
        end = input_data.get("end_coordinate")

        if not start or len(start) != 2 or not end or len(end) != 2:
            logger.warning(f"Invalid coordinates for left_click_drag: {start}, {end}")
            return None

        start_x, start_y = start
        end_x, end_y = end
        logger.debug(f"left_click_drag({start_x},{start_y})-({end_x},{end_y}) -> drag_and_drop")
        return {
            "name": "drag_and_drop",
            "args": {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            },
        }

    def _handle_scroll_action(self, _action: str, input_data: dict) -> dict[str, Any] | None:
        """Handle scroll action."""
        scroll_direction = input_data.get("scroll_direction")
        if not scroll_direction:
            logger.warning("scroll action missing 'scroll_direction' field")
            return None

        coordinate = input_data.get("coordinate")
        scroll_amount = input_data.get("scroll_amount", 5)

        # Convert scroll_amount (clicks) to magnitude (pixels)
        # Anthropic uses scroll buttons (1 click ≈ 160 pixels)
        magnitude = scroll_amount * 160

        if coordinate and len(coordinate) == 2:
            x, y = coordinate
            logger.debug(f"scroll({scroll_direction}, {scroll_amount}) at ({x},{y}) -> scroll_at")
            return {
                "name": "scroll_at",
                "args": {
                    "x": x,
                    "y": y,
                    "direction": scroll_direction,
                    "magnitude": magnitude,
                },
            }

        logger.debug(f"scroll({scroll_direction}, {scroll_amount}) -> scroll_document")
        return {
            "name": "scroll_document",
            "args": {"direction": scroll_direction, "magnitude": magnitude},
        }

    def _handle_hold_key_action(self, _action: str, input_data: dict) -> dict[str, Any] | None:
        """Handle hold_key action."""
        text = input_data.get("text")
        duration = input_data.get("duration")

        if not text:
            logger.warning("hold_key action missing 'text' field")
            return None

        if duration is None or not isinstance(duration, (int, float)):
            logger.warning(f"hold_key action invalid duration: {duration}")
            return None

        if duration < 0 or duration > 100:
            logger.warning(f"hold_key duration out of range (0-100): {duration}")
            return None

        logger.debug(f"hold_key({text}, {duration}s) -> hold_key")
        return {"name": "hold_key", "args": {"key": text, "duration": duration}}

    def _handle_screenshot_action(self, _action: str, _input_data: dict) -> dict[str, Any] | None:
        """Handle screenshot action (ignored, handled by agent loop)."""
        logger.debug("Ignoring screenshot action (handled by agent loop)")
        return None

    def _scale_coord_to_native(self, coord: int, api_dimension: int, native_dimension: int) -> int:
        """Scale a coordinate from API resolution to native resolution.

        Args:
            coord: Coordinate value at API resolution
            api_dimension: API resolution dimension (width or height)
            native_dimension: Native resolution dimension (width or height)

        Returns:
            Scaled coordinate at native resolution
        """
        scaling_factor = native_dimension / api_dimension
        return round(coord * scaling_factor)

    def extract_safety_decision(self, response: Any) -> dict[str, Any] | None:
        """Extract safety decision from model response if present.

        Uses Anthropic's official stop_reason field for structured refusal detection.
        No string pattern matching - only API-provided signals.

        Args:
            response: Anthropic Message object

        Returns:
            Safety decision dict or None
        """
        # Use Anthropic's official stop_reason field for refusal detection
        if hasattr(response, "stop_reason"):
            stop_reason = response.stop_reason

            # Anthropic API returns "refusal" when streaming classifiers intervene
            # for potential policy violations (official documented behavior)
            if stop_reason == "refusal":
                text = self.extract_text(response)
                logger.warning(f"Model refused via stop_reason='refusal': {text[:100]}")
                return {"action": "block", "reason": f"Model refused: {text[:200]}"}

        # No refusal - let the agent continue
        return None
