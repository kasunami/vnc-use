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
        self.model = model or os.getenv("COMPUTER_USE_MODEL", DEFAULT_MODEL)
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

    def extract_function_calls(self, response: Any) -> list[dict[str, Any]]:  # noqa: PLR0912, PLR0915
        """Extract function calls from model response.

        Maps Anthropic's computer tool actions to our VNC action format:
        - left_click -> click_at
        - type -> type_text
        - key -> press_key
        - screenshot -> (handled separately, not an action)
        - mouse_move -> hover_at
        - left_click_drag -> drag_and_drop
        - right_click -> right_click_at
        - double_click -> double_click_at

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

            # Map Anthropic's computer actions to VNC actions
            if action == "left_click":
                coordinate = block.input.get("coordinate")
                if not coordinate or len(coordinate) != 2:
                    logger.warning(f"Invalid coordinate for left_click: {coordinate}")
                    continue

                x, y = coordinate
                # Pass API coordinates directly - VNCController will denormalize them
                function_calls.append({"name": "click_at", "args": {"x": x, "y": y}})
                logger.debug(f"left_click({x},{y}) -> click_at({x},{y})")

            elif action == "double_click":
                coordinate = block.input.get("coordinate")
                if not coordinate or len(coordinate) != 2:
                    logger.warning(f"Invalid coordinate for double_click: {coordinate}")
                    continue

                x, y = coordinate
                # Pass API coordinates directly - VNCController will denormalize them
                function_calls.append({"name": "double_click_at", "args": {"x": x, "y": y}})
                logger.debug(f"double_click({x},{y}) -> double_click_at({x},{y})")

            elif action == "right_click":
                coordinate = block.input.get("coordinate")
                if not coordinate or len(coordinate) != 2:
                    logger.warning(f"Invalid coordinate for right_click: {coordinate}")
                    continue

                x, y = coordinate
                # Pass API coordinates directly - VNCController will denormalize them
                function_calls.append({"name": "right_click_at", "args": {"x": x, "y": y}})
                logger.debug(f"right_click({x},{y}) -> right_click_at({x},{y})")

            elif action == "triple_click":
                coordinate = block.input.get("coordinate")
                if not coordinate or len(coordinate) != 2:
                    logger.warning(f"Invalid coordinate for triple_click: {coordinate}")
                    continue

                x, y = coordinate
                function_calls.append({"name": "triple_click_at", "args": {"x": x, "y": y}})
                logger.debug(f"triple_click({x},{y}) -> triple_click_at({x},{y})")

            elif action == "middle_click":
                coordinate = block.input.get("coordinate")
                if not coordinate or len(coordinate) != 2:
                    logger.warning(f"Invalid coordinate for middle_click: {coordinate}")
                    continue

                x, y = coordinate
                function_calls.append({"name": "middle_click_at", "args": {"x": x, "y": y}})
                logger.debug(f"middle_click({x},{y}) -> middle_click_at({x},{y})")

            elif action == "left_mouse_down":
                # Press and hold left mouse button
                function_calls.append({"name": "left_mouse_down", "args": {}})
                logger.debug("left_mouse_down -> left_mouse_down")

            elif action == "left_mouse_up":
                # Release left mouse button
                function_calls.append({"name": "left_mouse_up", "args": {}})
                logger.debug("left_mouse_up -> left_mouse_up")

            elif action == "cursor_position":
                # Get current cursor position
                function_calls.append({"name": "cursor_position", "args": {}})
                logger.debug("cursor_position -> cursor_position")

            elif action == "type":
                text = block.input.get("text")
                if not text:
                    logger.warning("type action missing 'text' field")
                    continue

                function_calls.append({"name": "type_text", "args": {"text": text}})
                logger.debug(f"type -> type_text('{text[:50]}...')")

            elif action == "key":
                text = block.input.get("text")
                if not text:
                    logger.warning("key action missing 'text' field")
                    continue

                # Anthropic uses special key names like "Return", "Escape", etc.
                # Map to our key_combination format
                function_calls.append({"name": "key_combination", "args": {"keys": text}})
                logger.debug(f"key -> key_combination('{text}')")

            elif action == "mouse_move":
                coordinate = block.input.get("coordinate")
                if not coordinate or len(coordinate) != 2:
                    logger.warning(f"Invalid coordinate for mouse_move: {coordinate}")
                    continue

                x, y = coordinate
                # Pass API coordinates directly - VNCController will denormalize them
                function_calls.append({"name": "hover_at", "args": {"x": x, "y": y}})
                logger.debug(f"mouse_move({x},{y}) -> hover_at({x},{y})")

            elif action == "left_click_drag":
                start = block.input.get("start_coordinate")
                end = block.input.get("end_coordinate")

                if not start or len(start) != 2 or not end or len(end) != 2:
                    logger.warning(f"Invalid coordinates for left_click_drag: {start}, {end}")
                    continue

                start_x, start_y = start
                end_x, end_y = end

                # Pass API coordinates directly - VNCController will denormalize them
                function_calls.append(
                    {
                        "name": "drag_and_drop",
                        "args": {
                            "start_x": start_x,
                            "start_y": start_y,
                            "end_x": end_x,
                            "end_y": end_y,
                        },
                    }
                )
                logger.debug(
                    f"left_click_drag({start_x},{start_y})-({end_x},{end_y}) -> drag_and_drop"
                )

            elif action == "screenshot":
                # Screenshots are handled automatically by the agent loop
                logger.debug("Ignoring screenshot action (handled by agent loop)")
                continue

            elif action == "wait":
                # Map Claude's "wait" action to our wait_5_seconds action
                function_calls.append({"name": "wait_5_seconds", "args": {}})
                logger.debug("wait -> wait_5_seconds")

            elif action == "scroll":
                scroll_direction = block.input.get("scroll_direction")
                coordinate = block.input.get("coordinate")
                scroll_amount = block.input.get("scroll_amount", 5)

                if not scroll_direction:
                    logger.warning("scroll action missing 'scroll_direction' field")
                    continue

                # Convert scroll_amount (clicks) to magnitude (pixels)
                # Anthropic uses scroll buttons (1 click ≈ 160 pixels)
                magnitude = scroll_amount * 160

                if coordinate and len(coordinate) == 2:
                    # Scroll at specific location
                    x, y = coordinate
                    function_calls.append(
                        {
                            "name": "scroll_at",
                            "args": {
                                "x": x,
                                "y": y,
                                "direction": scroll_direction,
                                "magnitude": magnitude,
                            },
                        }
                    )
                    logger.debug(
                        f"scroll({scroll_direction}, {scroll_amount}) at ({x},{y}) -> scroll_at"
                    )
                else:
                    # Scroll without moving cursor
                    function_calls.append(
                        {
                            "name": "scroll_document",
                            "args": {
                                "direction": scroll_direction,
                                "magnitude": magnitude,
                            },
                        }
                    )
                    logger.debug(f"scroll({scroll_direction}, {scroll_amount}) -> scroll_document")

            elif action == "hold_key":
                text = block.input.get("text")
                duration = block.input.get("duration")

                if not text:
                    logger.warning("hold_key action missing 'text' field")
                    continue

                if duration is None or not isinstance(duration, (int, float)):
                    logger.warning(f"hold_key action invalid duration: {duration}")
                    continue

                if duration < 0 or duration > 100:
                    logger.warning(f"hold_key duration out of range (0-100): {duration}")
                    continue

                function_calls.append(
                    {"name": "hold_key", "args": {"key": text, "duration": duration}}
                )
                logger.debug(f"hold_key({text}, {duration}s) -> hold_key")

            else:
                logger.warning(f"Unknown action type: {action}")

        return function_calls

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
