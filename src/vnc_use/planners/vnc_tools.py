"""VNC action tool definitions for LangChain integration.

These tools describe the available VNC actions that LLMs can propose.
They do NOT execute actions - they return structured descriptions
that the agent will execute via the VNC backend.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Constants for field descriptions
COORD_RANGE = "(0-999 normalized)"


class ClickAtTool(BaseModel):
    """Click at specified coordinates on the screen."""

    x: int = Field(description=f"X coordinate {COORD_RANGE}")
    y: int = Field(description=f"Y coordinate {COORD_RANGE}")


class DoubleClickAtTool(BaseModel):
    """Double-click at specified coordinates on the screen."""

    x: int = Field(description=f"X coordinate {COORD_RANGE}")
    y: int = Field(description=f"Y coordinate {COORD_RANGE}")


class HoverAtTool(BaseModel):
    """Move mouse cursor to hover at specified coordinates."""

    x: int = Field(description=f"X coordinate {COORD_RANGE}")
    y: int = Field(description=f"Y coordinate {COORD_RANGE}")


class TypeTextAtTool(BaseModel):
    """Type text at specified coordinates.

    Clicks at the coordinates first, then types the text.
    """

    x: int = Field(description=f"X coordinate to click before typing {COORD_RANGE}")
    y: int = Field(description=f"Y coordinate to click before typing {COORD_RANGE}")
    text: str = Field(description="Text to type")
    press_enter: bool = Field(default=False, description="Whether to press Enter after typing")
    clear_before_typing: bool = Field(
        default=False,
        description="Whether to clear existing text (Ctrl+A then Delete) before typing",
    )


class KeyCombinationTool(BaseModel):
    """Press a keyboard shortcut or combination.

    Examples: 'control+c', 'control+v', 'alt+tab', 'control+shift+t'
    """

    keys: str | list[str] = Field(
        description="Key combination string (e.g., 'control+c', 'alt+f4') or list of key names (e.g., ['super','f'])"
    )


class ScrollDocumentTool(BaseModel):
    """Scroll the document in a direction.

    Uses Page Up/Down keys or arrow keys for scrolling.
    """

    direction: Literal["up", "down", "left", "right"] = Field(description="Direction to scroll")
    magnitude: int = Field(
        default=800,
        description="Scroll magnitude in pixels (approximate, implemented via key repeats)",
    )


class ScrollAtTool(BaseModel):
    """Scroll at specific coordinates on the screen.

    Useful for scrolling within a specific window or panel.
    """

    x: int = Field(description=f"X coordinate to scroll at {COORD_RANGE}")
    y: int = Field(description=f"Y coordinate to scroll at {COORD_RANGE}")
    direction: Literal["up", "down", "left", "right"] = Field(description="Direction to scroll")
    magnitude: int = Field(
        default=800,
        description="Scroll magnitude in pixels (approximate, implemented via key repeats)",
    )


class DragAndDropTool(BaseModel):
    """Drag from one location and drop at another.

    Performs a mouse drag operation from start to end coordinates.
    """

    x: int = Field(description=f"Starting X coordinate {COORD_RANGE}")
    y: int = Field(description=f"Starting Y coordinate {COORD_RANGE}")
    destination_x: int = Field(description=f"Ending X coordinate {COORD_RANGE}")
    destination_y: int = Field(description=f"Ending Y coordinate {COORD_RANGE}")


class Wait5SecondsTool(BaseModel):
    """Wait for 5 seconds.

    Essential for waiting on page loads, application launches, or animations to complete.
    """


# Tool name to Pydantic model mapping
VNC_TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "click_at": ClickAtTool,
    "double_click_at": DoubleClickAtTool,
    "hover_at": HoverAtTool,
    "type_text_at": TypeTextAtTool,
    "key_combination": KeyCombinationTool,
    "scroll_document": ScrollDocumentTool,
    "scroll_at": ScrollAtTool,
    "drag_and_drop": DragAndDropTool,
    "wait_5_seconds": Wait5SecondsTool,
}


def get_vnc_tools(excluded_actions: list[str] | None = None) -> dict[str, type[BaseModel]]:
    """Get VNC tool schemas excluding specified actions.

    Args:
        excluded_actions: List of action names to exclude

    Returns:
        Dictionary mapping tool names to Pydantic schemas
    """
    excluded = set(excluded_actions or [])
    return {name: schema for name, schema in VNC_TOOL_SCHEMAS.items() if name not in excluded}
