"""Tests for VNC tools module."""

from pydantic import BaseModel

from src.vnc_use.planners.vnc_tools import (
    VNC_TOOL_SCHEMAS,
    ClickAtTool,
    DoubleClickAtTool,
    DragAndDropTool,
    HoverAtTool,
    KeyCombinationTool,
    ScrollAtTool,
    ScrollDocumentTool,
    TypeTextAtTool,
    Wait5SecondsTool,
    get_vnc_tools,
)


class TestVNCToolSchemas:
    """Tests for VNC tool schema definitions."""

    def test_all_tools_are_pydantic_models(self):
        """All tool schemas should be Pydantic BaseModel subclasses."""
        for name, schema in VNC_TOOL_SCHEMAS.items():
            assert issubclass(schema, BaseModel), f"{name} is not a BaseModel"

    def test_click_at_tool(self):
        """ClickAtTool should have x, y fields."""
        tool = ClickAtTool(x=100, y=200)
        assert tool.x == 100
        assert tool.y == 200

    def test_double_click_at_tool(self):
        """DoubleClickAtTool should have x, y fields."""
        tool = DoubleClickAtTool(x=300, y=400)
        assert tool.x == 300
        assert tool.y == 400

    def test_hover_at_tool(self):
        """HoverAtTool should have x, y fields."""
        tool = HoverAtTool(x=500, y=600)
        assert tool.x == 500
        assert tool.y == 600

    def test_type_text_at_tool(self):
        """TypeTextAtTool should have x, y, text fields."""
        tool = TypeTextAtTool(x=100, y=200, text="hello")
        assert tool.x == 100
        assert tool.y == 200
        assert tool.text == "hello"
        assert tool.press_enter is False
        assert tool.clear_before_typing is False

    def test_type_text_at_tool_with_options(self):
        """TypeTextAtTool should accept press_enter and clear_before_typing."""
        tool = TypeTextAtTool(x=100, y=200, text="test", press_enter=True, clear_before_typing=True)
        assert tool.press_enter is True
        assert tool.clear_before_typing is True

    def test_key_combination_tool(self):
        """KeyCombinationTool should have keys field."""
        tool = KeyCombinationTool(keys="control+c")
        assert tool.keys == "control+c"

    def test_scroll_document_tool(self):
        """ScrollDocumentTool should have direction and magnitude fields."""
        tool = ScrollDocumentTool(direction="down")
        assert tool.direction == "down"
        assert tool.magnitude == 800  # default

    def test_scroll_document_tool_custom_magnitude(self):
        """ScrollDocumentTool should accept custom magnitude."""
        tool = ScrollDocumentTool(direction="up", magnitude=400)
        assert tool.direction == "up"
        assert tool.magnitude == 400

    def test_scroll_at_tool(self):
        """ScrollAtTool should have x, y, direction, magnitude fields."""
        tool = ScrollAtTool(x=500, y=500, direction="down")
        assert tool.x == 500
        assert tool.y == 500
        assert tool.direction == "down"
        assert tool.magnitude == 800  # default

    def test_drag_and_drop_tool(self):
        """DragAndDropTool should have start and destination coordinates."""
        tool = DragAndDropTool(x=100, y=100, destination_x=500, destination_y=500)
        assert tool.x == 100
        assert tool.y == 100
        assert tool.destination_x == 500
        assert tool.destination_y == 500

    def test_wait_5_seconds_tool(self):
        """Wait5SecondsTool should be instantiable."""
        tool = Wait5SecondsTool()
        assert tool is not None


class TestGetVncTools:
    """Tests for get_vnc_tools function."""

    def test_get_all_tools(self):
        """Should return all tools when no exclusions."""
        tools = get_vnc_tools()
        assert len(tools) == len(VNC_TOOL_SCHEMAS)
        assert "click_at" in tools
        assert "type_text_at" in tools
        assert "key_combination" in tools

    def test_get_tools_with_none_exclusions(self):
        """Should return all tools when exclusions is None."""
        tools = get_vnc_tools(None)
        assert len(tools) == len(VNC_TOOL_SCHEMAS)

    def test_get_tools_with_empty_exclusions(self):
        """Should return all tools when exclusions is empty list."""
        tools = get_vnc_tools([])
        assert len(tools) == len(VNC_TOOL_SCHEMAS)

    def test_exclude_single_tool(self):
        """Should exclude specified tool."""
        tools = get_vnc_tools(["click_at"])
        assert "click_at" not in tools
        assert len(tools) == len(VNC_TOOL_SCHEMAS) - 1

    def test_exclude_multiple_tools(self):
        """Should exclude multiple specified tools."""
        tools = get_vnc_tools(["click_at", "drag_and_drop", "scroll_at"])
        assert "click_at" not in tools
        assert "drag_and_drop" not in tools
        assert "scroll_at" not in tools
        assert len(tools) == len(VNC_TOOL_SCHEMAS) - 3

    def test_exclude_nonexistent_tool(self):
        """Should ignore nonexistent tool names in exclusions."""
        tools = get_vnc_tools(["nonexistent_tool"])
        assert len(tools) == len(VNC_TOOL_SCHEMAS)

    def test_returns_correct_types(self):
        """Returned values should be Pydantic model classes."""
        tools = get_vnc_tools()
        for name, schema in tools.items():
            assert isinstance(name, str)
            assert issubclass(schema, BaseModel)
