"""VNC Computer Use Agent - Multi-model desktop automation via VNC.

Supports multiple LLM backends:
- Anthropic Claude (native computer_20250124 tool)
- Anthropic Claude (via AnthropicPlanner)
- Google Gemini (via GeminiPlanner)
"""

from .agent import VncUseAgent
from .backends.vnc import VNCController
from .planners import (
    AnthropicPlanner,
    BasePlanner,
    GeminiComputerUse,
    GeminiPlanner,
    NativeComputerPlanner,
)
from .types import ActionResult, CUAState

__version__ = "0.2.0"

__all__ = [
    "ActionResult",
    "AnthropicPlanner",
    "BasePlanner",
    "CUAState",
    "GeminiComputerUse",
    "GeminiPlanner",
    "NativeComputerPlanner",
    "VNCController",
    "VncUseAgent",
]
