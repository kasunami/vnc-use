#!/usr/bin/env python3
"""Basic tests for multi-model support."""

import json
import os

import pytest

from vnc_use.agent import VncUseAgent
from vnc_use.planners import AnthropicPlanner, GeminiPlanner, OpenAICompatiblePlanner
from vnc_use.planners.openai_compatible import _first_json_object
from vnc_use.planners.base import BasePlanner


def test_gemini_planner_implements_base():
    """Test that GeminiPlanner implements BasePlanner interface."""
    assert issubclass(GeminiPlanner, BasePlanner)


def test_anthropic_planner_implements_base():
    """Test that AnthropicPlanner implements BasePlanner interface."""
    assert issubclass(AnthropicPlanner, BasePlanner)


def test_openai_compatible_planner_implements_base():
    """Test that OpenAICompatiblePlanner implements BasePlanner interface."""
    assert issubclass(OpenAICompatiblePlanner, BasePlanner)


def test_openai_compatible_json_parser_ignores_think_blocks():
    """Local thinking models may emit hidden reasoning before the action JSON."""
    parsed = _first_json_object(
        '<think>Ignore {"not": "the action plan"} while reasoning.</think>\n'
        '{"observation":"ready","done":true,"actions":[]}'
    )

    assert parsed == {"observation": "ready", "done": True, "actions": []}


def test_agent_default_model_provider():
    """Test that agent defaults to Gemini when no provider specified."""
    # Skip if no API keys available
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        pytest.skip("No Gemini API key available")

    agent = VncUseAgent(
        vnc_server="localhost::5901",
        vnc_password="test",
    )

    assert isinstance(agent.planner, GeminiPlanner)


def test_agent_gemini_provider():
    """Test that agent creates GeminiPlanner when provider='gemini'."""
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        pytest.skip("No Gemini API key available")

    agent = VncUseAgent(
        vnc_server="localhost::5901",
        vnc_password="test",
        model_provider="gemini",
    )

    assert isinstance(agent.planner, GeminiPlanner)


def test_agent_anthropic_provider():
    """Test that agent creates AnthropicPlanner when provider='anthropic'."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("No Anthropic API key available")

    agent = VncUseAgent(
        vnc_server="localhost::5901",
        vnc_password="test",
        model_provider="anthropic",
    )

    assert isinstance(agent.planner, AnthropicPlanner)


def test_agent_openai_compatible_provider(monkeypatch):
    """Test that agent creates OpenAICompatiblePlanner with local/OpenAI-compatible provider."""
    monkeypatch.setenv("OPENAI_MODEL", "test-vlm")

    agent = VncUseAgent(
        vnc_server="localhost::5901",
        vnc_password="test",
        model_provider="openai_compatible",
    )

    assert isinstance(agent.planner, OpenAICompatiblePlanner)


def test_openai_compatible_public_safe_default(monkeypatch):
    """Default OpenAI-compatible endpoint should be public-safe and local."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MESH_ROUTER_URL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "test-vlm")

    planner = OpenAICompatiblePlanner()

    assert planner.base_url == "http://localhost:4010/v1"


def test_openai_compatible_payload_tuning_defaults(monkeypatch):
    """Local VLM payload defaults should stay bounded for desktop-control latency."""
    monkeypatch.setenv("OPENAI_MODEL", "test-vlm")
    planner = OpenAICompatiblePlanner()
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return b'{"choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(req, timeout):
        captured["timeout"] = timeout
        captured["payload"] = req.data.decode("utf-8")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    planner._chat_completions([{"role": "user", "content": "test"}])

    payload = json.loads(captured["payload"])
    assert payload["max_tokens"] == 300
    assert captured["timeout"] == 60.0


def test_agent_invalid_provider():
    """Test that agent raises ValueError for invalid provider."""
    with pytest.raises(ValueError, match="Unknown model_provider"):
        VncUseAgent(
            vnc_server="localhost::5901",
            vnc_password="test",
            model_provider="invalid_provider",
        )


def test_anthropic_planner_initialization():
    """Test that AnthropicPlanner can be initialized."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("No Anthropic API key available")

    planner = AnthropicPlanner(excluded_actions=["drag_and_drop"])

    assert planner is not None
    assert hasattr(planner, "llm")
    assert hasattr(planner, "llm_with_tools")


def test_gemini_planner_initialization():
    """Test that GeminiPlanner can be initialized."""
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        pytest.skip("No Gemini API key available")

    planner = GeminiPlanner(excluded_actions=["drag_and_drop"])

    assert planner is not None
    assert hasattr(planner, "client")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
