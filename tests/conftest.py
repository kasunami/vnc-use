"""Pytest configuration and fixtures for vnc-use tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "external: marks tests requiring network access")
    config.addinivalue_line("markers", "slow: marks tests that may take several seconds")


@pytest.fixture
def mock_vnc_controller():
    """Mock VNC controller for unit tests.

    Returns a mock object that simulates VNC controller behavior
    without requiring an actual VNC connection.
    """
    from unittest.mock import AsyncMock, MagicMock

    controller = MagicMock()
    controller.connect = AsyncMock(return_value=True)
    controller.disconnect = AsyncMock(return_value=True)
    controller.capture_screen = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n")
    controller.click = AsyncMock(return_value=True)
    controller.type_text = AsyncMock(return_value=True)
    controller.key_press = AsyncMock(return_value=True)
    return controller


@pytest.fixture
def sample_screenshot():
    """Minimal valid PNG bytes for testing screenshot handling.

    This is a 1x1 pixel transparent PNG.
    """
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture
def sample_state():
    """Sample agent state for testing."""
    return {
        "messages": [],
        "screenshot": None,
        "current_action": None,
        "action_history": [],
        "error": None,
    }
