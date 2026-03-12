"""Tests for safety module."""

from src.vnc_use.safety import HITLGate, requires_confirmation, should_block


class TestRequiresConfirmation:
    """Tests for requires_confirmation function."""

    def test_none_safety_decision(self):
        """Should return False for None input."""
        assert requires_confirmation(None) is False

    def test_empty_dict(self):
        """Should return False for empty dict."""
        assert requires_confirmation({}) is False

    def test_confirm_action(self):
        """Should return True when action contains 'confirm'."""
        assert requires_confirmation({"action": "confirm"}) is True
        assert requires_confirmation({"action": "CONFIRM"}) is True
        assert requires_confirmation({"action": "please_confirm"}) is True
        assert requires_confirmation({"action": "confirm_action"}) is True

    def test_require_confirmation_action(self):
        """Should return True for 'require_confirmation' action."""
        assert requires_confirmation({"action": "require_confirmation"}) is True
        assert requires_confirmation({"action": "REQUIRE_CONFIRMATION"}) is True

    def test_other_actions(self):
        """Should return False for non-confirmation actions."""
        assert requires_confirmation({"action": "proceed"}) is False
        assert requires_confirmation({"action": "execute"}) is False
        assert requires_confirmation({"action": "block"}) is False
        assert requires_confirmation({"action": ""}) is False

    def test_missing_action_key(self):
        """Should return False when action key is missing."""
        assert requires_confirmation({"reason": "test"}) is False


class TestShouldBlock:
    """Tests for should_block function."""

    def test_none_safety_decision(self):
        """Should return False for None input."""
        assert should_block(None) is False

    def test_empty_dict(self):
        """Should return False for empty dict."""
        assert should_block({}) is False

    def test_block_action(self):
        """Should return True for 'block' action."""
        assert should_block({"action": "block"}) is True
        assert should_block({"action": "BLOCK"}) is True

    def test_deny_action(self):
        """Should return True for 'deny' action."""
        assert should_block({"action": "deny"}) is True
        assert should_block({"action": "DENY"}) is True

    def test_reject_action(self):
        """Should return True for 'reject' action."""
        assert should_block({"action": "reject"}) is True
        assert should_block({"action": "REJECT"}) is True

    def test_other_actions(self):
        """Should return False for non-blocking actions."""
        assert should_block({"action": "proceed"}) is False
        assert should_block({"action": "confirm"}) is False
        assert should_block({"action": "allow"}) is False
        assert should_block({"action": ""}) is False

    def test_missing_action_key(self):
        """Should return False when action key is missing."""
        assert should_block({"reason": "test"}) is False


class TestHITLGate:
    """Tests for HITLGate class."""

    def test_initialization(self):
        """Should initialize with None decision state."""
        gate = HITLGate()
        assert gate.pending_decision is None
        assert gate.pending_reason is None

    def test_set_decision_approve(self):
        """Should set approve decision correctly."""
        gate = HITLGate()
        gate.set_decision("approve", "Test reason")
        assert gate.pending_decision == "approve"
        assert gate.pending_reason == "Test reason"

    def test_set_decision_deny(self):
        """Should set deny decision correctly."""
        gate = HITLGate()
        gate.set_decision("deny", "Denied reason")
        assert gate.pending_decision == "deny"
        assert gate.pending_reason == "Denied reason"

    def test_set_decision_empty_reason(self):
        """Should accept empty reason."""
        gate = HITLGate()
        gate.set_decision("approve")
        assert gate.pending_decision == "approve"
        assert gate.pending_reason == ""

    def test_approve(self):
        """Should set approve decision via approve method."""
        gate = HITLGate()
        gate.approve("Approved by user")
        assert gate.pending_decision == "approve"
        assert gate.pending_reason == "Approved by user"

    def test_approve_default_reason(self):
        """Should use default reason for approve."""
        gate = HITLGate()
        gate.approve()
        assert gate.pending_decision == "approve"
        assert gate.pending_reason == "User approved"

    def test_deny(self):
        """Should set deny decision via deny method."""
        gate = HITLGate()
        gate.deny("Denied by user")
        assert gate.pending_decision == "deny"
        assert gate.pending_reason == "Denied by user"

    def test_deny_default_reason(self):
        """Should use default reason for deny."""
        gate = HITLGate()
        gate.deny()
        assert gate.pending_decision == "deny"
        assert gate.pending_reason == "User denied"

    def test_get_decision(self):
        """Should return current decision."""
        gate = HITLGate()
        assert gate.get_decision() is None

        gate.approve()
        assert gate.get_decision() == "approve"

        gate.deny()
        assert gate.get_decision() == "deny"

    def test_is_approved(self):
        """Should correctly identify approved state."""
        gate = HITLGate()
        assert gate.is_approved() is False

        gate.approve()
        assert gate.is_approved() is True

        gate.deny()
        assert gate.is_approved() is False

    def test_is_denied(self):
        """Should correctly identify denied state."""
        gate = HITLGate()
        assert gate.is_denied() is False

        gate.deny()
        assert gate.is_denied() is True

        gate.approve()
        assert gate.is_denied() is False

    def test_reset(self):
        """Should reset decision state."""
        gate = HITLGate()
        gate.approve("Test")
        assert gate.pending_decision == "approve"
        assert gate.pending_reason == "Test"

        gate.reset()
        assert gate.pending_decision is None
        assert gate.pending_reason is None

    def test_request_confirmation(self):
        """Should log confirmation request without error."""
        gate = HITLGate()
        safety_decision = {"action": "confirm", "reason": "Risky operation"}
        pending_calls = [{"name": "click_at", "args": {"x": 100, "y": 200}}]

        # Should not raise
        gate.request_confirmation(safety_decision, pending_calls)

    def test_request_confirmation_missing_reason(self):
        """Should handle missing reason in safety decision."""
        gate = HITLGate()
        safety_decision = {"action": "confirm"}
        pending_calls = [{"name": "type_text", "args": {"text": "test"}}]

        # Should not raise, uses default "Unknown reason"
        gate.request_confirmation(safety_decision, pending_calls)

    def test_request_confirmation_empty_pending(self):
        """Should handle empty pending calls."""
        gate = HITLGate()
        safety_decision = {"action": "confirm", "reason": "Test"}

        # Should not raise
        gate.request_confirmation(safety_decision, [])
