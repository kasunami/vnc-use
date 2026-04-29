import pytest

from vnc_use.policy import PolicyGuard, build_policy_task, get_policy_profile


def test_desktop_observe_excludes_all_actions():
    profile = get_policy_profile("desktop_observe")

    assert profile is not None
    assert profile.allowed_actions == frozenset()
    assert "click_at" in profile.excluded_actions()
    assert "type_text_at" in profile.excluded_actions()


def test_click_only_allows_click_and_wait_only():
    profile = get_policy_profile("click_only")

    assert profile is not None
    assert set(profile.allowed_actions) == {"click_at", "click_text_or_button", "wait_5_seconds"}
    assert "open_web_browser" in profile.excluded_actions()
    assert "type_text_at" in profile.excluded_actions()


def test_policy_guard_blocks_disallowed_action():
    profile = get_policy_profile("click_only")
    guard = PolicyGuard(profile)

    guard.validate_action({"name": "click_at", "args": {"x": 100, "y": 200}})
    with pytest.raises(ValueError, match="blocks action"):
        guard.validate_action({"name": "type_text_at", "args": {"text": "hello"}})


def test_form_fill_requires_exact_allowed_text():
    profile = get_policy_profile("form_fill")
    guard = PolicyGuard(profile, allowed_texts=["lead@example.com"])

    guard.validate_action(
        {"name": "type_text_at", "args": {"x": 100, "y": 200, "text": "lead@example.com"}}
    )
    with pytest.raises(ValueError, match="allowed_texts"):
        guard.validate_action(
            {"name": "type_text_at", "args": {"x": 100, "y": 200, "text": "other@example.com"}}
        )


def test_browser_tab_select_requires_exact_search_text():
    profile = get_policy_profile("browser_tab_select")
    guard = PolicyGuard(profile, allowed_texts=["Zoho Mail"])

    guard.validate_action({"name": "key_combination", "args": {"keys": "control+shift+a"}})
    guard.validate_action({"name": "type_text", "args": {"text": "Zoho Mail", "press_enter": True}})
    with pytest.raises(ValueError, match="allowed_texts"):
        guard.validate_action({"name": "type_text", "args": {"text": "wrong", "press_enter": True}})


def test_build_policy_task_includes_profile_guidance():
    profile = get_policy_profile("email_recipient_fill")
    task = build_policy_task("Fill the To field.", profile, allowed_texts=["lead@example.com"])

    assert "Policy profile: email_recipient_fill" in task
    assert "lead@example.com" in task
    assert "Do not send" in task
    assert "aim for the visual center" in task
    assert "center of the text label" in task


def test_unknown_policy_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown vnc-use policy profile"):
        get_policy_profile("missing")
