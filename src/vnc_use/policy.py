"""Policy presets and runtime guards for VNC desktop automation.

The policy layer keeps long workflows safe without forcing every caller to
spell out low-level action restrictions in each prompt. MC can choose a named
profile for each workflow step, while vnc-use enforces the resulting action
budget at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ALL_ACTIONS = {
    "click_at",
    "click_text_or_button",
    "double_click_at",
    "drag_and_drop",
    "hover_at",
    "key_combination",
    "navigate",
    "open_web_browser",
    "scroll_at",
    "scroll_document",
    "type_text",
    "type_text_at",
    "wait_5_seconds",
}

CLICK_TARGETING_GUIDANCE = (
    "Click-targeting rule: when clicking a button, menu item, icon, or text label, aim for the visual center "
    "of the target. If the target has text, click the center of the text label unless the control has a clearer "
    "center point. Do not click near edges, separators, or neighboring controls."
)


@dataclass(frozen=True)
class PolicyProfile:
    """Reusable action/risk profile for one bounded workflow step."""

    name: str
    description: str
    risk_level: str
    allowed_actions: frozenset[str]
    forbidden_actions: frozenset[str] = frozenset()
    requires_verification: bool = False
    requires_approval: bool = False
    typing_exact_text_only: bool = False
    guidance: tuple[str, ...] = field(default_factory=tuple)

    def excluded_actions(self, extra_excluded: list[str] | None = None) -> list[str]:
        """Return actions to exclude from the planner for this profile."""
        allowed = set(self.allowed_actions)
        excluded = (ALL_ACTIONS - allowed) | set(self.forbidden_actions)
        excluded |= set(extra_excluded or [])
        return sorted(excluded)


POLICY_PROFILES: dict[str, PolicyProfile] = {
    "desktop_observe": PolicyProfile(
        name="desktop_observe",
        description="Read-only visual inspection. No UI actions are allowed.",
        risk_level="read_only",
        allowed_actions=frozenset(),
        guidance=(
            "Observe only. Do not click, type, scroll, navigate, open launchers, or change the UI.",
            "Set done=true with actions=[] once the requested visual check is complete.",
        ),
    ),
    "browser_navigation": PolicyProfile(
        name="browser_navigation",
        description="Reversible browser/UI navigation without typing or submission.",
        risk_level="reversible",
        allowed_actions=frozenset({"click_at", "click_text_or_button", "scroll_at", "scroll_document", "wait_5_seconds"}),
        forbidden_actions=frozenset({"type_text_at", "open_web_browser", "navigate", "key_combination"}),
        guidance=(
            "Navigate only with visible UI clicks/scrolls.",
            "Do not type, submit forms, send messages, delete, purchase, or use launcher/hotkeys.",
        ),
    ),
    "click_only": PolicyProfile(
        name="click_only",
        description="One or more bounded clicks on visible UI elements.",
        risk_level="reversible",
        allowed_actions=frozenset({"click_at", "click_text_or_button", "wait_5_seconds"}),
        guidance=(
            "Use only visible UI clicks and optional waiting.",
            "Do not type, use hotkeys, navigate by URL, or open launchers.",
        ),
    ),
    "form_fill": PolicyProfile(
        name="form_fill",
        description="Fill fields with exact controller-provided text only.",
        risk_level="reversible",
        allowed_actions=frozenset({"click_at", "click_text_or_button", "type_text_at", "wait_5_seconds"}),
        forbidden_actions=frozenset({"open_web_browser", "navigate", "key_combination"}),
        typing_exact_text_only=True,
        requires_verification=True,
        guidance=(
            "Type only text explicitly provided by the controller in allowed_texts.",
            "Do not invent, transform, scrape, or autocomplete values.",
            "Do not submit, send, delete, purchase, or perform external-effect actions.",
        ),
    ),
    "external_effect_guarded": PolicyProfile(
        name="external_effect_guarded",
        description="Final externally visible action after a separate verification step.",
        risk_level="external_effect",
        allowed_actions=frozenset({"click_at", "click_text_or_button", "wait_5_seconds"}),
        forbidden_actions=frozenset({"type_text_at", "open_web_browser", "navigate", "key_combination"}),
        requires_verification=True,
        requires_approval=True,
        guidance=(
            "This profile is for a final externally visible click only after verification has passed.",
            "Do not type or modify content. If the expected target is not visible and unambiguous, stop.",
        ),
    ),
    "email_template_select": PolicyProfile(
        name="email_template_select",
        description="Select an email template without editing message content.",
        risk_level="reversible",
        allowed_actions=frozenset({"click_at", "click_text_or_button", "hover_at", "scroll_at", "scroll_document", "wait_5_seconds"}),
        forbidden_actions=frozenset({"type_text_at", "open_web_browser", "navigate", "key_combination"}),
        requires_verification=True,
        guidance=(
            "You may hover over icon-only controls to reveal tooltips before clicking.",
            "Select only the requested template. Do not edit recipients, subject, body, or send.",
            "If the template name is not visible or ambiguous, stop and report.",
        ),
    ),
    "browser_tab_select": PolicyProfile(
        name="browser_tab_select",
        description="Select an already-open browser tab using browser UI/search without submitting page content.",
        risk_level="reversible",
        allowed_actions=frozenset({"click_at", "click_text_or_button", "key_combination", "type_text", "wait_5_seconds"}),
        forbidden_actions=frozenset({"open_web_browser", "navigate", "type_text_at"}),
        typing_exact_text_only=True,
        requires_verification=True,
        guidance=(
            "Use browser tab search or visible tab clicks only.",
            "Type only exact controller-provided tab search text from allowed_texts.",
            "Do not type into web page forms. Do not submit page content.",
        ),
    ),
    "browser_address_inspect": PolicyProfile(
        name="browser_address_inspect",
        description="Inspect the active browser URL without changing page content.",
        risk_level="read_only",
        allowed_actions=frozenset({"key_combination", "wait_5_seconds"}),
        forbidden_actions=frozenset({"click_at", "click_text_or_button", "type_text", "type_text_at", "open_web_browser", "navigate"}),
        requires_verification=True,
        guidance=(
            "Inspect browser chrome only. Do not type, navigate, click page controls, submit forms, or alter page content.",
        ),
    ),
    "email_recipient_fill": PolicyProfile(
        name="email_recipient_fill",
        description="Fill an email recipient field with exact approved recipient text.",
        risk_level="reversible",
        allowed_actions=frozenset({"click_at", "click_text_or_button", "type_text_at", "wait_5_seconds"}),
        forbidden_actions=frozenset({"open_web_browser", "navigate", "key_combination"}),
        typing_exact_text_only=True,
        requires_verification=True,
        guidance=(
            "Type only the exact approved recipient from allowed_texts.",
            "Do not type subject/body content. Do not send.",
        ),
    ),
    "email_pre_send_verify": PolicyProfile(
        name="email_pre_send_verify",
        description="Read-only verification before sending email.",
        risk_level="read_only",
        allowed_actions=frozenset(),
        requires_verification=True,
        guidance=(
            "Verify recipient, template/body, and send readiness. Do not change anything.",
            "Return whether it is safe to send in the observation.",
        ),
    ),
    "email_send": PolicyProfile(
        name="email_send",
        description="Click Send after recipient/template verification has passed.",
        risk_level="external_effect",
        allowed_actions=frozenset({"click_at", "click_text_or_button", "wait_5_seconds"}),
        forbidden_actions=frozenset({"type_text_at", "open_web_browser", "navigate", "key_combination"}),
        requires_verification=True,
        requires_approval=True,
        guidance=(
            "Click Send only if the verified recipient and loaded template are visible and correct.",
            "If anything differs from the verified state, stop and report. Do not type.",
        ),
    ),
}


WORKFLOW_PRESETS: dict[str, list[str]] = {
    "zoho_send_template_email": [
        "browser_navigation",
        "email_template_select",
        "email_recipient_fill",
        "email_pre_send_verify",
        "email_send",
    ],
    "safe_form_submission": [
        "browser_navigation",
        "form_fill",
        "desktop_observe",
        "external_effect_guarded",
    ],
}


def get_policy_profile(name: str | None) -> PolicyProfile | None:
    """Return a named policy profile, raising for unknown names."""
    if not name:
        return None
    try:
        return POLICY_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(POLICY_PROFILES))
        raise ValueError(f"Unknown vnc-use policy profile {name!r}. Known profiles: {known}") from exc


def build_policy_task(task: str, profile: PolicyProfile | None, allowed_texts: list[str] | None = None) -> str:
    """Append concise policy guidance to the model-facing task."""
    if profile is None:
        return task
    lines = [
        task.strip(),
        "",
        f"Policy profile: {profile.name}",
        f"Risk level: {profile.risk_level}",
        "Allowed actions: " + (", ".join(sorted(profile.allowed_actions)) or "(none)"),
        "Forbidden actions: " + (", ".join(sorted(profile.forbidden_actions)) or "(profile default exclusions)"),
    ]
    if profile.requires_verification:
        lines.append("Verification is required; stop and report if the expected state is not clear.")
    if profile.requires_approval:
        lines.append("This step may create an external effect; proceed only if the task explicitly says verification passed.")
    if {"click_at", "click_text_or_button"} & set(profile.allowed_actions):
        lines.append(CLICK_TARGETING_GUIDANCE)
    if profile.typing_exact_text_only:
        values = allowed_texts or []
        lines.append("Typing policy: type only exact controller-provided text from allowed_texts.")
        lines.append("allowed_texts: " + (", ".join(repr(value) for value in values) if values else "(none provided)"))
    lines.extend(profile.guidance)
    return "\n".join(lines)


class PolicyGuard:
    """Runtime validator for model-proposed actions."""

    def __init__(self, profile: PolicyProfile | None, allowed_texts: list[str] | None = None) -> None:
        self.profile = profile
        self.allowed_texts = set(allowed_texts or [])

    def validate_action(self, action: dict[str, Any]) -> None:
        """Raise ValueError if an action violates the active policy."""
        if self.profile is None:
            return

        name = str(action.get("name") or "")
        if name not in self.profile.allowed_actions:
            raise ValueError(f"Policy {self.profile.name} blocks action {name!r}")
        if name in self.profile.forbidden_actions:
            raise ValueError(f"Policy {self.profile.name} forbids action {name!r}")

        if self.profile.typing_exact_text_only and name in {"type_text_at", "type_text"}:
            args = action.get("args") or {}
            text = str(args.get("text") or "")
            if text not in self.allowed_texts:
                raise ValueError(
                    f"Policy {self.profile.name} blocks typed text that was not provided in allowed_texts"
                )
