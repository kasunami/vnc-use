# vnc-use Policy Profiles

`vnc-use` supports named policy profiles so callers can run long workflows as a
sequence of bounded steps instead of giving the desktop agent unrestricted
control.

The recommended pattern is:

1. The controller, such as MC, owns workflow state and data.
2. `vnc-use` performs one bounded GUI step at a time.
3. Each step uses a policy profile with a small action budget.
4. External-effect actions, such as sending email, happen only after a separate
   read-only verification step.

## Profiles

- `desktop_observe`: no UI actions. Use for visual state checks.
- `browser_navigation`: visible UI clicks/scrolls only. No typing, hotkeys,
  launcher, or URL navigation.
- `click_only`: visible clicks and optional waits only.
- `form_fill`: click/type/wait, but typed text must exactly match
  `allowed_texts`.
- `external_effect_guarded`: final external-effect click after verification.
  No typing or navigation.
- `email_template_select`: choose a visible email template. No recipient/body
  editing and no send.
- `email_recipient_fill`: fill an email recipient with exact approved text only.
- `email_pre_send_verify`: read-only verification before sending.
- `email_send`: click Send only after recipient/template verification has
  passed.

## Text/Button Targeting

For visible labels, buttons, tabs, and menu items, prefer the
`click_text_or_button` action over raw `click_at`.

```json
{
  "name": "click_text_or_button",
  "args": {
    "label": "Template",
    "match_mode": "exact",
    "x": 180,
    "y": 410,
    "region": [0, 250, 350, 500]
  }
}
```

The action uses the system `tesseract` OCR binary, preprocesses/upscales the
screenshot for small UI text, and clicks the center of the matched text. If OCR
is unavailable or no match is found, it uses the optional fallback `x`/`y`
center coordinates. If neither OCR nor fallback coordinates can resolve the
target, the action fails explicitly.

Use `region` when the same label appears in multiple places. The region is
`[x, y, width, height]` in normalized coordinates within the current crop.

## CLI Examples

Observe only:

```bash
vnc-use run \
  --vnc live-desktop \
  --policy-profile desktop_observe \
  --screen-crop 4480,256,2560,1440 \
  --task "Confirm whether Zoho Mail is authenticated and the compose window is open."
```

Type an exact approved recipient:

```bash
vnc-use run \
  --vnc live-desktop \
  --policy-profile email_recipient_fill \
  --allowed-text lead@example.com \
  --screen-crop 4480,256,2560,1440 \
  --task "Click the To field and enter the approved recipient."
```

Send only after verification:

```bash
vnc-use run \
  --vnc live-desktop \
  --policy-profile email_send \
  --screen-crop 4480,256,2560,1440 \
  --task "Verification has passed. Click Send and stop."
```

## MCP Usage

MC should prefer `execute_vnc_policy_task`:

```json
{
  "hostname": "live-desktop",
  "policy_profile": "email_recipient_fill",
  "screen_crop": "4480,256,2560,1440",
  "allowed_texts": ["lead@example.com"],
  "task": "Click the To field and enter the approved recipient.",
  "step_limit": 4,
  "timeout": 120
}
```

`execute_vnc_task` still exists for compatibility and can also accept
`policy_profile`, but raw unrestricted desktop automation should be reserved for
debugging.

## Bulk Email Workflow Shape

The safe email campaign loop belongs in MC or another controller:

1. Pick the next unsent approved lead from a local source of truth.
2. Use `desktop_observe` to verify Zoho state.
3. Use `click_only` or `email_template_select` to open compose and select the
   requested template.
4. Use `email_recipient_fill` with `allowed_texts=[recipient]`.
5. Use `email_pre_send_verify` to confirm the visible recipient/template.
6. Use `email_send` only if verification passes.
7. Mark the lead sent only after the UI confirms the send/outbox state.

The VLM should never invent recipients, scrape a random visible address, type
credentials, send without verification, or continue after an auth/security
prompt.
