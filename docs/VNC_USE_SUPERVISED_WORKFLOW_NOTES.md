# vnc-use Supervised Workflow Notes

## Goal

vnc-use should be usable by Mesh Computer (MC) as a bounded desktop-control tool. MC should select vnc-use for visible GUI workflows, call policy-scoped actions, and use a watcher/verifier loop to keep long workflows on track.

## Lessons From The Live Desktop Pilot

- VNC connectivity and screenshot capture worked.
- Policy profiles reduced drift by limiting allowed actions per step.
- Cropping is necessary for large or multi-monitor desktops.
- Small menu/button targets need better targeting than raw model-proposed coordinates.
- `click_text_or_button` should be preferred for visible labels, buttons, tabs, and menu items. It uses OCR when available and can fall back to explicit center coordinates.
- Step-limit and timeout errors can misclassify successful UI state changes.
- External-effect actions need a deterministic post-action verifier instead of another open-ended actor step.

## Recommended Architecture

```text
MC job
  -> selects vnc-use workflow preset
  -> calls execute_vnc_policy_task for one bounded phase
  -> stores artifacts/state
  -> watcher/verifier model reviews post-action screenshot
  -> continue, retry with tighter guidance, deterministic fallback, or abort
```

## Implementation Priorities

1. Add verifier hooks for policy profiles.
2. Add final-action mode for external effects.
3. Add `click_text_or_button` or OCR-backed target selection.
4. Add named crop presets.
5. Teach MC to prefer `execute_vnc_policy_task` and policy profiles.
6. Add a watcher/supervisor loop that can correct guidance between actions.

## Safety Rules

- Exact typed values must come from controller-provided structured inputs.
- Mutating/external-effect steps require a read-only precondition verifier.
- After an external-effect click, run a post-action verifier and stop.
- Abort on login prompts, modal warnings, wrong target, wrong content, or uncertain visual state.

## Retest Notes

- Final-action mode is required for external-effect clicks; otherwise a successful click can still be reported as failed if the follow-up model call times out.
- The actor must validate tool arguments before execution. Malformed actions such as `type_text_at` without coordinates or `scroll_at` without direction should be rejected before reaching the VNC backend.
- Small text/menu targets remain the biggest practical weakness. The next reliability improvement should be OCR/text-backed target selection rather than more prompt wording.
- OCR requires a local provider. The implementation uses the system `tesseract` binary, preprocesses/upscales screenshots, and tries multiple page-segmentation modes. Without `tesseract`, callers should provide fallback coordinates or expect an explicit failure.
