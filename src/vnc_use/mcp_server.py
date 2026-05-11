"""MCP server for VNC Computer Use agent.

Provides a streaming MCP tool that executes tasks on VNC desktops,
reporting progress, observations, and screenshots in real-time.

Security: VNC credentials are stored securely using a credential store
(OS keyring, .netrc file, or environment variables). Never pass passwords
as tool parameters to avoid exposing them to LLMs.
"""

import base64
import logging
import os
import time
from contextlib import contextmanager
from typing import Any

from fastmcp import Context, FastMCP

from .agent import VncUseAgent
from .backends.vnc import VNCController
from .credential_store import VNCCredentials, get_default_store
from .planners.gemini import compress_screenshot
from .policy import PolicyGuard, build_policy_task, get_policy_profile
from .types import CUAState

logger = logging.getLogger(__name__)

# Create MCP server instance
mcp = FastMCP("VNC Computer Use Agent")

# Initialize credential store
credential_store = get_default_store()


def _build_error_result(error_msg: str) -> dict[str, Any]:
    """Build a standardized error result."""
    return {
        "success": False,
        "error": error_msg,
        "run_id": None,
        "run_dir": None,
        "steps": 0,
    }


@contextmanager
def _temporary_env(name: str, value: str | None) -> Any:
    """Temporarily set an environment variable for one run."""
    old_value = os.environ.get(name)
    if value is not None:
        os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


async def _lookup_credentials(
    hostname: str, ctx: Context | None
) -> tuple[VNCCredentials | None, dict[str, Any] | None]:
    """Look up credentials from store.

    Returns:
        Tuple of (credentials, error_result) - error_result is set if lookup failed
    """
    credentials = credential_store.get(hostname)
    if credentials:
        return credentials, None

    error_msg = (
        f"No credentials found for hostname '{hostname}'. "
        f"Configure credentials using: vnc-use credentials set {hostname}"
    )
    logger.error(error_msg)
    if ctx:
        await ctx.info(f"✗ Error: {error_msg}")
    return None, _build_error_result(error_msg)


def _create_hitl_callback(ctx: Context | None) -> Any:
    """Create HITL callback for user approval via MCP elicitation."""
    if not ctx:
        return None

    async def hitl_callback(safety_decision: dict, pending_calls: list) -> bool:
        """Request user approval via MCP elicitation."""
        reason = safety_decision.get("reason", "Unknown reason") if safety_decision else "Unknown"
        actions = ", ".join(call["name"] for call in pending_calls)

        await ctx.info(f"⚠️  Safety confirmation required: {reason}")
        await ctx.info(f"📋 Proposed actions: {actions}")

        try:
            result = await ctx.elicit(
                message=f"Safety confirmation required: {reason}\n"
                f"Proposed actions: {actions}\n"
                f"Approve execution?",
                response_type=None,
            )

            if result.action == "accept":
                await ctx.info("✓ User approved action")
                return True
            if result.action == "decline":
                await ctx.info("✗ User declined action")
                return False
            await ctx.info("✗ User cancelled operation")
            return False

        except Exception as e:
            logger.error(f"Elicitation failed: {e}")
            await ctx.info(f"✗ Approval request failed: {e}")
            return False

    return hitl_callback


async def _report_completion(ctx: Context, result: dict[str, Any]) -> None:
    """Report task completion status."""
    await ctx.info("Task execution completed")
    if result.get("success"):
        steps = result.get("final_state", {}).get("step", 0)
        await ctx.info(f"✓ Task completed successfully in {steps} steps")
    else:
        error = result.get("error") or result.get("final_state", {}).get("error", "Unknown error")
        await ctx.info(f"✗ Task failed: {error}")


@mcp.tool()
async def execute_vnc_task(
    hostname: str,
    task: str,
    step_limit: int = 40,
    timeout: int = 300,
    policy_profile: str | None = None,
    allowed_texts: list[str] | None = None,
    screen_crop: str | None = None,
    stop_after_successful_action: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Execute a task on a VNC desktop with streaming progress updates.

    This tool connects to a VNC server, executes the given task using the
    Gemini 2.5 Computer Use model, and streams progress updates, observations,
    and screenshots back to the client.

    Security: Credentials are looked up from the credential store by hostname.
    Never pass passwords as tool parameters - they would be exposed to the LLM.

    Args:
        hostname: VNC server hostname (e.g., "vnc-prod-01.example.com" or "vnc-desktop")
                  Used to look up credentials from credential store.
        task: Task description to execute
        step_limit: Maximum number of steps (default: 40)
        timeout: Timeout in seconds (default: 300)
        policy_profile: Optional named action policy profile. Prefer this for MC workflows.
        allowed_texts: Exact text strings allowed by exact-text typing profiles.
        screen_crop: Optional x,y,width,height crop for large/multi-monitor desktops.
        stop_after_successful_action: Stop immediately after the first successful action.
        ctx: FastMCP context for streaming (injected automatically)

    Returns:
        Result dictionary with:
        - success: Whether task completed successfully
        - run_id: Unique run identifier
        - run_dir: Path to run artifacts directory
        - steps: Number of steps executed
        - error: Error message (if failed)

    Raises:
        ValueError: If credentials for hostname not found in credential store
    """
    if ctx:
        await ctx.info(f"Starting VNC task: {task}")
        await ctx.info(f"Looking up credentials for hostname: {hostname}")

    try:
        # Look up credentials from store
        credentials, error_result = await _lookup_credentials(hostname, ctx)
        if error_result:
            return error_result
        if credentials is None:
            return _build_error_result(f"Credentials lookup failed for hostname '{hostname}'")

        if ctx:
            await ctx.info(f"Found credentials for {hostname}")
            await ctx.info(f"Connecting to VNC server: {credentials.server}")

        # Determine model provider from environment
        model_provider = os.getenv("MODEL_PROVIDER", "gemini")
        logger.info(f"Using model provider: {model_provider}")

        profile = get_policy_profile(policy_profile or os.getenv("VNC_POLICY_PROFILE"))
        guard = None
        excluded_actions = None
        task_for_model = task
        if profile is not None:
            allowed_text_values = list(allowed_texts or [])
            guard = PolicyGuard(profile, allowed_texts=allowed_text_values)
            excluded_actions = profile.excluded_actions()
            task_for_model = build_policy_task(task, profile, allowed_texts=allowed_text_values)
            if ctx:
                await ctx.info(f"Using policy profile: {profile.name}")
                await ctx.info(f"Allowed actions: {sorted(profile.allowed_actions)}")

        # Create agent with HITL enabled and elicitation callback
        agent = VncUseAgent(
            vnc_server=credentials.server,
            vnc_password=credentials.password,
            step_limit=step_limit,
            seconds_timeout=timeout,
            hitl_mode=True,
            hitl_callback=_create_hitl_callback(ctx),
            model_provider=model_provider,
            excluded_actions=excluded_actions,
            action_guard=guard,
            stop_after_successful_action=stop_after_successful_action,
        )

        # Monkey-patch agent nodes to add streaming
        if ctx:
            agent = _wrap_agent_for_streaming(agent, ctx, step_limit)

        # Execute task
        with _temporary_env("VNC_SCREEN_CROP", screen_crop):
            result = agent.run(task_for_model)

        # Report completion
        if ctx:
            await _report_completion(ctx, result)

        return {
            "success": result.get("success", False),
            "run_id": result.get("run_id"),
            "run_dir": result.get("run_dir"),
            "steps": result.get("final_state", {}).get("step", 0),
            "error": result.get("error") or result.get("final_state", {}).get("error"),
            "observation": result.get("final_state", {}).get("last_observation"),
            "policy_profile": profile.name if profile else None,
        }

    except Exception as e:
        error_msg = f"Task execution failed: {e}"
        logger.error(error_msg, exc_info=True)
        if ctx:
            await ctx.info(f"✗ Error: {error_msg}")
        return _build_error_result(error_msg)


@mcp.tool()
async def execute_vnc_policy_task(
    hostname: str,
    task: str,
    policy_profile: str,
    step_limit: int = 20,
    timeout: int = 180,
    allowed_texts: list[str] | None = None,
    screen_crop: str | None = None,
    stop_after_successful_action: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Execute a VNC task with a named guardrail profile.

    MC should prefer this tool for durable workflows. It applies a reusable
    action budget, augments the task with policy instructions, and validates
    model-proposed actions before execution.
    """
    return await execute_vnc_task(
        hostname=hostname,
        task=task,
        step_limit=step_limit,
        timeout=timeout,
        policy_profile=policy_profile,
        allowed_texts=allowed_texts,
        screen_crop=screen_crop,
        stop_after_successful_action=stop_after_successful_action,
        ctx=ctx,
    )


@mcp.tool()
async def execute_vnc_action(
    hostname: str,
    action_name: str,
    arguments: dict[str, Any] | None = None,
    policy_profile: str | None = None,
    allowed_texts: list[str] | None = None,
    screen_crop: str | None = None,
    wait_after_s: float = 0.5,
    include_screenshot: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Execute one deterministic VNC action under an optional policy profile.

    MC campaign runners should use this for known UI controls after a verifier
    has established the current state. It avoids spending a VLM turn to rediscover
    fixed buttons while still applying the same vnc-use policy guardrails.
    """
    if ctx:
        await ctx.info(f"Executing deterministic VNC action: {action_name}")
        await ctx.info(f"Looking up credentials for hostname: {hostname}")

    try:
        credentials, error_result = await _lookup_credentials(hostname, ctx)
        if error_result:
            return error_result
        if credentials is None:
            return _build_error_result(f"Credentials lookup failed for hostname '{hostname}'")

        args = dict(arguments or {})
        profile = get_policy_profile(policy_profile or os.getenv("VNC_POLICY_PROFILE"))
        if profile is not None:
            PolicyGuard(profile, allowed_texts=list(allowed_texts or [])).validate_action(
                {"name": action_name, "args": args}
            )

        controller = VNCController(vnc_host=hostname)
        with _temporary_env("VNC_SCREEN_CROP", screen_crop):
            controller.connect(credentials.server, credentials.password)
            try:
                controller.screenshot_png()
                result = controller.execute_action(action_name, args)
                if wait_after_s > 0:
                    time.sleep(min(float(wait_after_s), 10.0))
                    result.screenshot_png = controller.screenshot_png()
            finally:
                controller.disconnect()

        payload: dict[str, Any] = {
            "success": bool(result.success),
            "error": result.error,
            "action_name": action_name,
            "policy_profile": profile.name if profile else None,
            "screen_crop": screen_crop,
        }
        if result.output:
            payload["output"] = result.output
        if include_screenshot and result.screenshot_png:
            payload["screenshot_base64"] = base64.b64encode(result.screenshot_png).decode("ascii")
        if ctx:
            await ctx.info("✓ Action completed" if result.success else f"✗ Action failed: {result.error}")
        return payload

    except Exception as e:
        error_msg = f"Action execution failed: {e}"
        logger.error(error_msg, exc_info=True)
        if ctx:
            await ctx.info(f"✗ Error: {error_msg}")
        return _build_error_result(error_msg)


def _safe_async_run(coro: Any, error_context: str) -> None:
    """Safely run an async coroutine with error handling."""
    import asyncio

    try:
        asyncio.run(coro)
    except Exception as e:
        logger.warning(f"{error_context}: {e}")


def _truncate_text(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis if too long."""
    return text[:max_len] + "..." if len(text) > max_len else text


def _format_action_summary(proposed: list[dict[str, Any]]) -> str:
    """Format a summary of proposed actions."""
    action_names = [a["name"] for a in proposed[:3]]
    summary = ", ".join(action_names)
    if len(proposed) > 3:
        summary += f" (+{len(proposed) - 3} more)"
    return summary


def _format_executed_action(action: dict[str, Any], result_text: str) -> str:
    """Format an executed action for display."""
    action_name = action.get("name", "unknown")
    args = action.get("args", {})
    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
    status = "✓" if "Success" in result_text else "✗"
    return f"{status} Executed: {action_name}({args_str})"


def _wrap_agent_for_streaming(
    agent: VncUseAgent,
    ctx: Context,
    step_limit: int,
) -> VncUseAgent:
    """Wrap agent nodes to add streaming capabilities.

    Args:
        agent: Agent instance to wrap
        ctx: FastMCP context for streaming
        step_limit: Maximum steps for progress reporting

    Returns:
        Modified agent with streaming support
    """
    original_propose = agent._propose_node
    original_act = agent._act_node

    async def _report(message: str) -> None:
        await ctx.info(message)

    async def _progress(step: int, total: int, message: str) -> None:
        await ctx.report_progress(progress=step, total=total, message=message)

    async def _screenshot(screenshot_png: bytes, step: int) -> None:
        compressed = compress_screenshot(screenshot_png, max_width=256)
        encoded = base64.b64encode(compressed).decode("utf-8")
        await ctx.info(
            f"[Screenshot Step {step}] data:image/png;base64,{encoded[:100]}... ({len(compressed)} bytes)"
        )

    def streaming_propose_node(state: CUAState) -> dict[str, Any]:
        """Wrapped propose node with streaming."""
        step = state["step"]

        _safe_async_run(
            _progress(step, step_limit, f"Step {step}: Analyzing screenshot..."),
            "Progress reporting failed",
        )

        result = original_propose(state)

        observation = result.get("observation", "")
        if observation:
            obs_preview = _truncate_text(observation)
            _safe_async_run(
                _report(f"[Step {step}] Model observes: {obs_preview}"),
                "Observation streaming failed",
            )

        proposed = result.get("proposed_actions", [])
        if proposed:
            summary = _format_action_summary(proposed)
            _safe_async_run(
                _report(f"[Step {step}] Proposed: {summary}"),
                "Action streaming failed",
            )

        return result

    def streaming_act_node(state: CUAState) -> dict[str, Any]:
        """Wrapped act node with streaming."""
        step = state["step"]
        result = original_act(state)

        screenshot_png = result.get("last_screenshot_png")
        if screenshot_png:
            _safe_async_run(
                _screenshot(screenshot_png, step),
                "Screenshot streaming failed",
            )

        step_logs = result.get("step_logs", [])
        if step_logs:
            last_log = step_logs[-1]
            action = last_log.get("executed_action", {})
            result_text = last_log.get("result", "")
            formatted = _format_executed_action(action, result_text)
            _safe_async_run(
                _report(f"[Step {step}] {formatted}"),
                "Result streaming failed",
            )

        return result

    object.__setattr__(agent, "_propose_node", streaming_propose_node)
    object.__setattr__(agent, "_act_node", streaming_act_node)

    return agent
