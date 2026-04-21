"""OpenAI-compatible planner for VNC desktop control (local-first).

This planner talks to an OpenAI-compatible Chat Completions endpoint (e.g. a
local router like Mesh-Router) and asks the model to return a JSON action plan.

It does NOT require Anthropic/Gemini tokens. Network access is limited to the
configured OpenAI-compatible base URL.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .base import BasePlanner
from .gemini import compress_screenshot
from .vnc_tools import get_vnc_tools

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://10.0.1.47:4010/v1"


def _first_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a string."""
    # Prefer fenced blocks
    fenced = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # Fall back to "first { ... last }"
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


class OpenAICompatiblePlanner(BasePlanner):
    """Planner that uses an OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        excluded_actions: list[str] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.excluded_actions = excluded_actions or []

        self.base_url = (
            (base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("MESH_ROUTER_URL") or DEFAULT_BASE_URL)
            .rstrip("/")
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("MESH_ROUTER_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("LOCAL_MODEL")
        if not self.model:
            raise ValueError("Set OPENAI_MODEL (or LOCAL_MODEL) for the OpenAI-compatible planner")

        self.timeout_s = float(timeout_s or os.getenv("OPENAI_TIMEOUT_S") or "60")

        # Cache allowed tool names for validation
        self._allowed_tools = set(get_vnc_tools(self.excluded_actions).keys())
        logger.info(
            "Initialized OpenAI-compatible planner "
            f"(base_url={self.base_url!r}, model={self.model!r}, tools={sorted(self._allowed_tools)!r})"
        )

    def _chat_completions(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS") or "700"),
        }

        # Optional Mesh-Router pinning hints (non-OpenAI fields accepted by MR).
        pin_worker = os.getenv("MESH_PIN_WORKER") or os.getenv("OPENAI_PIN_WORKER")
        pin_base_url = os.getenv("MESH_PIN_BASE_URL") or os.getenv("OPENAI_PIN_BASE_URL")
        pin_lane_type = os.getenv("MESH_PIN_LANE_TYPE") or os.getenv("OPENAI_PIN_LANE_TYPE")
        pin_lane_id = os.getenv("MESH_PIN_LANE_ID") or os.getenv("OPENAI_PIN_LANE_ID")
        if pin_worker:
            payload["mesh_pin_worker"] = str(pin_worker)
        if pin_base_url:
            payload["mesh_pin_base_url"] = str(pin_base_url)
        if pin_lane_type:
            payload["mesh_pin_lane_type"] = str(pin_lane_type)
        if pin_lane_id:
            payload["mesh_pin_lane_id"] = str(pin_lane_id)

        # Some OpenAI-compatible servers support strict JSON mode.
        if os.getenv("OPENAI_RESPONSE_FORMAT", "").lower() in {"json", "json_object"}:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"OpenAI-compatible endpoint HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to reach OpenAI-compatible endpoint: {e}") from e

        return json.loads(body)

    def generate_stateless(self, task: str, action_history: list[str], screenshot_png: bytes) -> Any:
        compressed = compress_screenshot(screenshot_png, max_width=768)
        screenshot_b64 = base64.b64encode(compressed).decode("utf-8")

        allowed = ", ".join(sorted(self._allowed_tools))
        excluded = ", ".join(self.excluded_actions) if self.excluded_actions else "(none)"

        history = ""
        if action_history:
            # Keep it short to avoid ballooning context
            tail = action_history[-12:]
            history = "\n\nActions taken so far:\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(tail))

        system = (
            "You are controlling a computer via VNC. You will receive a screenshot of the desktop.\n"
            "Your job is to propose the next UI action(s) to accomplish the task.\n\n"
            f"Task: {task}\n\n"
            "Return ONLY a JSON object with this schema:\n"
            "{\n"
            '  "observation": string,\n'
            '  "done": boolean,\n'
            '  "actions": [ {"name": string, "args": object} ]\n'
            "}\n\n"
            "Your response MUST start with '{' and end with '}'.\n"
            "Do not include any other text, markdown, code fences, or explanations.\n"
            "Rules:\n"
            f"- Allowed action names: {allowed}\n"
            f"- Excluded actions: {excluded}\n"
            "- Coordinates x/y must be integers in the normalized range 0-999 (top-left is 0,0).\n"
            "- Prefer 1-2 actions per step. If the task is complete, set done=true and actions=[].\n"
            "- Do not include any keys other than observation/done/actions.\n"
            f"{history}"
        )

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Here is the current screenshot. What should I do next?",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            },
        ]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        raw = self._chat_completions(messages)
        message = (raw.get("choices", [{}])[0].get("message", {}) or {}) if isinstance(raw, dict) else {}
        content = message.get("content", "")
        # Some OpenAI-compatible llama.cpp servers emit "reasoning_content" separately (Qwen thinking-style).
        # Fall back so we can still extract JSON action plans.
        if (content is None or content == "") and isinstance(message.get("reasoning_content"), str):
            content = message.get("reasoning_content") or ""

        # Some servers return content blocks; normalize to string.
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in content
            ).strip()

        parsed: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            parsed = _first_json_object(str(content))
        except Exception as e:
            parse_error = str(e)

        return {"raw": raw, "content": str(content), "parsed": parsed, "parse_error": parse_error}

    def extract_text(self, response: Any) -> str:
        parsed = response.get("parsed") if isinstance(response, dict) else None
        if isinstance(parsed, dict):
            obs = parsed.get("observation")
            if isinstance(obs, str):
                return obs
        # Fall back to raw content (may include JSON)
        if isinstance(response, dict) and isinstance(response.get("content"), str):
            return response["content"]
        return ""

    def extract_function_calls(self, response: Any) -> list[dict[str, Any]]:
        parsed = response.get("parsed") if isinstance(response, dict) else None
        if not isinstance(parsed, dict):
            err = response.get("parse_error") if isinstance(response, dict) else "unknown"
            raise ValueError(f"Model output was not valid JSON actions: {err}")

        actions = parsed.get("actions", [])
        if not isinstance(actions, list):
            return []

        calls: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            name = action.get("name")
            args = action.get("args", {})
            if not isinstance(name, str) or name not in self._allowed_tools:
                continue
            if not isinstance(args, dict):
                continue

            calls.append({"name": name, "args": args})

        return calls

    def extract_safety_decision(self, response: Any) -> dict[str, Any] | None:
        # No provider-side safety signals in the generic OpenAI-compatible path.
        _ = response
        return None
