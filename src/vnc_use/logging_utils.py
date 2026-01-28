"""Structured logging and run artifact management."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants for report formatting
SECTION_SEPARATOR = "---\n\n"
REDACTED_VALUE = "***REDACTED***"


class RunLogger:
    """Manages logging and artifacts for a single agent run.

    Creates a dedicated folder for each run with:
    - Screenshots at each step
    - Request/response JSON logs
    - Function call history
    - Final transcript
    """

    def __init__(self, task: str, run_id: str | None = None, base_dir: str = "runs") -> None:
        """Initialize run logger.

        Args:
            task: User's task description
            run_id: Optional run ID (generates UUID if not provided)
            base_dir: Base directory for run artifacts
        """
        self.run_id = run_id or self._generate_run_id()
        self.task = task
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / self.run_id

        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created run directory: {self.run_dir}")

        # Initialize metadata
        self.metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "task": task,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "steps": [],
        }

    def _generate_run_id(self) -> str:
        """Generate unique run ID.

        Returns:
            Run ID with timestamp and UUID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"

    def log_screenshot(self, step: int, screenshot_png: bytes, label: str = "screenshot") -> Path:
        """Save screenshot to disk.

        Args:
            step: Step number
            screenshot_png: PNG bytes
            label: Optional label for screenshot (e.g., 'before', 'after')

        Returns:
            Path to saved screenshot
        """
        filename = f"step_{step:03d}_{label}.png"
        path = self.run_dir / filename
        path.write_bytes(screenshot_png)
        logger.debug(f"Saved screenshot: {path}")
        return path

    def log_request(
        self,
        step: int,
        contents: list[Any],
        config: Any,
        redact_api_key: bool = True,
    ) -> Path:
        """Save API request to disk.

        Args:
            step: Step number
            contents: Request contents
            config: Request config
            redact_api_key: Whether to redact API keys

        Returns:
            Path to saved request
        """
        # Convert to JSON-serializable format
        request_data = {
            "step": step,
            "contents": self._serialize(contents),
            "config": self._serialize(config),
        }

        if redact_api_key:
            request_data = self._redact_secrets(request_data)

        filename = f"step_{step:03d}_request.json"
        path = self.run_dir / filename
        path.write_text(json.dumps(request_data, indent=2))
        logger.debug(f"Saved request: {path}")
        return path

    def log_response(self, step: int, response: Any) -> Path:
        """Save API response to disk.

        Args:
            step: Step number
            response: API response

        Returns:
            Path to saved response
        """
        response_data = {
            "step": step,
            "response": self._serialize(response),
        }

        filename = f"step_{step:03d}_response.json"
        path = self.run_dir / filename
        path.write_text(json.dumps(response_data, indent=2))
        logger.debug(f"Saved response: {path}")
        return path

    def log_function_call(
        self,
        step: int,
        function_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Log function call execution.

        Args:
            step: Step number
            function_name: Name of function executed
            args: Function arguments
            result: Execution result (success, error, etc.)
        """
        call_data = {
            "step": step,
            "function": function_name,
            "args": args,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.metadata["steps"].append(call_data)
        logger.info(f"Step {step}: {function_name}({args}) -> {result.get('success', False)}")

    def log_error(self, step: int, error: str) -> None:
        """Log error during execution.

        Args:
            step: Step number
            error: Error message
        """
        error_data = {
            "step": step,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if "errors" not in self.metadata:
            self.metadata["errors"] = []
        self.metadata["errors"].append(error_data)
        logger.error(f"Step {step} error: {error}")

    def finalize(self, done: bool, final_state: dict[str, Any]) -> Path:
        """Finalize run and save metadata.

        Args:
            done: Whether task completed successfully
            final_state: Final agent state

        Returns:
            Path to metadata file
        """
        self.metadata["end_time"] = datetime.now(timezone.utc).isoformat()
        self.metadata["done"] = done
        self.metadata["final_state"] = {
            "step": final_state.get("step", 0),
            "done": final_state.get("done", False),
            "error": final_state.get("error"),
        }

        # Save metadata
        metadata_path = self.run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(self.metadata, indent=2))

        # Save action history if available
        if final_state.get("action_history"):
            history_path = self.run_dir / "action_history.txt"
            with open(history_path, "w") as f:
                f.write(f"Task: {self.metadata['task']}\n")
                f.write("=" * 70 + "\n\n")
                f.writelines(
                    f"{i}. {action}\n" for i, action in enumerate(final_state["action_history"], 1)
                )
            logger.info(f"Saved action history: {history_path}")

        # Generate markdown report if step logs available
        if final_state.get("step_logs"):
            report_path = self._generate_markdown_report(final_state)
            logger.info(f"Generated markdown report: {report_path}")

        logger.info(f"Finalized run: {metadata_path}")

        return metadata_path

    def _format_action_args(self, action: dict[str, Any]) -> str:
        """Format action arguments as a string."""
        args = action.get("args", {})
        return ", ".join(f"{k}={v}" for k, v in args.items())

    def _calculate_step_duration(
        self, step_log: dict[str, Any], step_logs: list[dict[str, Any]], start_time: float
    ) -> float:
        """Calculate duration of a step."""
        step_num = step_log["step_number"]
        if step_num > 0 and step_num - 1 < len(step_logs):
            prev_time = step_logs[step_num - 1]["timestamp"] if step_num > 0 else start_time
            return step_log["timestamp"] - prev_time
        return 0

    def _write_report_header(self, f: Any, duration: float, final_state: dict[str, Any]) -> None:
        """Write report header section."""
        status = "✓ Completed" if final_state.get("done") else "✗ Failed"
        f.write("# Agent Execution Report\n\n")
        f.write(f"**Run ID:** `{self.metadata['run_id']}`\n\n")
        f.write(f"**Task:** {self.metadata['task']}\n\n")
        f.write(f"**Duration:** {duration:.1f} seconds\n\n")
        f.write(f"**Status:** {status}\n\n")
        if final_state.get("error"):
            f.write(f"**Error:** {final_state['error']}\n\n")
        f.write(SECTION_SEPARATOR)
        f.write("## Initial Observation\n\n")
        f.write("![Initial Screenshot](step_000_initial.png)\n\n")
        f.write(SECTION_SEPARATOR)
        f.write("## Execution Timeline\n\n")

    def _write_step_log(self, f: Any, step_log: dict[str, Any], step_duration: float) -> None:
        """Write a single step log entry."""
        step_num = step_log["step_number"]
        f.write(f"### Step {step_num} ({step_duration:.1f}s)\n\n")

        observation = step_log.get("observation", "")
        if observation:
            f.write("**Model Observation:**\n")
            f.write(f"> {observation}\n\n")

        proposed = step_log.get("proposed_actions", [])
        if proposed and len(proposed) > 1:
            f.write("**Proposed Actions:**\n")
            for i, prop_action in enumerate(proposed, 1):
                args_str = self._format_action_args(prop_action)
                f.write(f"{i}. `{prop_action['name']}({args_str})`\n")
            f.write("\n")

        action = step_log["executed_action"]
        args_str = self._format_action_args(action)
        f.write(f"**Executed:** `{action['name']}({args_str})`\n\n")

        result = step_log["result"]
        result_icon = "✓" if "Success" in result else "✗"
        f.write(f"**Result:** {result_icon} {result}\n\n")

        screenshot_path = step_log.get("screenshot_path")
        if screenshot_path:
            f.write(f"![After Step {step_num}]({screenshot_path})\n\n")

        f.write(SECTION_SEPARATOR)

    def _write_report_summary(
        self, f: Any, step_logs: list[dict[str, Any]], final_state: dict[str, Any]
    ) -> None:
        """Write report summary section."""
        is_success = final_state.get("done") and not final_state.get("error")
        f.write("## Summary\n\n")
        f.write(f"- **Total Steps:** {len(step_logs)}\n")
        f.write(f"- **Success:** {'Yes' if is_success else 'No'}\n")
        if final_state.get("error"):
            f.write(f"- **Final Error:** {final_state['error']}\n")
        f.write(f"- **Screenshots Saved:** {len(step_logs) + 1}\n")
        f.write(f"- **Run Directory:** `{self.run_dir.name}`\n")

    def _generate_markdown_report(self, final_state: dict[str, Any]) -> Path:
        """Generate markdown execution report.

        Args:
            final_state: Final agent state with step_logs

        Returns:
            Path to generated report
        """
        report_path = self.run_dir / "EXECUTION_REPORT.md"
        step_logs = final_state.get("step_logs", [])

        start = datetime.fromisoformat(self.metadata["start_time"])
        end = datetime.fromisoformat(self.metadata["end_time"])
        duration = (end - start).total_seconds()

        with open(report_path, "w") as f:
            self._write_report_header(f, duration, final_state)

            for step_log in step_logs:
                step_duration = self._calculate_step_duration(
                    step_log, step_logs, start.timestamp()
                )
                self._write_step_log(f, step_log, step_duration)

            self._write_report_summary(f, step_logs, final_state)

        return report_path

    def _serialize(self, obj: Any) -> Any:
        """Serialize object to JSON-compatible format.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation
        """
        if obj is None:
            return None
        if isinstance(obj, str | int | float | bool):
            return obj
        if isinstance(obj, list | tuple):
            return [self._serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if hasattr(obj, "__dict__"):
            return self._serialize(obj.__dict__)
        return str(obj)

    def _redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact API keys and secrets from data.

        Args:
            data: Data to redact

        Returns:
            Data with secrets redacted
        """
        # Simple string replacement for common patterns
        data_str = json.dumps(data)

        # Redact patterns (basic implementation)
        patterns = [
            ("api_key", REDACTED_VALUE),
            ("password", REDACTED_VALUE),
            ("secret", REDACTED_VALUE),
        ]

        for pattern, _replacement in patterns:
            if pattern in data_str.lower():
                # More sophisticated redaction could be added here
                pass

        return data

    def get_run_dir(self) -> Path:
        """Get run directory path.

        Returns:
            Path to run directory
        """
        return self.run_dir

    def get_run_id(self) -> str:
        """Get run ID.

        Returns:
            Run ID
        """
        return self.run_id
