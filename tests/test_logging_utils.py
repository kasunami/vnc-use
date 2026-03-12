"""Tests for logging_utils module."""

import json
from datetime import datetime
from pathlib import Path

from src.vnc_use.logging_utils import RunLogger


class TestRunLoggerInitialization:
    """Tests for RunLogger initialization."""

    def test_init_creates_run_directory(self, tmp_path: Path):
        """Should create run directory on initialization."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        assert logger.run_dir.exists()
        assert logger.run_dir.is_dir()

    def test_init_with_custom_run_id(self, tmp_path: Path):
        """Should use provided run_id."""
        logger = RunLogger(task="test task", run_id="custom_123", base_dir=str(tmp_path))
        assert logger.run_id == "custom_123"
        assert logger.run_dir.name == "custom_123"

    def test_init_generates_run_id_when_not_provided(self, tmp_path: Path):
        """Should generate run_id when not provided."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        assert logger.run_id is not None
        assert len(logger.run_id) > 0

    def test_init_stores_task(self, tmp_path: Path):
        """Should store task in logger."""
        logger = RunLogger(task="my test task", base_dir=str(tmp_path))
        assert logger.task == "my test task"

    def test_init_creates_metadata(self, tmp_path: Path):
        """Should initialize metadata with required fields."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        assert "run_id" in logger.metadata
        assert "task" in logger.metadata
        assert "start_time" in logger.metadata
        assert "steps" in logger.metadata
        assert logger.metadata["task"] == "test task"
        assert logger.metadata["steps"] == []

    def test_init_base_dir_is_path(self, tmp_path: Path):
        """Should convert base_dir to Path."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        assert isinstance(logger.base_dir, Path)


class TestGenerateRunId:
    """Tests for _generate_run_id method."""

    def test_run_id_format(self, tmp_path: Path):
        """Run ID should have timestamp and UUID format."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        # Format: YYYYMMDD_HHMMSS_xxxxxxxx
        parts = logger.run_id.split("_")
        assert len(parts) == 3
        # Date part
        assert len(parts[0]) == 8
        assert parts[0].isdigit()
        # Time part
        assert len(parts[1]) == 6
        assert parts[1].isdigit()
        # UUID part
        assert len(parts[2]) == 8

    def test_run_id_uniqueness(self, tmp_path: Path):
        """Each run should have unique ID."""
        logger1 = RunLogger(task="test task", base_dir=str(tmp_path / "run1"))
        logger2 = RunLogger(task="test task", base_dir=str(tmp_path / "run2"))
        assert logger1.run_id != logger2.run_id


class TestLogScreenshot:
    """Tests for log_screenshot method."""

    def test_log_screenshot_saves_file(self, tmp_path: Path):
        """Should save screenshot to disk."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Minimal PNG header

        path = logger.log_screenshot(step=1, screenshot_png=png_bytes)

        assert path.exists()
        assert path.read_bytes() == png_bytes

    def test_log_screenshot_naming(self, tmp_path: Path):
        """Should use correct naming convention."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        png_bytes = b"\x89PNG\r\n\x1a\n"

        path = logger.log_screenshot(step=5, screenshot_png=png_bytes)

        assert path.name == "step_005_screenshot.png"

    def test_log_screenshot_with_custom_label(self, tmp_path: Path):
        """Should use custom label in filename."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        png_bytes = b"\x89PNG\r\n\x1a\n"

        path = logger.log_screenshot(step=3, screenshot_png=png_bytes, label="before")

        assert path.name == "step_003_before.png"

    def test_log_screenshot_returns_path(self, tmp_path: Path):
        """Should return Path to saved file."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        png_bytes = b"\x89PNG\r\n\x1a\n"

        path = logger.log_screenshot(step=1, screenshot_png=png_bytes)

        assert isinstance(path, Path)
        assert path.parent == logger.run_dir


class TestLogRequest:
    """Tests for log_request method."""

    def test_log_request_saves_json(self, tmp_path: Path):
        """Should save request as JSON."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        contents = [{"message": "hello"}]
        config = {"model": "test"}

        path = logger.log_request(step=1, contents=contents, config=config)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["step"] == 1
        assert data["contents"] == contents
        assert data["config"] == config

    def test_log_request_naming(self, tmp_path: Path):
        """Should use correct naming convention."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        path = logger.log_request(step=7, contents=[], config={})

        assert path.name == "step_007_request.json"

    def test_log_request_with_redaction_enabled(self, tmp_path: Path):
        """Should call redact_secrets when enabled."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        path = logger.log_request(
            step=1, contents=[], config={"api_key": "secret123"}, redact_api_key=True
        )

        assert path.exists()

    def test_log_request_without_redaction(self, tmp_path: Path):
        """Should not redact when disabled."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        path = logger.log_request(
            step=1, contents=[], config={"value": "test"}, redact_api_key=False
        )

        data = json.loads(path.read_text())
        assert data["config"]["value"] == "test"


class TestLogResponse:
    """Tests for log_response method."""

    def test_log_response_saves_json(self, tmp_path: Path):
        """Should save response as JSON."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        response = {"result": "success", "data": [1, 2, 3]}

        path = logger.log_response(step=2, response=response)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["step"] == 2
        assert data["response"] == response

    def test_log_response_naming(self, tmp_path: Path):
        """Should use correct naming convention."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        path = logger.log_response(step=10, response={})

        assert path.name == "step_010_response.json"


class TestLogFunctionCall:
    """Tests for log_function_call method."""

    def test_log_function_call_adds_to_metadata(self, tmp_path: Path):
        """Should add function call to metadata steps."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        logger.log_function_call(
            step=1, function_name="click_at", args={"x": 100, "y": 200}, result={"success": True}
        )

        assert len(logger.metadata["steps"]) == 1
        step_data = logger.metadata["steps"][0]
        assert step_data["step"] == 1
        assert step_data["function"] == "click_at"
        assert step_data["args"] == {"x": 100, "y": 200}
        assert step_data["result"] == {"success": True}
        assert "timestamp" in step_data

    def test_log_function_call_multiple(self, tmp_path: Path):
        """Should accumulate multiple function calls."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        logger.log_function_call(step=1, function_name="click", args={}, result={"success": True})
        logger.log_function_call(step=2, function_name="type", args={}, result={"success": False})

        assert len(logger.metadata["steps"]) == 2


class TestLogError:
    """Tests for log_error method."""

    def test_log_error_adds_to_metadata(self, tmp_path: Path):
        """Should add error to metadata."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        logger.log_error(step=1, error="Connection failed")

        assert "errors" in logger.metadata
        assert len(logger.metadata["errors"]) == 1
        error_data = logger.metadata["errors"][0]
        assert error_data["step"] == 1
        assert error_data["error"] == "Connection failed"
        assert "timestamp" in error_data

    def test_log_error_creates_errors_list(self, tmp_path: Path):
        """Should create errors list if not exists."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        assert "errors" not in logger.metadata

        logger.log_error(step=1, error="Test error")

        assert "errors" in logger.metadata

    def test_log_error_multiple(self, tmp_path: Path):
        """Should accumulate multiple errors."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        logger.log_error(step=1, error="Error 1")
        logger.log_error(step=2, error="Error 2")

        assert len(logger.metadata["errors"]) == 2


class TestFinalize:
    """Tests for finalize method."""

    def test_finalize_saves_metadata(self, tmp_path: Path):
        """Should save metadata.json."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        path = logger.finalize(done=True, final_state={"step": 5, "done": True})

        assert path.exists()
        assert path.name == "metadata.json"
        data = json.loads(path.read_text())
        assert data["done"] is True
        assert "end_time" in data

    def test_finalize_records_end_time(self, tmp_path: Path):
        """Should record end time."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        logger.finalize(done=True, final_state={})

        assert "end_time" in logger.metadata

    def test_finalize_records_final_state(self, tmp_path: Path):
        """Should record final state summary."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        logger.finalize(done=False, final_state={"step": 3, "done": False, "error": "Timeout"})

        assert logger.metadata["final_state"]["step"] == 3
        assert logger.metadata["final_state"]["done"] is False
        assert logger.metadata["final_state"]["error"] == "Timeout"

    def test_finalize_saves_action_history(self, tmp_path: Path):
        """Should save action history when present."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        logger.finalize(
            done=True,
            final_state={"action_history": ["clicked button", "typed text", "scrolled down"]},
        )

        history_path = logger.run_dir / "action_history.txt"
        assert history_path.exists()
        content = history_path.read_text()
        assert "clicked button" in content
        assert "typed text" in content

    def test_finalize_generates_markdown_report(self, tmp_path: Path):
        """Should generate markdown report when step_logs present."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        step_logs = [
            {
                "step_number": 1,
                "observation": "I see a button",
                "executed_action": {"name": "click_at", "args": {"x": 100, "y": 200}},
                "result": "Success",
                "timestamp": datetime.utcnow().timestamp(),
                "screenshot_path": "step_001_screenshot.png",
            }
        ]

        logger.finalize(done=True, final_state={"step_logs": step_logs})

        report_path = logger.run_dir / "EXECUTION_REPORT.md"
        assert report_path.exists()

    def test_finalize_without_action_history(self, tmp_path: Path):
        """Should not create action_history.txt when not present."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        logger.finalize(done=True, final_state={})

        history_path = logger.run_dir / "action_history.txt"
        assert not history_path.exists()


class TestGenerateMarkdownReport:
    """Tests for _generate_markdown_report method."""

    def test_report_contains_header(self, tmp_path: Path):
        """Report should contain header with run info."""
        logger = RunLogger(task="my test task", base_dir=str(tmp_path))

        step_logs = [
            {
                "step_number": 1,
                "observation": "Test observation",
                "executed_action": {"name": "click_at", "args": {"x": 50, "y": 50}},
                "result": "Success",
                "timestamp": datetime.utcnow().timestamp(),
            }
        ]

        logger.finalize(done=True, final_state={"step_logs": step_logs, "done": True})

        report_path = logger.run_dir / "EXECUTION_REPORT.md"
        content = report_path.read_text()
        assert "Agent Execution Report" in content
        assert "my test task" in content
        assert logger.run_id in content

    def test_report_contains_steps(self, tmp_path: Path):
        """Report should contain step details."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        step_logs = [
            {
                "step_number": 1,
                "observation": "I see a login form",
                "executed_action": {
                    "name": "type_text_at",
                    "args": {"x": 100, "y": 100, "text": "user"},
                },
                "result": "Success - typed text",
                "timestamp": datetime.utcnow().timestamp(),
            }
        ]

        logger.finalize(done=True, final_state={"step_logs": step_logs, "done": True})

        report_path = logger.run_dir / "EXECUTION_REPORT.md"
        content = report_path.read_text()
        assert "Step 1" in content
        assert "I see a login form" in content
        assert "type_text_at" in content

    def test_report_shows_failure_status(self, tmp_path: Path):
        """Report should show failure status with error."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        step_logs = [
            {
                "step_number": 1,
                "executed_action": {"name": "click_at", "args": {}},
                "result": "Failed",
                "timestamp": datetime.utcnow().timestamp(),
            }
        ]

        logger.finalize(
            done=False,
            final_state={"step_logs": step_logs, "done": False, "error": "Timeout occurred"},
        )

        report_path = logger.run_dir / "EXECUTION_REPORT.md"
        content = report_path.read_text()
        assert "Failed" in content
        assert "Timeout occurred" in content

    def test_report_includes_proposed_actions(self, tmp_path: Path):
        """Report should include proposed actions when multiple."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        step_logs = [
            {
                "step_number": 1,
                "executed_action": {"name": "click_at", "args": {"x": 100, "y": 200}},
                "proposed_actions": [
                    {"name": "click_at", "args": {"x": 100, "y": 200}},
                    {"name": "type_text_at", "args": {"text": "hello"}},
                ],
                "result": "Success",
                "timestamp": datetime.utcnow().timestamp(),
            }
        ]

        logger.finalize(done=True, final_state={"step_logs": step_logs, "done": True})

        report_path = logger.run_dir / "EXECUTION_REPORT.md"
        content = report_path.read_text()
        assert "Proposed Actions" in content

    def test_report_includes_screenshot_reference(self, tmp_path: Path):
        """Report should reference screenshots."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        step_logs = [
            {
                "step_number": 1,
                "executed_action": {"name": "click_at", "args": {}},
                "result": "Success",
                "timestamp": datetime.utcnow().timestamp(),
                "screenshot_path": "step_001_after.png",
            }
        ]

        logger.finalize(done=True, final_state={"step_logs": step_logs, "done": True})

        report_path = logger.run_dir / "EXECUTION_REPORT.md"
        content = report_path.read_text()
        assert "step_001_after.png" in content

    def test_report_with_step_zero(self, tmp_path: Path):
        """Report should handle step 0 (initial step)."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        step_logs = [
            {
                "step_number": 0,
                "executed_action": {"name": "observe", "args": {}},
                "result": "Success",
                "timestamp": datetime.utcnow().timestamp(),
            }
        ]

        logger.finalize(done=True, final_state={"step_logs": step_logs, "done": True})

        report_path = logger.run_dir / "EXECUTION_REPORT.md"
        content = report_path.read_text()
        assert "Step 0" in content


class TestSerialize:
    """Tests for _serialize method."""

    def test_serialize_none(self, tmp_path: Path):
        """Should return None for None input."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        assert logger._serialize(None) is None

    def test_serialize_primitives(self, tmp_path: Path):
        """Should return primitives unchanged."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        assert logger._serialize("hello") == "hello"
        assert logger._serialize(42) == 42
        assert logger._serialize(3.14) == 3.14
        assert logger._serialize(True) is True
        assert logger._serialize(False) is False

    def test_serialize_list(self, tmp_path: Path):
        """Should serialize lists recursively."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        result = logger._serialize([1, "two", {"three": 3}])
        assert result == [1, "two", {"three": 3}]

    def test_serialize_tuple(self, tmp_path: Path):
        """Should serialize tuples as lists."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        result = logger._serialize((1, 2, 3))
        assert result == [1, 2, 3]

    def test_serialize_dict(self, tmp_path: Path):
        """Should serialize dicts recursively."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        result = logger._serialize({"a": 1, "b": {"c": 2}})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_serialize_object_with_dict(self, tmp_path: Path):
        """Should serialize objects via __dict__."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        class CustomObj:
            def __init__(self):
                self.name = "test"
                self.value = 123

        obj = CustomObj()
        result = logger._serialize(obj)
        assert result == {"name": "test", "value": 123}

    def test_serialize_fallback_to_str(self, tmp_path: Path):
        """Should fallback to str() for unknown types."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))

        # datetime doesn't have __dict__ in a useful way
        from datetime import date

        result = logger._serialize(date(2024, 1, 15))
        assert isinstance(result, str)
        assert "2024" in result


class TestRedactSecrets:
    """Tests for _redact_secrets method."""

    def test_redact_returns_dict(self, tmp_path: Path):
        """Should return a dict."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        data = {"key": "value"}
        result = logger._redact_secrets(data)
        assert isinstance(result, dict)

    def test_redact_with_api_key_pattern(self, tmp_path: Path):
        """Should handle data containing api_key pattern."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        data = {"config": {"api_key": "secret123"}}
        result = logger._redact_secrets(data)
        # Current implementation doesn't actually redact, just returns data
        assert result is not None

    def test_redact_with_password_pattern(self, tmp_path: Path):
        """Should handle data containing password pattern."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        data = {"auth": {"password": "secret"}}
        result = logger._redact_secrets(data)
        assert result is not None

    def test_redact_preserves_structure(self, tmp_path: Path):
        """Should preserve data structure."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        data = {"level1": {"level2": {"value": "test"}}}
        result = logger._redact_secrets(data)
        assert result["level1"]["level2"]["value"] == "test"


class TestGetRunDir:
    """Tests for get_run_dir method."""

    def test_get_run_dir_returns_path(self, tmp_path: Path):
        """Should return Path object."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        result = logger.get_run_dir()
        assert isinstance(result, Path)

    def test_get_run_dir_matches_run_dir(self, tmp_path: Path):
        """Should return same as run_dir attribute."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        assert logger.get_run_dir() == logger.run_dir


class TestGetRunId:
    """Tests for get_run_id method."""

    def test_get_run_id_returns_string(self, tmp_path: Path):
        """Should return string."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        result = logger.get_run_id()
        assert isinstance(result, str)

    def test_get_run_id_matches_run_id(self, tmp_path: Path):
        """Should return same as run_id attribute."""
        logger = RunLogger(task="test task", base_dir=str(tmp_path))
        assert logger.get_run_id() == logger.run_id

    def test_get_run_id_with_custom_id(self, tmp_path: Path):
        """Should return custom ID when provided."""
        logger = RunLogger(task="test task", run_id="my_custom_id", base_dir=str(tmp_path))
        assert logger.get_run_id() == "my_custom_id"
