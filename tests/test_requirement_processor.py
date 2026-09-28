"""
AI-Based Test Case Generator
Phase 2: Unit Tests for requirement_processor.py

All tests mock the Gemini API — no real network calls are made.

Test classes:
  TestProcessedRequirementModel   - Pydantic model validation
  TestRequirementProcessorInit    - __init__ / env loading
  TestValidateRequirement         - _validate_requirement() edge cases
  TestParseResponse               - _parse_response() JSON & Pydantic handling
  TestProcess                     - end-to-end process() with mocked Gemini
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure a dummy key is always present so the module can be imported
os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-key")

from ai.requirement_processor import ProcessedRequirement, RequirementProcessor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_valid_payload() -> dict:
    """Return a dict that satisfies all ProcessedRequirement fields."""
    return {
        "actor": "registered user",
        "action": "submit the login form",
        "conditions": ["user has a valid account", "user is not already logged in"],
        "inputs": ["email address", "password"],
        "expected_outcome": "user is authenticated and redirected to the dashboard",
    }


def _make_mock_gemini_response(payload: dict) -> MagicMock:
    """
    Build a MagicMock that mimics a Gemini GenerateContentResponse.
    The .text attribute returns a JSON string of *payload*.
    """
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(payload)
    return mock_resp


# ---------------------------------------------------------------------------
# ProcessedRequirement model tests
# ---------------------------------------------------------------------------

class TestProcessedRequirementModel:
    def test_valid_model(self):
        pr = ProcessedRequirement(**_make_valid_payload())
        assert pr.actor == "registered user"
        assert pr.action == "submit the login form"
        assert len(pr.conditions) == 2
        assert len(pr.inputs) == 2
        assert "dashboard" in pr.expected_outcome

    def test_empty_conditions_and_inputs_allowed(self):
        payload = _make_valid_payload()
        payload["conditions"] = []
        payload["inputs"] = []
        pr = ProcessedRequirement(**payload)
        assert pr.conditions == []
        assert pr.inputs == []

    def test_missing_actor_raises(self):
        payload = _make_valid_payload()
        del payload["actor"]
        with pytest.raises(Exception):
            ProcessedRequirement(**payload)

    def test_missing_expected_outcome_raises(self):
        payload = _make_valid_payload()
        del payload["expected_outcome"]
        with pytest.raises(Exception):
            ProcessedRequirement(**payload)

    def test_missing_action_raises(self):
        payload = _make_valid_payload()
        del payload["action"]
        with pytest.raises(Exception):
            ProcessedRequirement(**payload)


# ---------------------------------------------------------------------------
# RequirementProcessor __init__ tests
# ---------------------------------------------------------------------------

class TestRequirementProcessorInit:
    def test_init_with_valid_api_key(self):
        with patch("ai.requirement_processor.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                proc = RequirementProcessor()
                assert proc._api_key == "fake-key"

    def test_init_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(EnvironmentError, match="GEMINI_API_KEY is not set"):
                RequirementProcessor()

    def test_repr(self):
        with patch("ai.requirement_processor.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                proc = RequirementProcessor()
                assert "gemini-flash-latest" in repr(proc)


# ---------------------------------------------------------------------------
# _validate_requirement() tests
# ---------------------------------------------------------------------------

class TestValidateRequirement:
    def setup_method(self):
        with patch("ai.requirement_processor.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                self.proc = RequirementProcessor()

    def test_valid_requirement_passes(self):
        # Should not raise
        self.proc._validate_requirement("A user should be able to log in.")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self.proc._validate_requirement("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self.proc._validate_requirement("     ")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self.proc._validate_requirement(None)  # type: ignore

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self.proc._validate_requirement(42)  # type: ignore


# ---------------------------------------------------------------------------
# _parse_response() tests
# ---------------------------------------------------------------------------

class TestParseResponse:
    def setup_method(self):
        with patch("ai.requirement_processor.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                self.proc = RequirementProcessor()

    def test_valid_json_parses_correctly(self):
        raw = json.dumps(_make_valid_payload())
        result = self.proc._parse_response(raw)
        assert isinstance(result, ProcessedRequirement)
        assert result.actor == "registered user"

    def test_markdown_fenced_json_is_handled(self):
        """Gemini sometimes wraps JSON in ```json ... ``` fences."""
        raw = "```json\n" + json.dumps(_make_valid_payload()) + "\n```"
        result = self.proc._parse_response(raw)
        assert isinstance(result, ProcessedRequirement)

    def test_plain_fence_without_language_tag(self):
        raw = "```\n" + json.dumps(_make_valid_payload()) + "\n```"
        result = self.proc._parse_response(raw)
        assert isinstance(result, ProcessedRequirement)

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            self.proc._parse_response("this is not json at all")

    def test_missing_required_field_raises_value_error(self):
        payload = _make_valid_payload()
        del payload["actor"]
        with pytest.raises(ValueError, match="missing required fields"):
            self.proc._parse_response(json.dumps(payload))

    def test_extra_fields_are_ignored(self):
        """Pydantic v2 ignores extra fields by default."""
        payload = _make_valid_payload()
        payload["extra_unknown_field"] = "should be ignored"
        result = self.proc._parse_response(json.dumps(payload))
        assert isinstance(result, ProcessedRequirement)

    def test_empty_json_object_raises(self):
        with pytest.raises(ValueError, match="missing required fields"):
            self.proc._parse_response("{}")


# ---------------------------------------------------------------------------
# process() end-to-end tests (Gemini fully mocked)
# ---------------------------------------------------------------------------

class TestProcess:
    def _make_processor(self):
        """Helper: create a RequirementProcessor with mocked genai."""
        with patch("ai.requirement_processor.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                return RequirementProcessor()

    def _mock_generate(self, proc: RequirementProcessor, payload: dict) -> MagicMock:
        """Attach a mock to proc._client.models.generate_content returning *payload*."""
        mock_resp = _make_mock_gemini_response(payload)
        proc._client.models.generate_content = MagicMock(return_value=mock_resp)
        return proc._client.models.generate_content

    def test_process_valid_requirement(self):
        proc = self._make_processor()
        self._mock_generate(proc, _make_valid_payload())

        result = proc.process(
            "A registered user should be able to log in using their email and password."
        )
        assert isinstance(result, ProcessedRequirement)
        assert result.actor == "registered user"
        assert result.action == "submit the login form"
        assert "dashboard" in result.expected_outcome

    def test_process_empty_requirement_raises(self):
        proc = self._make_processor()
        with pytest.raises(ValueError, match="non-empty string"):
            proc.process("")

    def test_process_whitespace_requirement_raises(self):
        proc = self._make_processor()
        with pytest.raises(ValueError, match="non-empty string"):
            proc.process("   ")

    def test_process_gemini_returns_empty_raises(self):
        proc = self._make_processor()
        mock_resp = MagicMock()
        mock_resp.text = ""
        proc._client.models.generate_content = MagicMock(return_value=mock_resp)

        with pytest.raises(ValueError, match="empty response"):
            proc.process("User should be able to reset their password via email.")

    def test_process_gemini_returns_malformed_json_raises(self):
        proc = self._make_processor()
        mock_resp = MagicMock()
        mock_resp.text = "Sorry, I cannot process this request."
        proc._client.models.generate_content = MagicMock(return_value=mock_resp)

        with pytest.raises(ValueError, match="not valid JSON"):
            proc.process("User should be able to reset their password.")

    def test_process_gemini_returns_incomplete_json_raises(self):
        proc = self._make_processor()
        # Valid JSON but missing 'actor' and 'inputs'
        partial = {"action": "log in", "conditions": [], "expected_outcome": "access granted"}
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(partial)
        proc._client.models.generate_content = MagicMock(return_value=mock_resp)

        with pytest.raises(ValueError, match="missing required fields"):
            proc.process("User logs in to the system.")

    def test_process_gemini_is_called_once(self):
        """Verify we send exactly one request to Gemini per process() call."""
        proc = self._make_processor()
        mock_fn = self._mock_generate(proc, _make_valid_payload())

        proc.process("A user should be able to register with their email address.")
        mock_fn.assert_called_once()

    def test_process_requirement_included_in_prompt(self):
        """The requirement text must appear verbatim inside the prompt sent to Gemini."""
        proc = self._make_processor()
        mock_fn = self._mock_generate(proc, _make_valid_payload())

        requirement = "Admin can delete any user account from the dashboard."
        proc.process(requirement)

        call_args = mock_fn.call_args
        # New SDK uses keyword args: model=..., contents=...
        prompt_sent = call_args.kwargs.get("contents", "")
        assert requirement in prompt_sent

