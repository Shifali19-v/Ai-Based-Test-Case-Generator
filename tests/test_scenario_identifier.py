"""
AI-Based Test Case Generator
Phase 3: Unit Tests for scenario_identifier.py

All tests mock the Gemini API — no real network calls are made.

Test classes:
  TestScenarioSetModel          - Pydantic model validation
  TestScenarioIdentifierInit    - __init__ / env loading
  TestValidateInput             - _validate_input() type checks
  TestParseResponse             - _parse_response() JSON & Pydantic handling
  TestIdentify                  - end-to-end identify() with mocked Gemini
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure a dummy key is always present so the module can be imported
os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-key")

from ai.requirement_processor import ProcessedRequirement
from ai.scenario_identifier import ScenarioSet, ScenarioIdentifier


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_processed_req(**overrides) -> ProcessedRequirement:
    """Return a minimal valid ProcessedRequirement."""
    base = {
        "actor": "registered user",
        "action": "submit the login form",
        "conditions": ["user has a valid account"],
        "inputs": ["email address", "password"],
        "expected_outcome": "user is authenticated and redirected to the dashboard",
    }
    base.update(overrides)
    return ProcessedRequirement(**base)


def _make_valid_scenario_payload() -> dict:
    """Return a dict satisfying all ScenarioSet fields."""
    return {
        "positive_scenarios": [
            "User logs in successfully with valid email and password.",
            "User is redirected to the dashboard after successful login.",
        ],
        "negative_scenarios": [
            "User submits the form with an incorrect password.",
            "User submits the form with an unregistered email address.",
            "User submits the form with both fields empty.",
        ],
        "boundary_scenarios": [],  # no numeric limits stated in the requirement
        "edge_cases": [
            "User submits the form with leading/trailing whitespace in the email.",
            "User attempts login while the session is already active.",
        ],
    }


def _make_mock_response(payload: dict) -> MagicMock:
    """Build a MagicMock mimicking a Gemini GenerateContentResponse."""
    mock = MagicMock()
    mock.text = json.dumps(payload)
    return mock


def _make_identifier() -> ScenarioIdentifier:
    """Create a ScenarioIdentifier with genai.Client mocked."""
    with patch("ai.scenario_identifier.genai.Client"):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            return ScenarioIdentifier()


# ---------------------------------------------------------------------------
# ScenarioSet model tests
# ---------------------------------------------------------------------------

class TestScenarioSetModel:
    def test_valid_scenario_set(self):
        ss = ScenarioSet(**_make_valid_scenario_payload())
        assert len(ss.positive_scenarios) == 2
        assert len(ss.negative_scenarios) == 3
        assert ss.boundary_scenarios == []
        assert len(ss.edge_cases) == 2

    def test_all_empty_lists_allowed(self):
        ss = ScenarioSet(
            positive_scenarios=[],
            negative_scenarios=[],
            boundary_scenarios=[],
            edge_cases=[],
        )
        assert ss.positive_scenarios == []
        assert ss.boundary_scenarios == []

    def test_missing_positive_scenarios_raises(self):
        payload = _make_valid_scenario_payload()
        del payload["positive_scenarios"]
        with pytest.raises(Exception):
            ScenarioSet(**payload)

    def test_missing_negative_scenarios_raises(self):
        payload = _make_valid_scenario_payload()
        del payload["negative_scenarios"]
        with pytest.raises(Exception):
            ScenarioSet(**payload)

    def test_missing_boundary_scenarios_raises(self):
        payload = _make_valid_scenario_payload()
        del payload["boundary_scenarios"]
        with pytest.raises(Exception):
            ScenarioSet(**payload)

    def test_missing_edge_cases_raises(self):
        payload = _make_valid_scenario_payload()
        del payload["edge_cases"]
        with pytest.raises(Exception):
            ScenarioSet(**payload)

    def test_boundary_scenarios_can_be_empty(self):
        """An empty boundary list is valid when no limits are stated."""
        ss = ScenarioSet(**_make_valid_scenario_payload())
        assert isinstance(ss.boundary_scenarios, list)
        assert ss.boundary_scenarios == []

    def test_extra_fields_are_ignored(self):
        payload = _make_valid_scenario_payload()
        payload["unexpected_field"] = "should be ignored"
        ss = ScenarioSet(**payload)
        assert isinstance(ss, ScenarioSet)


# ---------------------------------------------------------------------------
# ScenarioIdentifier __init__ tests
# ---------------------------------------------------------------------------

class TestScenarioIdentifierInit:
    def test_init_with_valid_api_key(self):
        with patch("ai.scenario_identifier.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                si = ScenarioIdentifier()
                assert si._api_key == "fake-key"

    def test_init_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(EnvironmentError, match="GEMINI_API_KEY is not set"):
                ScenarioIdentifier()

    def test_repr(self):
        with patch("ai.scenario_identifier.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                si = ScenarioIdentifier()
                assert "gemini-flash-latest" in repr(si)
                assert "google-genai" in repr(si)


# ---------------------------------------------------------------------------
# _validate_input() tests
# ---------------------------------------------------------------------------

class TestValidateInput:
    def setup_method(self):
        self.si = _make_identifier()

    def test_valid_processed_requirement_passes(self):
        # Should not raise
        self.si._validate_input(_make_processed_req())

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError, match="ProcessedRequirement"):
            self.si._validate_input("just a string")  # type: ignore

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="ProcessedRequirement"):
            self.si._validate_input(None)  # type: ignore

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="ProcessedRequirement"):
            self.si._validate_input({"actor": "user"})  # type: ignore

    def test_integer_raises_type_error(self):
        with pytest.raises(TypeError, match="ProcessedRequirement"):
            self.si._validate_input(42)  # type: ignore


# ---------------------------------------------------------------------------
# _parse_response() tests
# ---------------------------------------------------------------------------

class TestParseResponse:
    def setup_method(self):
        self.si = _make_identifier()

    def test_valid_json_parses_correctly(self):
        raw = json.dumps(_make_valid_scenario_payload())
        result = self.si._parse_response(raw)
        assert isinstance(result, ScenarioSet)
        assert len(result.positive_scenarios) == 2

    def test_markdown_fenced_json_is_handled(self):
        raw = "```json\n" + json.dumps(_make_valid_scenario_payload()) + "\n```"
        result = self.si._parse_response(raw)
        assert isinstance(result, ScenarioSet)

    def test_plain_fence_without_language_tag(self):
        raw = "```\n" + json.dumps(_make_valid_scenario_payload()) + "\n```"
        result = self.si._parse_response(raw)
        assert isinstance(result, ScenarioSet)

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            self.si._parse_response("This is not JSON.")

    def test_missing_field_raises_value_error(self):
        payload = _make_valid_scenario_payload()
        del payload["edge_cases"]
        with pytest.raises(ValueError, match="missing required fields"):
            self.si._parse_response(json.dumps(payload))

    def test_empty_json_object_raises_value_error(self):
        with pytest.raises(ValueError, match="missing required fields"):
            self.si._parse_response("{}")

    def test_extra_fields_are_ignored(self):
        payload = _make_valid_scenario_payload()
        payload["extra"] = "ignored"
        result = self.si._parse_response(json.dumps(payload))
        assert isinstance(result, ScenarioSet)

    def test_empty_boundary_list_is_valid(self):
        """Boundary scenarios may legitimately be empty."""
        payload = _make_valid_scenario_payload()
        payload["boundary_scenarios"] = []
        result = self.si._parse_response(json.dumps(payload))
        assert result.boundary_scenarios == []


# ---------------------------------------------------------------------------
# identify() end-to-end tests (Gemini fully mocked)
# ---------------------------------------------------------------------------

class TestIdentify:
    def _mock_generate(self, si: ScenarioIdentifier, payload: dict) -> MagicMock:
        """Attach a mock to si._client.models.generate_content."""
        mock_resp = _make_mock_response(payload)
        si._client.models.generate_content = MagicMock(return_value=mock_resp)
        return si._client.models.generate_content

    # --- Happy path ---

    def test_identify_returns_scenario_set(self):
        si = _make_identifier()
        self._mock_generate(si, _make_valid_scenario_payload())
        result = si.identify(_make_processed_req())
        assert isinstance(result, ScenarioSet)

    def test_positive_scenarios_populated(self):
        si = _make_identifier()
        self._mock_generate(si, _make_valid_scenario_payload())
        result = si.identify(_make_processed_req())
        assert len(result.positive_scenarios) > 0
        assert all(isinstance(s, str) for s in result.positive_scenarios)

    def test_negative_scenarios_populated(self):
        si = _make_identifier()
        self._mock_generate(si, _make_valid_scenario_payload())
        result = si.identify(_make_processed_req())
        assert len(result.negative_scenarios) > 0
        assert all(isinstance(s, str) for s in result.negative_scenarios)

    def test_boundary_scenarios_empty_when_none_stated(self):
        """When the requirement states no numeric limits, boundary list is empty."""
        si = _make_identifier()
        payload = _make_valid_scenario_payload()
        payload["boundary_scenarios"] = []
        self._mock_generate(si, payload)
        result = si.identify(_make_processed_req())
        assert result.boundary_scenarios == []

    def test_boundary_scenarios_populated_when_stated(self):
        """When the requirement explicitly states a boundary, it is reflected."""
        si = _make_identifier()
        payload = _make_valid_scenario_payload()
        payload["boundary_scenarios"] = [
            "User submits the form with a password of exactly the minimum allowed length."
        ]
        self._mock_generate(si, payload)
        req = _make_processed_req(
            conditions=["password must be at least 8 characters"]
        )
        result = si.identify(req)
        assert len(result.boundary_scenarios) == 1

    def test_edge_cases_populated(self):
        si = _make_identifier()
        self._mock_generate(si, _make_valid_scenario_payload())
        result = si.identify(_make_processed_req())
        assert len(result.edge_cases) > 0
        assert all(isinstance(s, str) for s in result.edge_cases)

    # --- Error paths ---

    def test_non_processed_req_raises_type_error(self):
        si = _make_identifier()
        with pytest.raises(TypeError, match="ProcessedRequirement"):
            si.identify("not a processed requirement")  # type: ignore

    def test_none_input_raises_type_error(self):
        si = _make_identifier()
        with pytest.raises(TypeError, match="ProcessedRequirement"):
            si.identify(None)  # type: ignore

    def test_gemini_returns_empty_raises_value_error(self):
        si = _make_identifier()
        mock_resp = MagicMock()
        mock_resp.text = ""
        si._client.models.generate_content = MagicMock(return_value=mock_resp)
        with pytest.raises(ValueError, match="empty response"):
            si.identify(_make_processed_req())

    def test_gemini_returns_malformed_json_raises_value_error(self):
        si = _make_identifier()
        mock_resp = MagicMock()
        mock_resp.text = "Sorry, I cannot help with that."
        si._client.models.generate_content = MagicMock(return_value=mock_resp)
        with pytest.raises(ValueError, match="not valid JSON"):
            si.identify(_make_processed_req())

    def test_gemini_returns_incomplete_json_raises_value_error(self):
        si = _make_identifier()
        partial = {"positive_scenarios": ["User logs in."]}  # missing 3 fields
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(partial)
        si._client.models.generate_content = MagicMock(return_value=mock_resp)
        with pytest.raises(ValueError, match="missing required fields"):
            si.identify(_make_processed_req())

    # --- Call verification ---

    def test_gemini_called_exactly_once(self):
        si = _make_identifier()
        mock_fn = self._mock_generate(si, _make_valid_scenario_payload())
        si.identify(_make_processed_req())
        mock_fn.assert_called_once()

    def test_prompt_contains_actor_and_action(self):
        """Actor and action from the requirement must appear in the prompt."""
        si = _make_identifier()
        mock_fn = self._mock_generate(si, _make_valid_scenario_payload())

        req = _make_processed_req(
            actor="system administrator",
            action="delete a user account",
        )
        si.identify(req)

        call_args = mock_fn.call_args
        prompt_sent = call_args.kwargs.get("contents", "")
        assert "system administrator" in prompt_sent
        assert "delete a user account" in prompt_sent

    def test_prompt_contains_expected_outcome(self):
        """Expected outcome must appear verbatim in the prompt."""
        si = _make_identifier()
        mock_fn = self._mock_generate(si, _make_valid_scenario_payload())

        req = _make_processed_req(
            expected_outcome="account is permanently removed from the system"
        )
        si.identify(req)

        call_args = mock_fn.call_args
        prompt_sent = call_args.kwargs.get("contents", "")
        assert "account is permanently removed from the system" in prompt_sent
