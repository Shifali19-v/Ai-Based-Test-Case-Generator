"""
AI-Based Test Case Generator
Phase 1: Unit Tests for generator.py

Tests cover:
  - TestCase and GeneratorConfig Pydantic model validation
  - TestCaseGenerator initialization (with and without API key)
  - validate_input() edge cases
  - Placeholder NotImplementedError for Phase 2 methods
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure a fake API key is always present so import doesn't fail
os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-key")

from ai.generator import TestCase, GeneratorConfig, TestCaseGenerator


# ---------------------------------------------------------------------------
# TestCase model tests
# ---------------------------------------------------------------------------

class TestTestCaseModel:
    def test_valid_test_case(self):
        tc = TestCase(
            title="Login with valid credentials",
            description="Verify that a user can log in with correct details.",
            steps=["Navigate to /login", "Enter valid email", "Enter valid password", "Click Submit"],
            expected_result="User is redirected to the dashboard.",
        )
        assert tc.title == "Login with valid credentials"
        assert tc.priority == "Medium"  # default value

    def test_custom_priority(self):
        tc = TestCase(
            title="Password reset",
            description="Verify password reset flow.",
            steps=["Click forgot password"],
            expected_result="Reset email is sent.",
            priority="High",
        )
        assert tc.priority == "High"

    def test_missing_required_fields_raises(self):
        with pytest.raises(Exception):
            TestCase(title="Missing description")  # description/steps/expected_result missing


# ---------------------------------------------------------------------------
# GeneratorConfig model tests
# ---------------------------------------------------------------------------

class TestGeneratorConfig:
    def test_defaults(self):
        config = GeneratorConfig()
        assert config.model_name == "gemini-flash-latest"
        assert config.max_test_cases == 10
        assert config.temperature == 0.7

    def test_custom_values(self):
        config = GeneratorConfig(model_name="gemini-1.5-pro", max_test_cases=5, temperature=0.3)
        assert config.model_name == "gemini-1.5-pro"
        assert config.max_test_cases == 5
        assert config.temperature == 0.3


# ---------------------------------------------------------------------------
# TestCaseGenerator initialization tests
# ---------------------------------------------------------------------------

class TestTestCaseGeneratorInit:
    def test_init_with_valid_api_key(self):
        with patch("ai.generator.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-testing"}):
                gen = TestCaseGenerator()
                assert gen.api_key == "fake-key-for-testing"
                assert gen._client is not None  # client is now initialised

    def test_init_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove GEMINI_API_KEY from environment
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(EnvironmentError, match="GEMINI_API_KEY is not set"):
                TestCaseGenerator()

    def test_default_config(self):
        with patch("ai.generator.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-testing"}):
                gen = TestCaseGenerator()
                assert gen.config.model_name == "gemini-flash-latest"
                assert gen.config.max_test_cases == 10

    def test_custom_config(self):
        with patch("ai.generator.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-testing"}):
                config = GeneratorConfig(max_test_cases=3)
                gen = TestCaseGenerator(config=config)
                assert gen.config.max_test_cases == 3

    def test_repr(self):
        with patch("ai.generator.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-testing"}):
                gen = TestCaseGenerator()
                assert "gemini-flash-latest" in repr(gen)
                assert "10" in repr(gen)


# ---------------------------------------------------------------------------
# validate_input() tests
# ---------------------------------------------------------------------------

class TestValidateInput:
    def setup_method(self):
        with patch("ai.generator.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-testing"}):
                self.gen = TestCaseGenerator()

    def test_valid_input(self):
        assert self.gen.validate_input("User can log in with valid credentials") is True

    def test_input_exactly_10_chars(self):
        assert self.gen.validate_input("1234567890") is True

    def test_input_too_short(self):
        assert self.gen.validate_input("short") is False

    def test_empty_string(self):
        assert self.gen.validate_input("") is False

    def test_whitespace_only(self):
        assert self.gen.validate_input("         ") is False

    def test_non_string_input(self):
        assert self.gen.validate_input(12345) is False  # type: ignore

    def test_none_input(self):
        assert self.gen.validate_input(None) is False  # type: ignore


# ---------------------------------------------------------------------------
# TestCaseGenerator.generate() implementation tests (Phase 4)
# ---------------------------------------------------------------------------

class TestGenerate:
    def setup_method(self):
        with patch("ai.generator.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-testing"}):
                self.gen = TestCaseGenerator()

    def _valid_payload(self):
        return [
            {
                "title": "Login with valid credentials",
                "description": "Verify successful login.",
                "steps": ["Open /login", "Enter valid email", "Enter password", "Click Submit"],
                "expected_result": "User is redirected to dashboard.",
                "priority": "High",
            }
        ]

    def test_generate_returns_list_of_test_cases(self):
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(self._valid_payload())
        self.gen._client.models.generate_content = MagicMock(return_value=mock_resp)

        result = self.gen.generate("User should be able to log in with email and password.")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].title == "Login with valid credentials"
        assert result[0].priority == "High"

    def test_generate_raises_on_short_description(self):
        with pytest.raises(ValueError, match="at least 10 characters"):
            self.gen.generate("short")

    def test_generate_raises_on_empty_description(self):
        with pytest.raises(ValueError, match="at least 10 characters"):
            self.gen.generate("")

    def test_generate_raises_on_empty_gemini_response(self):
        mock_resp = MagicMock()
        mock_resp.text = ""
        self.gen._client.models.generate_content = MagicMock(return_value=mock_resp)

        with pytest.raises(ValueError, match="empty response"):
            self.gen.generate("User should be able to log in with valid credentials.")

    def test_generate_raises_on_malformed_json(self):
        mock_resp = MagicMock()
        mock_resp.text = "Not JSON at all."
        self.gen._client.models.generate_content = MagicMock(return_value=mock_resp)

        with pytest.raises(ValueError, match="not valid JSON"):
            self.gen.generate("User should be able to log in with valid credentials.")

    def test_generate_raises_when_response_is_object_not_array(self):
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({"title": "something"})  # object, not array
        self.gen._client.models.generate_content = MagicMock(return_value=mock_resp)

        with pytest.raises(ValueError, match="JSON array"):
            self.gen.generate("User should be able to log in with valid credentials.")

    def test_initialize_client_sets_client(self):
        """_initialize_client() populates self._client."""
        with patch("ai.generator.genai.Client") as mock_cls:
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
                gen = TestCaseGenerator()
                mock_cls.assert_called_once_with(api_key="fake-key")
                assert gen._client is not None
