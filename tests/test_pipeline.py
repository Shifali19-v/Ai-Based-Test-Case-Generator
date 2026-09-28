"""
AI-Based Test Case Generator
Phase 4: Unit Tests for pipeline.py

All tests mock the three component classes — no real Gemini API calls.

The pipeline is tested by patching:
  - ai.pipeline.RequirementProcessor
  - ai.pipeline.ScenarioIdentifier
  - ai.pipeline.TestCaseGenerator

Test classes:
  TestPipelineResultModel   - PipelineResult data-holder
  TestPipelineInit          - __init__ wires all three components
  TestValidateInput         - _validate_input() edge cases
  TestPipelineGenerate      - end-to-end generate() flow and error paths
  TestStageDataPassing      - verifies data flows correctly between stages
"""

import os
import pytest
from unittest.mock import patch, MagicMock, call

os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-key")

from ai.requirement_processor import ProcessedRequirement
from ai.scenario_identifier import ScenarioSet
from ai.generator import TestCase, GeneratorConfig
from ai.pipeline import TestCasePipeline, PipelineResult


# ---------------------------------------------------------------------------
# Shared test-data helpers
# ---------------------------------------------------------------------------

def _make_processed_req(**overrides) -> ProcessedRequirement:
    base = {
        "actor": "registered user",
        "action": "submit the login form",
        "conditions": ["user has a valid account"],
        "inputs": ["email address", "password"],
        "expected_outcome": "user is authenticated and redirected to the dashboard",
    }
    base.update(overrides)
    return ProcessedRequirement(**base)


def _make_scenario_set(**overrides) -> ScenarioSet:
    base = {
        "positive_scenarios": ["Login succeeds with valid credentials."],
        "negative_scenarios": ["Login fails with wrong password."],
        "boundary_scenarios": [],
        "edge_cases": ["Login attempted with SQL injection in email field."],
    }
    base.update(overrides)
    return ScenarioSet(**base)


def _make_test_case(**overrides) -> TestCase:
    base = {
        "title": "TC-001: Valid Login",
        "description": "Verify login with correct credentials.",
        "steps": ["Go to /login", "Enter valid email", "Enter valid password", "Click Submit"],
        "expected_result": "User lands on dashboard.",
        "priority": "High",
    }
    base.update(overrides)
    return TestCase(**base)


def _patch_all():
    """Context manager that patches all three pipeline components."""
    return patch.multiple(
        "ai.pipeline",
        RequirementProcessor=MagicMock,
        ScenarioIdentifier=MagicMock,
        TestCaseGenerator=MagicMock,
    )


def _make_pipeline_with_mocks(
    proc_return=None,
    identifier_return=None,
    generator_return=None,
):
    """
    Return (pipeline, mock_proc_instance, mock_id_instance, mock_gen_instance).
    Default return values produce a valid, happy-path pipeline run.
    """
    proc_ret = proc_return or _make_processed_req()
    id_ret = identifier_return or _make_scenario_set()
    gen_ret = generator_return or [_make_test_case()]

    mock_proc_cls = MagicMock()
    mock_id_cls = MagicMock()
    mock_gen_cls = MagicMock()

    mock_proc_cls.return_value.process.return_value = proc_ret
    mock_id_cls.return_value.identify.return_value = id_ret
    mock_gen_cls.return_value.generate.return_value = gen_ret

    with patch("ai.pipeline.RequirementProcessor", mock_proc_cls), \
         patch("ai.pipeline.ScenarioIdentifier", mock_id_cls), \
         patch("ai.pipeline.TestCaseGenerator", mock_gen_cls):
        pipeline = TestCasePipeline()

    return pipeline, mock_proc_cls.return_value, mock_id_cls.return_value, mock_gen_cls.return_value


# ---------------------------------------------------------------------------
# PipelineResult model tests
# ---------------------------------------------------------------------------

class TestPipelineResultModel:
    def test_result_holds_all_outputs(self):
        req_text = "A user can log in."
        pr = _make_processed_req()
        ss = _make_scenario_set()
        tcs = [_make_test_case()]

        result = PipelineResult(
            requirement_text=req_text,
            processed_req=pr,
            scenarios=ss,
            test_cases=tcs,
        )

        assert result.requirement_text == req_text
        assert result.processed_req is pr
        assert result.scenarios is ss
        assert result.test_cases is tcs

    def test_result_repr_contains_counts(self):
        result = PipelineResult(
            requirement_text="some requirement",
            processed_req=_make_processed_req(),
            scenarios=_make_scenario_set(),
            test_cases=[_make_test_case(), _make_test_case()],
        )
        r = repr(result)
        assert "test_cases=2" in r

    def test_result_test_cases_is_list(self):
        result = PipelineResult(
            requirement_text="req",
            processed_req=_make_processed_req(),
            scenarios=_make_scenario_set(),
            test_cases=[],
        )
        assert isinstance(result.test_cases, list)


# ---------------------------------------------------------------------------
# TestCasePipeline.__init__ tests
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_pipeline_creates_all_three_components(self):
        mock_proc_cls = MagicMock()
        mock_id_cls = MagicMock()
        mock_gen_cls = MagicMock()

        with patch("ai.pipeline.RequirementProcessor", mock_proc_cls), \
             patch("ai.pipeline.ScenarioIdentifier", mock_id_cls), \
             patch("ai.pipeline.TestCaseGenerator", mock_gen_cls):
            pipeline = TestCasePipeline()

        mock_proc_cls.assert_called_once()
        mock_id_cls.assert_called_once()
        mock_gen_cls.assert_called_once()
        assert pipeline._processor is mock_proc_cls.return_value
        assert pipeline._identifier is mock_id_cls.return_value
        assert pipeline._generator is mock_gen_cls.return_value

    def test_pipeline_passes_config_to_generator(self):
        config = GeneratorConfig(max_test_cases=5)
        mock_gen_cls = MagicMock()

        with patch("ai.pipeline.RequirementProcessor"), \
             patch("ai.pipeline.ScenarioIdentifier"), \
             patch("ai.pipeline.TestCaseGenerator", mock_gen_cls):
            TestCasePipeline(config=config)

        mock_gen_cls.assert_called_once_with(config=config)

    def test_pipeline_repr_contains_component_reprs(self):
        pipeline, _, _, _ = _make_pipeline_with_mocks()
        r = repr(pipeline)
        assert "TestCasePipeline" in r


# ---------------------------------------------------------------------------
# _validate_input() tests
# ---------------------------------------------------------------------------

class TestValidateInput:
    def setup_method(self):
        self.pipeline, *_ = _make_pipeline_with_mocks()

    def test_valid_string_passes(self):
        # Should not raise
        self.pipeline._validate_input("A valid requirement string.")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self.pipeline._validate_input("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self.pipeline._validate_input("    ")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self.pipeline._validate_input(None)  # type: ignore

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self.pipeline._validate_input(123)  # type: ignore


# ---------------------------------------------------------------------------
# generate() end-to-end tests
# ---------------------------------------------------------------------------

class TestPipelineGenerate:

    # --- Happy path ---

    def test_generate_returns_pipeline_result(self):
        pipeline, _, _, _ = _make_pipeline_with_mocks()
        result = pipeline.generate(
            "A registered user should be able to log in using email and password."
        )
        assert isinstance(result, PipelineResult)

    def test_generate_result_contains_test_cases(self):
        tc = _make_test_case()
        pipeline, _, _, _ = _make_pipeline_with_mocks(generator_return=[tc])
        result = pipeline.generate(
            "A registered user should be able to log in using email and password."
        )
        assert len(result.test_cases) == 1
        assert result.test_cases[0].title == "TC-001: Valid Login"

    def test_generate_result_preserves_requirement_text(self):
        pipeline, _, _, _ = _make_pipeline_with_mocks()
        req = "User should be able to reset their password via email link."
        result = pipeline.generate(req)
        assert result.requirement_text == req

    def test_generate_result_contains_processed_req(self):
        pr = _make_processed_req(actor="admin")
        pipeline, _, _, _ = _make_pipeline_with_mocks(proc_return=pr)
        result = pipeline.generate(
            "Admin should be able to delete user accounts from the system."
        )
        assert result.processed_req.actor == "admin"

    def test_generate_result_contains_scenarios(self):
        ss = _make_scenario_set(
            positive_scenarios=["User resets password successfully."]
        )
        pipeline, _, _, _ = _make_pipeline_with_mocks(identifier_return=ss)
        result = pipeline.generate(
            "User should be able to reset their password via email link."
        )
        assert "User resets password successfully." in result.scenarios.positive_scenarios

    # --- Each component is called ---

    def test_requirement_processor_is_called(self):
        pipeline, mock_proc, _, _ = _make_pipeline_with_mocks()
        req = "A registered user should be able to log in using email and password."
        pipeline.generate(req)
        mock_proc.process.assert_called_once_with(req)

    def test_scenario_identifier_is_called(self):
        pr = _make_processed_req()
        pipeline, _, mock_id, _ = _make_pipeline_with_mocks(proc_return=pr)
        pipeline.generate(
            "A registered user should be able to log in using email and password."
        )
        mock_id.identify.assert_called_once_with(pr)

    def test_test_case_generator_is_called(self):
        pr = _make_processed_req()
        ss = _make_scenario_set()
        pipeline, _, _, mock_gen = _make_pipeline_with_mocks(
            proc_return=pr, identifier_return=ss
        )
        req = "A registered user should be able to log in using email and password."
        pipeline.generate(req)
        mock_gen.generate.assert_called_once_with(
            feature_description=req,
            processed_req=pr,
            scenarios=ss,
        )

    def test_each_stage_called_exactly_once(self):
        pipeline, mock_proc, mock_id, mock_gen = _make_pipeline_with_mocks()
        pipeline.generate(
            "A registered user should be able to log in using email and password."
        )
        assert mock_proc.process.call_count == 1
        assert mock_id.identify.call_count == 1
        assert mock_gen.generate.call_count == 1

    # --- Error paths ---

    def test_empty_requirement_raises_before_any_stage(self):
        pipeline, mock_proc, mock_id, mock_gen = _make_pipeline_with_mocks()
        with pytest.raises(ValueError, match="non-empty string"):
            pipeline.generate("")
        mock_proc.process.assert_not_called()
        mock_id.identify.assert_not_called()
        mock_gen.generate.assert_not_called()

    def test_whitespace_requirement_raises_before_any_stage(self):
        pipeline, mock_proc, mock_id, mock_gen = _make_pipeline_with_mocks()
        with pytest.raises(ValueError, match="non-empty string"):
            pipeline.generate("   ")
        mock_proc.process.assert_not_called()

    def test_processor_error_propagates(self):
        pipeline, mock_proc, _, _ = _make_pipeline_with_mocks()
        mock_proc.process.side_effect = ValueError("Gemini returned an empty response.")
        with pytest.raises(ValueError, match="empty response"):
            pipeline.generate(
                "A registered user should be able to log in using email and password."
            )

    def test_identifier_error_propagates(self):
        pipeline, _, mock_id, _ = _make_pipeline_with_mocks()
        mock_id.identify.side_effect = ValueError("Gemini response is not valid JSON")
        with pytest.raises(ValueError, match="not valid JSON"):
            pipeline.generate(
                "A registered user should be able to log in using email and password."
            )

    def test_generator_error_propagates(self):
        pipeline, _, _, mock_gen = _make_pipeline_with_mocks()
        mock_gen.generate.side_effect = ValueError("Test case #1 has missing or invalid fields")
        with pytest.raises(ValueError, match="missing or invalid fields"):
            pipeline.generate(
                "A registered user should be able to log in using email and password."
            )

    def test_none_input_raises(self):
        pipeline, *_ = _make_pipeline_with_mocks()
        with pytest.raises(ValueError, match="non-empty string"):
            pipeline.generate(None)  # type: ignore


# ---------------------------------------------------------------------------
# Data-passing between stages
# ---------------------------------------------------------------------------

class TestStageDataPassing:
    def test_processor_output_is_passed_to_identifier(self):
        """The exact ProcessedRequirement returned by the processor must reach the identifier."""
        pr = _make_processed_req(actor="system administrator")
        pipeline, _, mock_id, _ = _make_pipeline_with_mocks(proc_return=pr)

        pipeline.generate(
            "System administrator should be able to deactivate any user account."
        )

        args, _ = mock_id.identify.call_args
        assert args[0] is pr
        assert args[0].actor == "system administrator"

    def test_identifier_output_is_passed_to_generator(self):
        """The exact ScenarioSet returned by the identifier must reach the generator."""
        ss = _make_scenario_set(
            positive_scenarios=["Admin deactivates account successfully."]
        )
        pipeline, _, _, mock_gen = _make_pipeline_with_mocks(identifier_return=ss)

        req = "Admin should be able to deactivate user accounts from the dashboard."
        pipeline.generate(req)

        _, kwargs = mock_gen.generate.call_args
        assert kwargs["scenarios"] is ss
        assert "Admin deactivates account successfully." in kwargs["scenarios"].positive_scenarios

    def test_original_requirement_text_is_passed_to_generator(self):
        """The unmodified requirement string must be forwarded to the generator."""
        pipeline, _, _, mock_gen = _make_pipeline_with_mocks()
        req = "User should be able to register with a unique email address."
        pipeline.generate(req)

        _, kwargs = mock_gen.generate.call_args
        assert kwargs["feature_description"] == req
