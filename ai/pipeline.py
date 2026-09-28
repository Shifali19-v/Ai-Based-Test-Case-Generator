"""
AI-Based Test Case Generator
Phase 4: Pipeline Integration

Connects the three AI components into a single, easy-to-use pipeline:

  requirement_text
        ↓
  RequirementProcessor   (Phase 2) – extracts structured information
        ↓
  ScenarioIdentifier     (Phase 3) – identifies testing scenarios
        ↓
  TestCaseGenerator      (Phase 1/4) – generates final test cases
        ↓
  list[TestCase]

Usage:
    pipeline = TestCasePipeline()
    test_cases = pipeline.generate(
        "A registered user should be able to log in using email and password."
    )
    for tc in test_cases:
        print(tc.title, tc.priority)
"""

from ai.requirement_processor import RequirementProcessor, ProcessedRequirement
from ai.scenario_identifier import ScenarioIdentifier, ScenarioSet
from ai.generator import TestCaseGenerator, TestCase, GeneratorConfig
from typing import Optional


class PipelineResult:
    """
    Holds the full output of a TestCasePipeline run.

    Attributes:
        requirement_text  - Original input string
        processed_req     - Structured requirement from Phase 2
        scenarios         - Identified scenarios from Phase 3
        test_cases        - Final generated test cases from Phase 4
    """

    def __init__(
        self,
        requirement_text: str,
        processed_req: ProcessedRequirement,
        scenarios: ScenarioSet,
        test_cases: list[TestCase],
    ):
        self.requirement_text = requirement_text
        self.processed_req = processed_req
        self.scenarios = scenarios
        self.test_cases = test_cases

    def __repr__(self) -> str:
        return (
            f"PipelineResult("
            f"test_cases={len(self.test_cases)}, "
            f"positive={len(self.scenarios.positive_scenarios)}, "
            f"negative={len(self.scenarios.negative_scenarios)})"
        )


class TestCasePipeline:
    """
    End-to-end pipeline that takes a plain-text requirement and produces
    fully structured, AI-generated test cases.

    Each stage is handled by a dedicated Phase module:
      - Stage 1: RequirementProcessor   (Phase 2)
      - Stage 2: ScenarioIdentifier     (Phase 3)
      - Stage 3: TestCaseGenerator      (Phase 1 / Phase 4)

    Errors from any individual stage propagate as-is so the caller can
    handle them specifically.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialise all three pipeline components.

        Args:
            config: Optional GeneratorConfig for the TestCaseGenerator.

        Raises:
            EnvironmentError: If GEMINI_API_KEY is not set.
        """
        self._processor = RequirementProcessor()
        self._identifier = ScenarioIdentifier()
        self._generator = TestCaseGenerator(config=config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, requirement_text: str) -> PipelineResult:
        """
        Run the full pipeline for a single requirement.

        Args:
            requirement_text: A plain-English software requirement.

        Returns:
            A PipelineResult containing all intermediate and final outputs.

        Raises:
            ValueError:       If requirement_text is empty or any stage
                              receives an invalid / malformed AI response.
            EnvironmentError: If GEMINI_API_KEY is not set.
            TypeError:        If data between stages has an unexpected type.
        """
        self._validate_input(requirement_text)

        # Stage 1 – extract structured information
        processed_req: ProcessedRequirement = self._processor.process(requirement_text)

        # Stage 2 – identify testing scenarios
        scenarios: ScenarioSet = self._identifier.identify(processed_req)

        # Stage 3 – generate test cases using all available context
        test_cases: list[TestCase] = self._generator.generate(
            feature_description=requirement_text,
            processed_req=processed_req,
            scenarios=scenarios,
        )

        return PipelineResult(
            requirement_text=requirement_text,
            processed_req=processed_req,
            scenarios=scenarios,
            test_cases=test_cases,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_input(self, requirement_text: str) -> None:
        """Raise ValueError if the input is unusable."""
        if not isinstance(requirement_text, str) or not requirement_text.strip():
            raise ValueError(
                "requirement_text must be a non-empty string."
            )

    def __repr__(self) -> str:
        return (
            f"TestCasePipeline("
            f"processor={self._processor!r}, "
            f"identifier={self._identifier!r}, "
            f"generator={self._generator!r})"
        )
