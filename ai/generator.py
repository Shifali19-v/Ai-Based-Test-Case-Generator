"""
AI-Based Test Case Generator
Phase 1 / Phase 4: Test Case Generator

Defines the TestCaseGenerator class, which uses Google Gemini to produce
structured TestCase objects from a feature description. Phase 4 implements
the generate() and _initialize_client() methods that were placeholders in
Phase 1. Optional ProcessedRequirement and ScenarioSet context (from
Phases 2 and 3) can be passed to generate() to enrich the prompt.
"""

import os
import json
import time

from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Optional

# Load environment variables from .env
load_dotenv()


class TestCase(BaseModel):
    """Represents a single generated test case."""

    title: str
    description: str
    steps: list[str]
    expected_result: str
    priority: str = "Medium"  # Low | Medium | High


class GeneratorConfig(BaseModel):
    """Configuration for the TestCaseGenerator."""

    model_name: str = "gemini-flash-latest"
    max_test_cases: int = 10
    temperature: float = 0.7


class TestCaseGenerator:
    """
    AI-powered test case generator using Google Gemini.

    Accepts an optional ProcessedRequirement (Phase 2) and ScenarioSet
    (Phase 3) to produce richer, context-aware test cases.
    """

    MODEL_NAME = "gemini-flash-latest"

    _PROMPT_BASE = """
You are a senior QA engineer.

Generate test cases for the following software requirement.

Return ONLY a valid JSON array of up to {max_test_cases} test case objects.
Each object must have exactly these fields:
  - "title"           : string  (short, descriptive test case name)
  - "description"     : string  (what is being tested)
  - "steps"           : array of strings (ordered test steps)
  - "expected_result" : string  (what the system should do)
  - "priority"        : "Low", "Medium", or "High"

Do NOT include any explanation, markdown, or extra text. Return raw JSON array only.

Requirement:
{feature_description}
""".strip()

    _PROMPT_WITH_CONTEXT = """
You are a senior QA engineer.

Generate test cases using the structured analysis below.

Structured Analysis:
  Actor           : {actor}
  Action          : {action}
  Conditions      : {conditions}
  Inputs          : {inputs}
  Expected Outcome: {expected_outcome}

Scenarios to cover:
  Positive  : {positive}
  Negative  : {negative}
  Boundary  : {boundary}
  Edge Cases: {edge_cases}

Return ONLY a valid JSON array of up to {max_test_cases} test case objects.
Each object must have exactly these fields:
  - "title"           : string  (short, descriptive test case name)
  - "description"     : string  (what is being tested)
  - "steps"           : array of strings (ordered test steps)
  - "expected_result" : string  (what the system should do)
  - "priority"        : "Low", "Medium", or "High"

Do NOT include any explanation, markdown, or extra text. Return raw JSON array only.

Requirement:
{feature_description}
""".strip()

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialise the generator.
        """
        self.config = config or GeneratorConfig()
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Please add it to your .env file."
            )

        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialise the Google Gemini client."""
        self._client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        feature_description: str,
        processed_req=None,
        scenarios=None,
    ) -> list[TestCase]:

        if not self.validate_input(feature_description):
            raise ValueError(
                "feature_description must be a non-empty string "
                "with at least 10 characters."
            )

        raw_json = self._call_gemini(
            feature_description,
            processed_req,
            scenarios,
        )

        return self._parse_response(raw_json)

    def validate_input(self, feature_description: str) -> bool:
        """Validate that the input description is usable."""

        if not isinstance(feature_description, str):
            return False

        stripped = feature_description.strip()
        return len(stripped) >= 10

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        feature_description: str,
        processed_req=None,
        scenarios=None,
    ) -> str:

        if processed_req is not None and scenarios is not None:
            prompt = self._PROMPT_WITH_CONTEXT.format(
                actor=processed_req.actor,
                action=processed_req.action,
                conditions=", ".join(processed_req.conditions) or "none",
                inputs=", ".join(processed_req.inputs) or "none",
                expected_outcome=processed_req.expected_outcome,
                positive=", ".join(scenarios.positive_scenarios) or "none",
                negative=", ".join(scenarios.negative_scenarios) or "none",
                boundary=", ".join(scenarios.boundary_scenarios) or "none",
                edge_cases=", ".join(scenarios.edge_cases) or "none",
                max_test_cases=self.config.max_test_cases,
                feature_description=feature_description,
            )
        else:
            prompt = self._PROMPT_BASE.format(
                max_test_cases=self.config.max_test_cases,
                feature_description=feature_description,
            )

        # Retry temporary Gemini availability errors
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=prompt,
                )

                raw_text = response.text.strip() if response.text else ""

                if not raw_text:
                    raise ValueError(
                        "Gemini returned an empty response. "
                        "Check your API key and model availability."
                    )

                return raw_text

            except Exception as exc:
                error_message = str(exc)

                # Retry only temporary availability/high-demand errors
                if "503" not in error_message and "UNAVAILABLE" not in error_message:
                    raise

                if attempt == max_retries - 1:
                    raise

                # Wait before trying again
                time.sleep(2 ** attempt)

        raise RuntimeError("Unable to generate test cases.")

    def _parse_response(self, raw_text: str) -> list[TestCase]:
        """Strip fences, parse JSON array, validate each item with Pydantic."""

        cleaned = raw_text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = [
                line for line in lines
                if not line.strip().startswith("```")
            ]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini response is not valid JSON: {exc}\n"
                f"Raw response was: {raw_text!r}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Gemini response must be a JSON array, "
                f"got: {type(data).__name__}"
            )

        test_cases = []

        for i, item in enumerate(data):
            try:
                test_cases.append(TestCase(**item))

            except (ValidationError, TypeError) as exc:
                raise ValueError(
                    f"Test case #{i + 1} has missing or invalid fields: {exc}"
                ) from exc

        return test_cases

    def __repr__(self) -> str:
        return (
            f"TestCaseGenerator("
            f"model={self.config.model_name!r}, "
            f"max_test_cases={self.config.max_test_cases})"
        )
