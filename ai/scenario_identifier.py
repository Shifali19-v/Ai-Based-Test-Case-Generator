"""
AI-Based Test Case Generator
Phase 3: Scenario Identifier

Takes a ProcessedRequirement (from Phase 2) and asks Google Gemini to
identify the important testing scenarios grouped into four categories.
"""

import os
import json
import time

from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from ai.requirement_processor import ProcessedRequirement

# Load environment variables from the project .env file
load_dotenv()


class ScenarioSet(BaseModel):
    """
    A grouped set of testing scenarios derived from a single requirement.
    """

    positive_scenarios: list[str]
    negative_scenarios: list[str]
    boundary_scenarios: list[str]
    edge_cases: list[str]


class ScenarioIdentifier:
    """
    Identifies testing scenarios for a given ProcessedRequirement
    using Google Gemini.
    """

    MODEL_NAME = "gemini-flash-lite-latest"

    _PROMPT_TEMPLATE = """
You are a senior QA engineer.

Based on the structured software requirement below, identify testing scenarios.

IMPORTANT RULES:
- Base scenarios ONLY on the information provided. Do NOT invent numeric limits,
  business rules, or constraints that are not explicitly stated.
- If no boundary conditions are mentioned, return an empty array for
  "boundary_scenarios".
- Keep each scenario as a concise single sentence.

Structured Requirement:
  Actor           : {actor}
  Action          : {action}
  Conditions      : {conditions}
  Inputs          : {inputs}
  Expected Outcome: {expected_outcome}

Return ONLY a valid JSON object with exactly these four fields:
  - "positive_scenarios" : array of strings (happy-path scenarios that should succeed)
  - "negative_scenarios" : array of strings (scenarios with invalid input or state)
  - "boundary_scenarios" : array of strings (scenarios at stated boundary conditions; empty if none stated)
  - "edge_cases"         : array of strings (unusual or unexpected situations)

Do NOT include any explanation, markdown, or extra text. Return raw JSON only.
""".strip()

    def __init__(self):
        """
        Initialise the identifier.
        """

        self._api_key = os.getenv("GEMINI_API_KEY")

        if not self._api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Please add it to your .env file."
            )

        self._client = genai.Client(api_key=self._api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify(
        self,
        processed_req: ProcessedRequirement
    ) -> ScenarioSet:
        """
        Identify testing scenarios for the given processed requirement.
        """

        self._validate_input(processed_req)

        raw_json = self._call_gemini(processed_req)

        return self._parse_response(raw_json)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_input(
        self,
        processed_req: ProcessedRequirement
    ) -> None:
        """Raise TypeError if the input is not a ProcessedRequirement."""

        if not isinstance(processed_req, ProcessedRequirement):
            raise TypeError(
                f"Expected a ProcessedRequirement instance, "
                f"got {type(processed_req).__name__!r}."
            )

    def _call_gemini(
        self,
        processed_req: ProcessedRequirement
    ) -> str:
        """
        Build the prompt and call Gemini.

        Retries temporary 503 / UNAVAILABLE errors up to 3 times.
        """

        prompt = self._PROMPT_TEMPLATE.format(
            actor=processed_req.actor,
            action=processed_req.action,
            conditions=", ".join(processed_req.conditions) or "none",
            inputs=", ".join(processed_req.inputs) or "none",
            expected_outcome=processed_req.expected_outcome,
        )

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

                # Retry only temporary Gemini availability errors
                if (
                    "503" not in error_message
                    and "UNAVAILABLE" not in error_message
                ):
                    raise

                # Final attempt: raise the original error
                if attempt == max_retries - 1:
                    raise

                # Wait 1 second, then 2 seconds
                time.sleep(2 ** attempt)

        raise RuntimeError(
            "Unable to identify testing scenarios."
        )

    def _parse_response(
        self,
        raw_text: str
    ) -> ScenarioSet:
        """
        Parse and validate the raw JSON string from Gemini.
        """

        cleaned = raw_text.strip()

        # Strip markdown code fences if Gemini wraps its JSON
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = [
                line
                for line in lines
                if not line.strip().startswith("```")
            ]
            cleaned = "\n".join(lines).strip()

        # Parse JSON
        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini response is not valid JSON: {exc}\n"
                f"Raw response was: {raw_text!r}"
            ) from exc

        # Validate with Pydantic
        try:
            return ScenarioSet(**data)

        except ValidationError as exc:
            raise ValueError(
                f"Gemini response is missing required fields: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"ScenarioIdentifier("
            f"model={self.MODEL_NAME!r}, "
            f"sdk='google-genai')"
        )
