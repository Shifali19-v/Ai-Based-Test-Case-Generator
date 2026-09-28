"""
AI-Based Test Case Generator
Phase 2: Requirement Processor

This module accepts a plain-text software requirement, sends it to
Google Gemini, and returns a structured ProcessedRequirement object
containing: actor, action, conditions, inputs, and expected_outcome.
"""

import os
import json
import time
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Load environment variables from the project .env file
load_dotenv()


class ProcessedRequirement(BaseModel):
    """
    Structured representation of a parsed software requirement.
    """

    actor: str
    action: str
    conditions: list[str]
    inputs: list[str]
    expected_outcome: str


class RequirementProcessor:
    """
    Processes a plain-text software requirement using Google Gemini
    and returns a structured ProcessedRequirement.
    """

    MODEL_NAME = "gemini-flash-latest"

    _PROMPT_TEMPLATE = """
You are a software requirements analyst.

Analyze the following software requirement and extract its key components.
Return ONLY a valid JSON object with exactly these five fields:
  - "actor"            : string  (who performs the action)
  - "action"           : string  (what they do)
  - "conditions"       : array of strings (pre-conditions or constraints; empty array if none)
  - "inputs"           : array of strings (data provided by the actor; empty array if none)
  - "expected_outcome" : string  (what the system does in response)

Do NOT include any explanation, markdown, or extra text. Return raw JSON only.

Requirement:
{requirement}
""".strip()

    def __init__(self):
        """
        Initialise the processor.
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

    def process(self, requirement: str) -> ProcessedRequirement:
        """
        Parse a plain-text requirement into a ProcessedRequirement.
        """

        self._validate_requirement(requirement)

        raw_json = self._call_gemini(requirement)

        return self._parse_response(raw_json)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_requirement(self, requirement: str) -> None:
        """Raise ValueError if the requirement string is unusable."""

        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError(
                "Requirement must be a non-empty string."
            )

    def _call_gemini(self, requirement: str) -> str:
        """
        Send the requirement to Gemini and return the raw text response.

        Retries temporary 503 / UNAVAILABLE errors up to 3 times.
        """

        prompt = self._PROMPT_TEMPLATE.format(
            requirement=requirement
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

                # If this was the final attempt, raise the error
                if attempt == max_retries - 1:
                    raise

                # Wait before retrying: 1s, then 2s
                time.sleep(2 ** attempt)

        raise RuntimeError(
            "Unable to generate requirement analysis."
        )

    def _parse_response(
        self,
        raw_text: str
    ) -> ProcessedRequirement:
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
            return ProcessedRequirement(**data)

        except ValidationError as exc:
            raise ValueError(
                f"Gemini response is missing required fields: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"RequirementProcessor("
            f"model={self.MODEL_NAME!r}, "
            f"sdk='google-genai')"
        )
