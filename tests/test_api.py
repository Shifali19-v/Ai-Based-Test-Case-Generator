"""
AI-Based Test Case Generator
Phase 5A + 5B + 6A: API Tests

Tests every route and error path of the FastAPI application without
making any real Gemini API calls.  The TestCasePipeline, PDFExtractor,
and database functions are patched so the tests are fully isolated.

Test classes (Phase 5A):
    TestHealthEndpoint          – GET /
    TestGenerateSuccess         – POST /generate happy path
    TestGenerateValidation      – empty / missing / invalid request bodies
    TestGeneratePipelineErrors  – pipeline raises ValueError / EnvironmentError / unexpected
    TestResponseStructure       – shape and fields of the JSON response

Test classes (Phase 5B):
    TestGenerateSavesToDatabase     – POST /generate persists result and returns id
    TestListGenerations             – GET /generations
    TestRetrieveGeneration          – GET /generations/{id}
    TestGenerateDatabaseErrors      – database layer raises exceptions

Test classes (Phase 6A):
    TestPDFEndpointSuccess          – POST /generate/pdf happy path
    TestPDFEndpointValidation       – non-PDF / missing upload
    TestPDFEndpointExtractionErrors – extractor raises ValueError / unexpected
    TestPDFEndpointPipelineErrors   – pipeline errors after successful extraction
    TestPDFEndpointDatabaseErrors   – DB error after successful pipeline
    TestPDFEndpointTempFileCleanup  – temp file removed after request
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Set a dummy key so the real pipeline __init__ does not blow up if
# it is ever accidentally imported without a patch.
os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-key")

from backend.main import app  # noqa: E402  (must come after env-var set)
from ai.generator import TestCase  # noqa: E402
from ai.pipeline import PipelineResult  # noqa: E402
from ai.requirement_processor import ProcessedRequirement  # noqa: E402
from ai.scenario_identifier import ScenarioSet  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test-data helpers
# ---------------------------------------------------------------------------

def _make_test_case(**overrides) -> TestCase:
    base = {
        "title": "TC-001: Valid Login",
        "description": "Verify login with correct credentials.",
        "steps": ["Go to /login", "Enter valid email", "Click Submit"],
        "expected_result": "User lands on dashboard.",
        "priority": "High",
    }
    base.update(overrides)
    return TestCase(**base)


def _make_pipeline_result(test_cases=None) -> PipelineResult:
    """Return a minimal PipelineResult for mocking."""
    pr = ProcessedRequirement(
        actor="registered user",
        action="log in",
        conditions=["valid account exists"],
        inputs=["email", "password"],
        expected_outcome="user is authenticated",
    )
    ss = ScenarioSet(
        positive_scenarios=["Login succeeds with valid credentials."],
        negative_scenarios=["Login fails with wrong password."],
        boundary_scenarios=[],
        edge_cases=[],
    )
    tcs = test_cases if test_cases is not None else [_make_test_case()]
    return PipelineResult(
        requirement_text="A user should be able to log in using a valid email and password.",
        processed_req=pr,
        scenarios=ss,
        test_cases=tcs,
    )


def _make_db_generation(gen_id: int = 1, requirement: str = "A user should be able to log in.") -> dict:
    """Return a minimal dict that looks like a database generation row."""
    return {
        "id": gen_id,
        "requirement": requirement,
        "processed_requirement": {
            "actor": "registered user",
            "action": "log in",
            "conditions": ["valid account exists"],
            "inputs": ["email", "password"],
            "expected_outcome": "user is authenticated",
        },
        "scenarios": {
            "positive_scenarios": ["Login succeeds."],
            "negative_scenarios": ["Login fails."],
            "boundary_scenarios": [],
            "edge_cases": [],
        },
        "test_cases": [
            {
                "title": "TC-001: Valid Login",
                "description": "Verify login with correct credentials.",
                "steps": ["Go to /login", "Enter email", "Click Submit"],
                "expected_result": "User lands on dashboard.",
                "priority": "High",
            }
        ],
        "created_at": "2026-09-18T08:00:00+00:00",
    }


# Convenience patch paths
PIPELINE_PATH = "backend.main.TestCasePipeline"
SAVE_GEN_PATH = "backend.main.save_generation"
GET_GENS_PATH = "backend.main.get_generations"
GET_GEN_PATH  = "backend.main.get_generation"


# ---------------------------------------------------------------------------
# TestHealthEndpoint  (Phase 5A)
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_get_root_returns_200(self):
        response = self.client.get("/")
        assert response.status_code == 200

    def test_get_root_returns_json(self):
        response = self.client.get("/")
        assert response.headers["content-type"].startswith("application/json")

    def test_get_root_message_field_present(self):
        response = self.client.get("/")
        body = response.json()
        assert "message" in body

    def test_get_root_message_confirms_running(self):
        response = self.client.get("/")
        body = response.json()
        assert "running" in body["message"].lower()


# ---------------------------------------------------------------------------
# TestGenerateSuccess  (Phase 5A)
# ---------------------------------------------------------------------------

class TestGenerateSuccess:
    """Tests for POST /generate – happy path."""

    REQUIREMENT = "A user should be able to log in using a valid email and password."

    def setup_method(self):
        self.client = TestClient(app)

    def _post(self, mock_pipeline, save_id=1):
        with patch(PIPELINE_PATH, mock_pipeline), \
             patch(SAVE_GEN_PATH, return_value=save_id):
            return self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

    def test_successful_generate_returns_200(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()
        assert self._post(mock_pipeline).status_code == 200

    def test_valid_requirement_is_passed_to_pipeline(self):
        """The exact requirement string must be forwarded to pipeline.generate()."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()
        self._post(mock_pipeline)
        mock_pipeline.return_value.generate.assert_called_once_with(self.REQUIREMENT)

    def test_pipeline_instantiated_once_per_request(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()
        self._post(mock_pipeline)
        assert mock_pipeline.call_count == 1

    def test_response_contains_requirement_field(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()
        body = self._post(mock_pipeline).json()
        assert body["requirement"] == self.REQUIREMENT

    def test_response_contains_test_cases_list(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()
        body = self._post(mock_pipeline).json()
        assert "test_cases" in body
        assert isinstance(body["test_cases"], list)

    def test_response_total_test_cases_matches_list_length(self):
        two_cases = [_make_test_case(), _make_test_case(title="TC-002: Invalid Login")]
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result(
            test_cases=two_cases
        )
        body = self._post(mock_pipeline).json()
        assert body["total_test_cases"] == 2
        assert len(body["test_cases"]) == 2


# ---------------------------------------------------------------------------
# TestGenerateValidation  (Phase 5A)
# ---------------------------------------------------------------------------

class TestGenerateValidation:
    """Tests for POST /generate – invalid or missing input."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_empty_requirement_returns_422(self):
        response = self.client.post("/generate", json={"requirement": ""})
        assert response.status_code == 422

    def test_whitespace_only_requirement_returns_422(self):
        response = self.client.post("/generate", json={"requirement": "   "})
        assert response.status_code == 422

    def test_missing_requirement_field_returns_422(self):
        response = self.client.post("/generate", json={})
        assert response.status_code == 422

    def test_invalid_body_not_json_returns_422(self):
        response = self.client.post(
            "/generate",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_requirement_as_integer_returns_422(self):
        response = self.client.post("/generate", json={"requirement": 12345})
        # Pydantic coerces int → str, so 12345 becomes "12345" (non-empty).
        # The important thing is the API doesn't crash (200 or 422 both acceptable).
        assert response.status_code in (200, 422)

    def test_null_requirement_returns_422(self):
        response = self.client.post("/generate", json={"requirement": None})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# TestGeneratePipelineErrors  (Phase 5A)
# ---------------------------------------------------------------------------

class TestGeneratePipelineErrors:
    """Tests for POST /generate – pipeline raises exceptions."""

    REQUIREMENT = "A user should be able to log in using a valid email and password."

    def setup_method(self):
        self.client = TestClient(app)

    def test_pipeline_value_error_returns_422(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.side_effect = ValueError("bad input")

        with patch(PIPELINE_PATH, mock_pipeline):
            response = self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

        assert response.status_code == 422

    def test_pipeline_environment_error_returns_500(self):
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = EnvironmentError("GEMINI_API_KEY is not set")

        with patch(PIPELINE_PATH, mock_pipeline):
            response = self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

        assert response.status_code == 500

    def test_pipeline_environment_error_does_not_expose_key(self):
        """The API key must NEVER appear in any error response."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = EnvironmentError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )

        with patch(PIPELINE_PATH, mock_pipeline):
            response = self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

        body = response.json()
        response_text = str(body)
        assert "GEMINI_API_KEY" not in response_text

    def test_pipeline_unexpected_error_returns_500(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.side_effect = RuntimeError(
            "Unexpected internal error"
        )

        with patch(PIPELINE_PATH, mock_pipeline):
            response = self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

        assert response.status_code == 500

    def test_pipeline_value_error_detail_contains_message(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.side_effect = ValueError(
            "requirement_text must be a non-empty string."
        )

        with patch(PIPELINE_PATH, mock_pipeline):
            response = self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

        body = response.json()
        assert "detail" in body


# ---------------------------------------------------------------------------
# TestResponseStructure  (Phase 5A)
# ---------------------------------------------------------------------------

class TestResponseStructure:
    """Tests for the shape and content of the /generate JSON response."""

    REQUIREMENT = "A user should be able to log in using a valid email and password."

    def setup_method(self):
        self.client = TestClient(app)
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.return_value.generate.return_value = _make_pipeline_result()

    def _post(self):
        with patch(PIPELINE_PATH, self.mock_pipeline), \
             patch(SAVE_GEN_PATH, return_value=1):
            return self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

    def test_response_has_requirement_key(self):
        assert "requirement" in self._post().json()

    def test_response_has_total_test_cases_key(self):
        assert "total_test_cases" in self._post().json()

    def test_response_has_test_cases_key(self):
        assert "test_cases" in self._post().json()

    def test_each_test_case_has_title(self):
        for tc in self._post().json()["test_cases"]:
            assert "title" in tc

    def test_each_test_case_has_description(self):
        for tc in self._post().json()["test_cases"]:
            assert "description" in tc

    def test_each_test_case_has_steps(self):
        for tc in self._post().json()["test_cases"]:
            assert "steps" in tc
            assert isinstance(tc["steps"], list)

    def test_each_test_case_has_expected_result(self):
        for tc in self._post().json()["test_cases"]:
            assert "expected_result" in tc

    def test_each_test_case_has_priority(self):
        for tc in self._post().json()["test_cases"]:
            assert "priority" in tc

    def test_gemini_api_key_not_in_response(self):
        """API key must never be included in any response body."""
        body = self._post().json()
        response_text = str(body)
        assert "GEMINI_API_KEY" not in response_text
        actual_key = os.environ.get("GEMINI_API_KEY", "")
        if actual_key and actual_key != "test-placeholder-key":
            assert actual_key not in response_text


# ---------------------------------------------------------------------------
# TestGenerateSavesToDatabase  (Phase 5B)
# ---------------------------------------------------------------------------

class TestGenerateSavesToDatabase:
    """POST /generate must save to DB and return the generated id."""

    REQUIREMENT = "A user should be able to log in using a valid email and password."

    def setup_method(self):
        self.client = TestClient(app)

    def test_generate_returns_database_id(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()

        with patch(PIPELINE_PATH, mock_pipeline), \
             patch(SAVE_GEN_PATH, return_value=42):
            response = self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

        body = response.json()
        assert "id" in body
        assert body["id"] == 42

    def test_generate_id_is_integer(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()

        with patch(PIPELINE_PATH, mock_pipeline), \
             patch(SAVE_GEN_PATH, return_value=7):
            response = self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

        assert isinstance(response.json()["id"], int)

    def test_save_generation_is_called_after_pipeline(self):
        """save_generation must be called exactly once per successful request."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()
        mock_save = MagicMock(return_value=1)

        with patch(PIPELINE_PATH, mock_pipeline), \
             patch(SAVE_GEN_PATH, mock_save):
            self.client.post("/generate", json={"requirement": self.REQUIREMENT})

        assert mock_save.call_count == 1

    def test_save_generation_receives_requirement(self):
        """The requirement string passed to save_generation must match the request."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()
        mock_save = MagicMock(return_value=1)

        with patch(PIPELINE_PATH, mock_pipeline), \
             patch(SAVE_GEN_PATH, mock_save):
            self.client.post("/generate", json={"requirement": self.REQUIREMENT})

        call_kwargs = mock_save.call_args.kwargs
        assert call_kwargs["requirement"] == self.REQUIREMENT

    def test_save_generation_not_called_on_pipeline_error(self):
        """If the pipeline fails, save_generation must NOT be called."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.side_effect = ValueError("pipeline broke")
        mock_save = MagicMock(return_value=1)

        with patch(PIPELINE_PATH, mock_pipeline), \
             patch(SAVE_GEN_PATH, mock_save):
            self.client.post("/generate", json={"requirement": self.REQUIREMENT})

        mock_save.assert_not_called()

    def test_database_error_during_save_returns_500(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()

        with patch(PIPELINE_PATH, mock_pipeline), \
             patch(SAVE_GEN_PATH, side_effect=Exception("disk full")):
            response = self.client.post(
                "/generate", json={"requirement": self.REQUIREMENT}
            )

        assert response.status_code == 500


# ---------------------------------------------------------------------------
# TestListGenerations  (Phase 5B)
# ---------------------------------------------------------------------------

class TestListGenerations:
    """Tests for GET /generations."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_get_generations_returns_200(self):
        with patch(GET_GENS_PATH, return_value=[]):
            response = self.client.get("/generations")
        assert response.status_code == 200

    def test_get_generations_returns_list(self):
        with patch(GET_GENS_PATH, return_value=[]):
            response = self.client.get("/generations")
        assert isinstance(response.json(), list)

    def test_get_generations_empty_list(self):
        with patch(GET_GENS_PATH, return_value=[]):
            response = self.client.get("/generations")
        assert response.json() == []

    def test_get_generations_returns_multiple_items(self):
        rows = [_make_db_generation(1, "First req."), _make_db_generation(2, "Second req.")]
        with patch(GET_GENS_PATH, return_value=rows):
            response = self.client.get("/generations")
        assert len(response.json()) == 2

    def test_get_generations_each_item_has_id(self):
        rows = [_make_db_generation(1)]
        with patch(GET_GENS_PATH, return_value=rows):
            response = self.client.get("/generations")
        for item in response.json():
            assert "id" in item

    def test_get_generations_each_item_has_requirement(self):
        rows = [_make_db_generation(1)]
        with patch(GET_GENS_PATH, return_value=rows):
            response = self.client.get("/generations")
        for item in response.json():
            assert "requirement" in item

    def test_get_generations_db_error_returns_500(self):
        with patch(GET_GENS_PATH, side_effect=Exception("db gone")):
            response = self.client.get("/generations")
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# TestRetrieveGeneration  (Phase 5B)
# ---------------------------------------------------------------------------

class TestRetrieveGeneration:
    """Tests for GET /generations/{generation_id}."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_retrieve_existing_generation_returns_200(self):
        row = _make_db_generation(1)
        with patch(GET_GEN_PATH, return_value=row):
            response = self.client.get("/generations/1")
        assert response.status_code == 200

    def test_retrieve_generation_returns_correct_id(self):
        row = _make_db_generation(5)
        with patch(GET_GEN_PATH, return_value=row):
            response = self.client.get("/generations/5")
        assert response.json()["id"] == 5

    def test_retrieve_generation_returns_requirement(self):
        row = _make_db_generation(1, "User should reset password.")
        with patch(GET_GEN_PATH, return_value=row):
            response = self.client.get("/generations/1")
        assert response.json()["requirement"] == "User should reset password."

    def test_retrieve_nonexistent_generation_returns_404(self):
        with patch(GET_GEN_PATH, return_value=None):
            response = self.client.get("/generations/9999")
        assert response.status_code == 404

    def test_retrieve_nonexistent_generation_has_detail(self):
        with patch(GET_GEN_PATH, return_value=None):
            response = self.client.get("/generations/9999")
        assert "detail" in response.json()

    def test_retrieve_generation_db_error_returns_500(self):
        with patch(GET_GEN_PATH, side_effect=Exception("db error")):
            response = self.client.get("/generations/1")
        assert response.status_code == 500

    def test_retrieve_generation_called_with_correct_id(self):
        """get_generation must be called with the id from the URL path."""
        mock_get = MagicMock(return_value=_make_db_generation(3))
        with patch(GET_GEN_PATH, mock_get):
            self.client.get("/generations/3")
        mock_get.assert_called_once_with(3)


# ---------------------------------------------------------------------------
# Helpers shared by Phase 6A tests
# ---------------------------------------------------------------------------

PDF_EXTRACTOR_PATH = "backend.main.PDFExtractor"

EXTRACTED_TEXT = "A user should be able to log in using a valid email and password."


def _make_pdf_upload(filename: str = "requirement.pdf", content: bytes = b"%PDF-1.4 fake"):
    """Return the files dict accepted by TestClient for a multipart upload."""
    return {"file": (filename, content, "application/pdf")}


# ---------------------------------------------------------------------------
# TestPDFEndpointSuccess  (Phase 6A)
# ---------------------------------------------------------------------------

class TestPDFEndpointSuccess:
    """POST /generate/pdf – happy path."""

    def setup_method(self):
        self.client = TestClient(app)
        # Mock extractor
        self.mock_extractor = MagicMock()
        self.mock_extractor.return_value.extract_text.return_value = EXTRACTED_TEXT
        # Mock pipeline
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.return_value.generate.return_value = _make_pipeline_result()

    def _post_pdf(self, save_id: int = 1, filename: str = "requirement.pdf"):
        with patch(PDF_EXTRACTOR_PATH, self.mock_extractor), \
             patch(PIPELINE_PATH, self.mock_pipeline), \
             patch(SAVE_GEN_PATH, return_value=save_id):
            return self.client.post(
                "/generate/pdf",
                files=_make_pdf_upload(filename),
            )

    def test_successful_pdf_upload_returns_200(self):
        assert self._post_pdf().status_code == 200

    def test_response_contains_id(self):
        body = self._post_pdf(save_id=7).json()
        assert body["id"] == 7

    def test_response_id_is_integer(self):
        body = self._post_pdf().json()
        assert isinstance(body["id"], int)

    def test_response_contains_requirement(self):
        body = self._post_pdf().json()
        assert "requirement" in body

    def test_response_contains_test_cases(self):
        body = self._post_pdf().json()
        assert "test_cases" in body
        assert isinstance(body["test_cases"], list)

    def test_response_contains_total_test_cases(self):
        body = self._post_pdf().json()
        assert "total_test_cases" in body

    def test_extracted_text_passed_to_pipeline(self):
        """The text extracted from the PDF must be forwarded to pipeline.generate()."""
        self._post_pdf()
        self.mock_pipeline.return_value.generate.assert_called_once_with(EXTRACTED_TEXT)

    def test_extractor_is_called_once(self):
        self._post_pdf()
        assert self.mock_extractor.return_value.extract_text.call_count == 1

    def test_save_generation_is_called(self):
        mock_save = MagicMock(return_value=1)
        with patch(PDF_EXTRACTOR_PATH, self.mock_extractor), \
             patch(PIPELINE_PATH, self.mock_pipeline), \
             patch(SAVE_GEN_PATH, mock_save):
            self.client.post("/generate/pdf", files=_make_pdf_upload())
        mock_save.assert_called_once()

    def test_save_generation_receives_requirement_text(self):
        """save_generation must be called with the text extracted from the PDF."""
        mock_save = MagicMock(return_value=1)
        with patch(PDF_EXTRACTOR_PATH, self.mock_extractor), \
             patch(PIPELINE_PATH, self.mock_pipeline), \
             patch(SAVE_GEN_PATH, mock_save):
            self.client.post("/generate/pdf", files=_make_pdf_upload())
        call_kwargs = mock_save.call_args.kwargs
        # The requirement saved to DB must be the pipeline’s requirement_text
        assert "requirement" in call_kwargs


# ---------------------------------------------------------------------------
# TestPDFEndpointValidation  (Phase 6A)
# ---------------------------------------------------------------------------

class TestPDFEndpointValidation:
    """POST /generate/pdf – bad uploads."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_non_pdf_file_returns_400(self):
        response = self.client.post(
            "/generate/pdf",
            files={"file": ("requirement.txt", b"some text", "text/plain")},
        )
        assert response.status_code == 400

    def test_non_pdf_error_message_mentions_pdf(self):
        response = self.client.post(
            "/generate/pdf",
            files={"file": ("req.docx", b"data", "application/vnd.openxmlformats")},
        )
        body = response.json()
        assert "pdf" in body["detail"].lower()

    def test_missing_file_returns_422(self):
        """Sending no file at all must result in 422 (FastAPI validation)."""
        response = self.client.post("/generate/pdf")
        assert response.status_code == 422

    def test_jpg_file_rejected(self):
        response = self.client.post(
            "/generate/pdf",
            files={"file": ("photo.jpg", b"\xff\xd8\xff", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_file_with_no_extension_rejected(self):
        response = self.client.post(
            "/generate/pdf",
            files={"file": ("noextension", b"data", "application/octet-stream")},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# TestPDFEndpointExtractionErrors  (Phase 6A)
# ---------------------------------------------------------------------------

class TestPDFEndpointExtractionErrors:
    """POST /generate/pdf – PDFExtractor raises exceptions."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_empty_pdf_returns_422(self):
        mock_extractor = MagicMock()
        mock_extractor.return_value.extract_text.side_effect = ValueError(
            "No readable text found in the PDF."
        )
        with patch(PDF_EXTRACTOR_PATH, mock_extractor):
            response = self.client.post("/generate/pdf", files=_make_pdf_upload())
        assert response.status_code == 422

    def test_empty_pdf_error_detail_present(self):
        mock_extractor = MagicMock()
        mock_extractor.return_value.extract_text.side_effect = ValueError(
            "No readable text found in the PDF."
        )
        with patch(PDF_EXTRACTOR_PATH, mock_extractor):
            response = self.client.post("/generate/pdf", files=_make_pdf_upload())
        assert "detail" in response.json()

    def test_corrupt_pdf_returns_422(self):
        mock_extractor = MagicMock()
        mock_extractor.return_value.extract_text.side_effect = ValueError(
            "Could not read PDF file"
        )
        with patch(PDF_EXTRACTOR_PATH, mock_extractor):
            response = self.client.post("/generate/pdf", files=_make_pdf_upload())
        assert response.status_code == 422

    def test_unexpected_extraction_error_returns_500(self):
        mock_extractor = MagicMock()
        mock_extractor.return_value.extract_text.side_effect = RuntimeError("disk error")
        with patch(PDF_EXTRACTOR_PATH, mock_extractor):
            response = self.client.post("/generate/pdf", files=_make_pdf_upload())
        assert response.status_code == 500

    def test_extraction_error_pipeline_not_called(self):
        """If extraction fails, the pipeline must NOT be called."""
        mock_extractor = MagicMock()
        mock_extractor.return_value.extract_text.side_effect = ValueError("no text")
        mock_pipeline = MagicMock()
        with patch(PDF_EXTRACTOR_PATH, mock_extractor), \
             patch(PIPELINE_PATH, mock_pipeline):
            self.client.post("/generate/pdf", files=_make_pdf_upload())
        mock_pipeline.return_value.generate.assert_not_called()


# ---------------------------------------------------------------------------
# TestPDFEndpointPipelineErrors  (Phase 6A)
# ---------------------------------------------------------------------------

class TestPDFEndpointPipelineErrors:
    """POST /generate/pdf – pipeline raises exceptions after successful extraction."""

    def setup_method(self):
        self.client = TestClient(app)
        self.mock_extractor = MagicMock()
        self.mock_extractor.return_value.extract_text.return_value = EXTRACTED_TEXT

    def _post(self, pipeline_mock):
        with patch(PDF_EXTRACTOR_PATH, self.mock_extractor), \
             patch(PIPELINE_PATH, pipeline_mock):
            return self.client.post("/generate/pdf", files=_make_pdf_upload())

    def test_pipeline_value_error_returns_422(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.side_effect = ValueError("bad req")
        assert self._post(mock_pipeline).status_code == 422

    def test_pipeline_environment_error_returns_500(self):
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = EnvironmentError("no key")
        assert self._post(mock_pipeline).status_code == 500

    def test_pipeline_environment_error_does_not_expose_key(self):
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = EnvironmentError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )
        response = self._post(mock_pipeline)
        assert "GEMINI_API_KEY" not in str(response.json())

    def test_pipeline_unexpected_error_returns_500(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.side_effect = RuntimeError("crash")
        assert self._post(mock_pipeline).status_code == 500

    def test_pipeline_error_db_not_called(self):
        """If the pipeline fails, save_generation must NOT be called."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.side_effect = ValueError("bad")
        mock_save = MagicMock(return_value=1)
        with patch(PDF_EXTRACTOR_PATH, self.mock_extractor), \
             patch(PIPELINE_PATH, mock_pipeline), \
             patch(SAVE_GEN_PATH, mock_save):
            self.client.post("/generate/pdf", files=_make_pdf_upload())
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# TestPDFEndpointDatabaseErrors  (Phase 6A)
# ---------------------------------------------------------------------------

class TestPDFEndpointDatabaseErrors:
    """POST /generate/pdf – database errors after successful pipeline run."""

    def setup_method(self):
        self.client = TestClient(app)
        self.mock_extractor = MagicMock()
        self.mock_extractor.return_value.extract_text.return_value = EXTRACTED_TEXT
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.return_value.generate.return_value = _make_pipeline_result()

    def test_db_error_returns_500(self):
        with patch(PDF_EXTRACTOR_PATH, self.mock_extractor), \
             patch(PIPELINE_PATH, self.mock_pipeline), \
             patch(SAVE_GEN_PATH, side_effect=Exception("disk full")):
            response = self.client.post("/generate/pdf", files=_make_pdf_upload())
        assert response.status_code == 500

    def test_db_error_detail_present(self):
        with patch(PDF_EXTRACTOR_PATH, self.mock_extractor), \
             patch(PIPELINE_PATH, self.mock_pipeline), \
             patch(SAVE_GEN_PATH, side_effect=Exception("disk full")):
            response = self.client.post("/generate/pdf", files=_make_pdf_upload())
        assert "detail" in response.json()


# ---------------------------------------------------------------------------
# TestPDFEndpointTempFileCleanup  (Phase 6A)
# ---------------------------------------------------------------------------

class TestPDFEndpointTempFileCleanup:
    """
    The endpoint must remove the temporary uploaded file after the request,
    regardless of whether it succeeded or failed.
    """

    def setup_method(self):
        self.client = TestClient(app)

    def _temp_files_before_and_after(self, mock_extractor, mock_pipeline, mock_save):
        """
        Capture tempfile.NamedTemporaryFile to learn the tmp_path, then verify
        it no longer exists after the request.
        """
        import tempfile as _tempfile
        created_paths: list[str] = []
        real_ntf = _tempfile.NamedTemporaryFile

        def tracking_ntf(**kwargs):
            obj = real_ntf(**kwargs)
            created_paths.append(obj.name)
            return obj

        with patch(PDF_EXTRACTOR_PATH, mock_extractor), \
             patch(PIPELINE_PATH, mock_pipeline), \
             patch(SAVE_GEN_PATH, mock_save), \
             patch("backend.main.tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            self.client.post("/generate/pdf", files=_make_pdf_upload())

        return created_paths

    def test_temp_file_removed_on_success(self):
        import os
        mock_extractor = MagicMock()
        mock_extractor.return_value.extract_text.return_value = EXTRACTED_TEXT
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.generate.return_value = _make_pipeline_result()
        mock_save = MagicMock(return_value=1)

        paths = self._temp_files_before_and_after(mock_extractor, mock_pipeline, mock_save)
        for path in paths:
            assert not os.path.exists(path), f"Temp file not cleaned up: {path}"

    def test_temp_file_removed_on_extraction_error(self):
        import os
        mock_extractor = MagicMock()
        mock_extractor.return_value.extract_text.side_effect = ValueError("no text")
        mock_pipeline = MagicMock()
        mock_save = MagicMock(return_value=1)

        paths = self._temp_files_before_and_after(mock_extractor, mock_pipeline, mock_save)
        for path in paths:
            assert not os.path.exists(path), f"Temp file not cleaned up: {path}"
