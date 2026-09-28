"""
AI-Based Test Case Generator
Phase 6C: FastAPI Backend

Exposes the TestCasePipeline via a REST API and persists every generation
to a local SQLite database.

Endpoints:
    GET  /
    POST /generate
    POST /generate/pdf
    POST /generate/docx
    POST /generate/csv
    GET  /generations
    GET  /generations/{generation_id}
"""

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from ai.pipeline import TestCasePipeline, PipelineResult
from ai.pdf_extractor import PDFExtractor
from ai.docx_extractor import DOCXExtractor
from ai.csv_extractor import CSVExtractor

from database.db import (
    initialize_database,
    save_generation,
    get_generations,
    get_generation,
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialise the database when the application starts."""
    initialize_database()
    yield


app = FastAPI(
    title="AI-Based Test Case Generator API",
    description="Generates structured test cases from plain-English software requirements.",
    version="2.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    """Request body for POST /generate."""

    requirement: str

    @field_validator("requirement")
    @classmethod
    def requirement_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("requirement must not be empty.")
        return value


class TestCaseResponse(BaseModel):
    """JSON-serialisable representation of a single TestCase."""

    title: str
    description: str
    steps: list[str]
    expected_result: str
    priority: str


class GenerateResponse(BaseModel):
    """Response body for generation endpoints."""

    id: int
    requirement: str
    total_test_cases: int
    test_cases: list[TestCaseResponse]


@app.get("/", summary="Health check")
def root() -> dict:
    return {"message": "AI-Based Test Case Generator API is running."}


@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate test cases from a requirement",
)
def generate(request: GenerateRequest) -> GenerateResponse:

    try:
        pipeline = TestCasePipeline()
        result: PipelineResult = pipeline.generate(request.requirement)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except EnvironmentError as exc:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error. Please contact the administrator.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(exc)}",
        ) from exc

    processed_req_dict = result.processed_req.model_dump()
    scenarios_dict = result.scenarios.model_dump()
    test_cases_list = [tc.model_dump() for tc in result.test_cases]

    try:
        generation_id = save_generation(
            requirement=result.requirement_text,
            processed_requirement=processed_req_dict,
            scenarios=scenarios_dict,
            test_cases=test_cases_list,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(exc)}",
        ) from exc

    test_cases = [
        TestCaseResponse(
            title=tc.title,
            description=tc.description,
            steps=tc.steps,
            expected_result=tc.expected_result,
            priority=tc.priority,
        )
        for tc in result.test_cases
    ]

    return GenerateResponse(
        id=generation_id,
        requirement=result.requirement_text,
        total_test_cases=len(test_cases),
        test_cases=test_cases,
    )


@app.post(
    "/generate/pdf",
    response_model=GenerateResponse,
    summary="Generate test cases from an uploaded PDF requirement",
)
def generate_from_pdf(
    file: UploadFile = File(
        ...,
        description="A text-based PDF containing a software requirement",
    ),
) -> GenerateResponse:

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as tmp:

            tmp_path = tmp.name
            content = file.file.read()
            tmp.write(content)

        try:
            extractor = PDFExtractor()
            requirement_text = extractor.extract_text(tmp_path)

        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"PDF extraction error: {str(exc)}",
            ) from exc

        try:
            pipeline = TestCasePipeline()
            result: PipelineResult = pipeline.generate(requirement_text)

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except EnvironmentError as exc:
            raise HTTPException(
                status_code=500,
                detail="Server configuration error. Please contact the administrator.",
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {str(exc)}",
            ) from exc

        processed_req_dict = result.processed_req.model_dump()
        scenarios_dict = result.scenarios.model_dump()
        test_cases_list = [tc.model_dump() for tc in result.test_cases]

        try:
            generation_id = save_generation(
                requirement=result.requirement_text,
                processed_requirement=processed_req_dict,
                scenarios=scenarios_dict,
                test_cases=test_cases_list,
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(exc)}",
            ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    test_cases = [
        TestCaseResponse(
            title=tc.title,
            description=tc.description,
            steps=tc.steps,
            expected_result=tc.expected_result,
            priority=tc.priority,
        )
        for tc in result.test_cases
    ]

    return GenerateResponse(
        id=generation_id,
        requirement=result.requirement_text,
        total_test_cases=len(test_cases),
        test_cases=test_cases,
    )


@app.post(
    "/generate/docx",
    response_model=GenerateResponse,
    summary="Generate test cases from an uploaded DOCX requirement",
)
def generate_from_docx(
    file: UploadFile = File(
        ...,
        description="A DOCX containing a software requirement",
    ),
) -> GenerateResponse:

    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Only DOCX files are accepted. Please upload a .docx file.",
        )

    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".docx",
        ) as tmp:

            tmp_path = tmp.name
            content = file.file.read()
            tmp.write(content)

        try:
            extractor = DOCXExtractor()
            requirement_text = extractor.extract_text(tmp_path)

        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"DOCX extraction error: {str(exc)}",
            ) from exc

        try:
            pipeline = TestCasePipeline()
            result: PipelineResult = pipeline.generate(requirement_text)

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except EnvironmentError as exc:
            raise HTTPException(
                status_code=500,
                detail="Server configuration error. Please contact the administrator.",
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {str(exc)}",
            ) from exc

        processed_req_dict = result.processed_req.model_dump()
        scenarios_dict = result.scenarios.model_dump()
        test_cases_list = [tc.model_dump() for tc in result.test_cases]

        try:
            generation_id = save_generation(
                requirement=result.requirement_text,
                processed_requirement=processed_req_dict,
                scenarios=scenarios_dict,
                test_cases=test_cases_list,
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(exc)}",
            ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    test_cases = [
        TestCaseResponse(
            title=tc.title,
            description=tc.description,
            steps=tc.steps,
            expected_result=tc.expected_result,
            priority=tc.priority,
        )
        for tc in result.test_cases
    ]

    return GenerateResponse(
        id=generation_id,
        requirement=result.requirement_text,
        total_test_cases=len(test_cases),
        test_cases=test_cases,
    )


@app.post(
    "/generate/csv",
    response_model=GenerateResponse,
    summary="Generate test cases from an uploaded CSV requirement",
)
def generate_from_csv(
    file: UploadFile = File(
        ...,
        description="A CSV containing software requirements",
    ),
) -> GenerateResponse:

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted. Please upload a .csv file.",
        )

    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as tmp:

            tmp_path = tmp.name
            content = file.file.read()
            tmp.write(content)

        try:
            extractor = CSVExtractor()
            requirement_text = extractor.extract_text(tmp_path)

        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"CSV extraction error: {str(exc)}",
            ) from exc

        try:
            pipeline = TestCasePipeline()
            result: PipelineResult = pipeline.generate(requirement_text)

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except EnvironmentError as exc:
            raise HTTPException(
                status_code=500,
                detail="Server configuration error. Please contact the administrator.",
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {str(exc)}",
            ) from exc

        processed_req_dict = result.processed_req.model_dump()
        scenarios_dict = result.scenarios.model_dump()
        test_cases_list = [tc.model_dump() for tc in result.test_cases]

        try:
            generation_id = save_generation(
                requirement=result.requirement_text,
                processed_requirement=processed_req_dict,
                scenarios=scenarios_dict,
                test_cases=test_cases_list,
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(exc)}",
            ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    test_cases = [
        TestCaseResponse(
            title=tc.title,
            description=tc.description,
            steps=tc.steps,
            expected_result=tc.expected_result,
            priority=tc.priority,
        )
        for tc in result.test_cases
    ]

    return GenerateResponse(
        id=generation_id,
        requirement=result.requirement_text,
        total_test_cases=len(test_cases),
        test_cases=test_cases,
    )


@app.get(
    "/generations",
    summary="List all saved generations",
)
def list_generations() -> list[dict]:

    try:
        return get_generations()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(exc)}",
        ) from exc


@app.get(
    "/generations/{generation_id}",
    summary="Retrieve one saved generation by id",
)
def retrieve_generation(generation_id: int) -> dict:

    try:
        generation = get_generation(generation_id)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(exc)}",
        ) from exc

    if generation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Generation with id {generation_id} not found.",
        )

    return generation