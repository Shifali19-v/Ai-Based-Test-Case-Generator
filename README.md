# AI-Powered Test Case Generator

An AI-powered application that converts natural-language software requirements into structured and prioritized test cases.

## Overview

The system uses a multi-stage AI pipeline to process software requirements, identify relevant test scenarios, and generate structured test cases. Requirements can be provided as plain text or extracted from supported document formats.

## Features

- Natural-language requirement processing
- Requirement extraction from PDF, DOCX, and CSV files
- Automated scenario identification
- AI-powered test case generation
- Structured and prioritized test cases
- FastAPI backend
- React-based frontend
- SQLite database
- Automated test suite for core components

## Architecture

```text
Requirements
     ↓
Requirement Processing
     ↓
Scenario Identification
     ↓
Test Case Generation
     ↓
Structured Test Cases
     ↓
SQLite Database
     ↓
React Frontend

Tech Stack
Backend: Python, FastAPI
Frontend: React.js, Vite
AI: Google Gemini API
Database: SQLite
Testing: Pytest

Project Structure

ai-test-case-generator/
├── ai/              # AI pipeline and document extraction
├── backend/         # FastAPI backend
├── database/        # Database operations
├── frontend/        # React frontend
├── tests/           # Automated tests
├── data/            # Application data
└── requirements.txt

Getting Started
Backend
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload

Frontend
cd frontend
npm install
npm run dev

Configure your Gemini API key as an environment variable before running the application.

Workflow

Requirement → Analysis → Scenario Identification → Test Case Generation → Structured Output
