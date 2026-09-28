# AI-Powered Test Case Generator

An AI-powered application that converts natural-language software requirements into structured and prioritized test cases.

## Features

- Process requirements from text and supported document formats
- Extract and analyze software requirements
- Identify test scenarios
- Generate structured test cases using AI
- FastAPI backend
- React-based frontend
- SQLite database
- Automated tests for core components

## Tech Stack

- Python
- FastAPI
- React.js
- SQLite
- Google Gemini API
- NumPy / Pandas

## Workflow

**Requirements → Requirement Processing → Scenario Identification → Test Case Generation → Structured Test Cases**

## Running Locally

### Backend

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload

Frontend
cd frontend
npm install
npm run dev

Configure the required Gemini API key in your environment before running the application.