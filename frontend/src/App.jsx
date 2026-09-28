import { useState } from "react";
import axios from "axios";

function App() {
  const [requirement, setRequirement] = useState("");
  const [file, setFile] = useState(null);
  const [testCases, setTestCases] = useState([]);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");

  const generateTestCases = async () => {
    if (!requirement.trim() && !file) {
      setError("Please enter a requirement or upload a file.");
      return;
    }

    setLoading(true);
    setError("");
    setTestCases([]);

    try {
      let response;

      if (file) {
        const formData = new FormData();
        formData.append("file", file);

        const fileType = file.name.split(".").pop().toLowerCase();

        let endpoint = "";

        if (fileType === "pdf") {
          endpoint = "http://127.0.0.1:8000/generate/pdf";
        } else if (fileType === "docx") {
          endpoint = "http://127.0.0.1:8000/generate/docx";
        } else if (fileType === "csv") {
          endpoint = "http://127.0.0.1:8000/generate/csv";
        } else {
          setError("Only PDF, DOCX, and CSV files are supported.");
          setLoading(false);
          return;
        }

        response = await axios.post(endpoint, formData);
      } else {
        response = await axios.post(
          "http://127.0.0.1:8000/generate",
          {
            requirement: requirement,
          }
        );
      }

      setTestCases(response.data.test_cases);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Something went wrong while generating test cases."
      );
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    setShowHistory(true);
    setHistoryLoading(true);
    setError("");

    try {
      const response = await axios.get(
        "http://127.0.0.1:8000/generations"
      );

      setHistory(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Unable to load history."
      );
    } finally {
      setHistoryLoading(false);
    }
  };

  const showGenerator = () => {
    setShowHistory(false);
    setError("");
  };

  const viewHistoryTestCases = (item) => {
    setTestCases(item.test_cases);
    setRequirement(item.requirement);
    setShowHistory(false);
    setError("");
  };

  return (
    <div className="min-h-screen bg-gray-100">

      {/* Navbar */}
      <nav className="bg-gray-900 px-8 py-4 text-white shadow">
        <div className="mx-auto flex max-w-6xl items-center justify-between">

          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-blue-600 px-3 py-2 text-xl">
              🤖
            </div>

            <div>
              <h1 className="text-lg font-bold">
                AI Test Case Generator
              </h1>

              <p className="text-xs text-gray-400">
                Intelligent Software Testing
              </p>
            </div>
          </div>

          <div className="flex gap-6 text-sm">

            <button
              onClick={showGenerator}
              className={
                !showHistory
                  ? "font-semibold text-blue-400"
                  : "text-gray-300 hover:text-white"
              }
            >
              Generate
            </button>

            <button
              onClick={loadHistory}
              className={
                showHistory
                  ? "font-semibold text-blue-400"
                  : "text-gray-300 hover:text-white"
              }
            >
              History
            </button>

          </div>

        </div>
      </nav>

      {/* Main Content */}
      <main className="mx-auto max-w-6xl px-6 py-10">

        {/* GENERATOR PAGE */}
        {!showHistory && (
          <>
            <div className="mb-8">
              <h2 className="text-3xl font-bold text-gray-900">
                Generate Test Cases
              </h2>

              <p className="mt-2 text-gray-600">
                Enter a software requirement or upload a requirement file.
              </p>
            </div>

            {/* Input Card */}
            <div className="rounded-xl bg-white p-6 shadow-md">

              <label className="mb-2 block font-semibold text-gray-800">
                Software Requirement
              </label>

              <textarea
                value={requirement}
                onChange={(e) => setRequirement(e.target.value)}
                placeholder="Example: The user should be able to reset their password using their registered email address."
                className="h-40 w-full rounded-lg border border-gray-300 p-3 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
              />

              {/* OR */}
              <div className="my-6 flex items-center">
                <div className="h-px flex-1 bg-gray-200"></div>

                <span className="px-4 text-sm text-gray-500">
                  OR
                </span>

                <div className="h-px flex-1 bg-gray-200"></div>
              </div>

              {/* File Upload */}
              <div>
                <label className="mb-3 block font-semibold text-gray-800">
                  Upload Requirement File
                </label>

                <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-gray-300 bg-gray-50 px-6 py-8 transition hover:border-blue-400 hover:bg-blue-50">

                  <div className="mb-3 text-4xl">
                    📁
                  </div>

                  <p className="font-semibold text-gray-700">
                    Click to upload a requirement file
                  </p>

                  <p className="mt-1 text-sm text-gray-500">
                    PDF, DOCX or CSV
                  </p>

                  <input
                    type="file"
                    accept=".pdf,.docx,.csv"
                    onChange={(e) => setFile(e.target.files[0])}
                    className="hidden"
                  />

                </label>

                {file && (
                  <div className="mt-3 flex items-center justify-between rounded-lg bg-blue-50 p-3">

                    <div>
                      <p className="text-sm font-semibold text-blue-800">
                        Selected File
                      </p>

                      <p className="text-sm text-blue-600">
                        {file.name}
                      </p>
                    </div>

                    <button
                      onClick={() => setFile(null)}
                      className="rounded-md px-3 py-1 text-sm font-semibold text-red-600 hover:bg-red-100"
                    >
                      Remove
                    </button>

                  </div>
                )}

                <div className="mt-4 grid grid-cols-3 gap-3">

                  <div className="rounded-lg border bg-gray-50 p-3 text-center">
                    <div className="text-2xl">📄</div>
                    <p className="mt-1 text-sm font-semibold">
                      PDF
                    </p>
                  </div>

                  <div className="rounded-lg border bg-gray-50 p-3 text-center">
                    <div className="text-2xl">📝</div>
                    <p className="mt-1 text-sm font-semibold">
                      DOCX
                    </p>
                  </div>

                  <div className="rounded-lg border bg-gray-50 p-3 text-center">
                    <div className="text-2xl">📊</div>
                    <p className="mt-1 text-sm font-semibold">
                      CSV
                    </p>
                  </div>

                </div>
              </div>

              {/* Generate Button */}
              <button
                onClick={generateTestCases}
                disabled={loading}
                className="mt-6 w-full rounded-lg bg-blue-600 px-7 py-3 font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Generating..." : "Generate Test Cases"}
              </button>

              {/* Error */}
              {error && (
                <p className="mt-4 rounded-lg bg-red-50 p-3 text-red-600">
                  {error}
                </p>
              )}

            </div>

            {/* Generated Results */}
            {testCases.length > 0 && (
              <div className="mt-10">

                <div className="mb-5 flex items-center justify-between">

                  <h2 className="text-2xl font-bold text-gray-900">
                    Generated Test Cases
                  </h2>

                  <span className="rounded-full bg-blue-100 px-4 py-1 text-sm font-semibold text-blue-700">
                    {testCases.length} Test Cases
                  </span>

                </div>

                <div className="space-y-5">

                  {testCases.map((testCase, index) => (
                    <div
                      key={index}
                      className="rounded-xl bg-white p-6 shadow-md transition hover:shadow-lg"
                    >

                      <div className="flex items-start justify-between">

                        <h3 className="text-xl font-bold text-gray-900">
                          {testCase.title}
                        </h3>

                        <span className="rounded-full bg-blue-100 px-3 py-1 text-xs font-semibold text-blue-700">
                          {testCase.priority}
                        </span>

                      </div>

                      <p className="mt-4 text-gray-700">
                        <strong>Description:</strong>{" "}
                        {testCase.description}
                      </p>

                      <div className="mt-4">

                        <strong>Steps:</strong>

                        <ol className="ml-6 mt-2 list-decimal space-y-1 text-gray-700">

                          {testCase.steps.map(
                            (step, stepIndex) => (
                              <li key={stepIndex}>
                                {step}
                              </li>
                            )
                          )}

                        </ol>

                      </div>

                      <div className="mt-5 rounded-lg bg-gray-50 p-4">

                        <strong>Expected Result:</strong>

                        <p className="mt-1 text-gray-700">
                          {testCase.expected_result}
                        </p>

                      </div>

                    </div>
                  ))}

                </div>

              </div>
            )}

          </>
        )}

        {/* HISTORY PAGE */}
        {showHistory && (
          <>
            <div className="mb-8">

              <h2 className="text-3xl font-bold text-gray-900">
                Generation History
              </h2>

              <p className="mt-2 text-gray-600">
                View your previously generated test cases.
              </p>

            </div>

            {/* Loading */}
            {historyLoading && (
              <div className="rounded-xl bg-white p-8 text-center shadow-md">

                <p className="text-gray-600">
                  Loading history...
                </p>

              </div>
            )}

            {/* Empty History */}
            {!historyLoading && history.length === 0 && (
              <div className="rounded-xl bg-white p-8 text-center shadow-md">

                <div className="text-5xl">
                  📋
                </div>

                <h3 className="mt-4 text-xl font-bold text-gray-800">
                  No History Found
                </h3>

                <p className="mt-2 text-gray-500">
                  Generate some test cases and they will appear here.
                </p>

              </div>
            )}

            {/* History List */}
            {!historyLoading && history.length > 0 && (
              <div className="space-y-4">

                {history.map((item) => (
                  <div
                    key={item.id}
                    className="rounded-xl bg-white p-6 shadow-md"
                  >

                    <div className="flex items-start justify-between">

                      <div>

                        <h3 className="text-lg font-bold text-gray-900">
                          Generation #{item.id}
                        </h3>

                        <p className="mt-2 text-gray-700">
                          {item.requirement}
                        </p>

                        <p className="mt-3 text-sm text-gray-500">
                          {new Date(
                            item.created_at
                          ).toLocaleString()}
                        </p>

                      </div>

                      <span className="rounded-full bg-blue-100 px-3 py-1 text-xs font-semibold text-blue-700">
                        {item.test_cases.length} Test Cases
                      </span>

                    </div>

                    <button
                      onClick={() =>
                        viewHistoryTestCases(item)
                      }
                      className="mt-4 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700"
                    >
                      View Test Cases
                    </button>

                  </div>
                ))}

              </div>
            )}

            {/* History Error */}
            {error && (
              <p className="mt-4 rounded-lg bg-red-50 p-3 text-red-600">
                {error}
              </p>
            )}

          </>
        )}

      </main>

    </div>
  );
}

export default App;