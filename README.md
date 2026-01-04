# PDF Q&A Extractor

A Streamlit app that extracts question-and-answer pairs from a PDF using the Gemini API, then displays results and lets you export them as CSV or JSON.

## Features

- Upload a PDF (text-based or scanned)
- Extract Q&A pairs using Gemini
- View results as a table, cards, or raw JSON
- Search within extracted questions and answers
- Export results to CSV and JSON

## Requirements

- Windows PowerShell (recommended)
- Python (installed via python.org or the Windows py launcher)
- A Gemini API key

## Project Files

- app.py: Streamlit UI and controller logic
- processor.py: Gemini API call and JSON parsing/normalization
- assets/styles.css: Optional UI styling
- .env: Local environment variables (not committed)
- requirements.txt: Python dependencies

## Setup

### 1) Create and activate a virtual environment

From the project folder:

```powershell
cd C:\Nemeno\tap_ex
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for your user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If you do not have requirements.txt installed yet, you can install directly:

```powershell
python -m pip install streamlit google-generativeai python-dotenv pandas
```

### 3) Configure your API key

Create a file named .env in the project root (same folder as app.py) and add:

```text
GEMINI_API_KEY=YOUR_KEY_HERE
```

Notes:
- Keep .env private and do not commit it.
- The app reads .env automatically via python-dotenv.

## Run the App

With the virtual environment activated:

```powershell
cd C:\Nemeno\tap_ex
python -m streamlit run app.py
```

Streamlit will print a local URL (typically http://localhost:8501). Open it in your browser.

## How to Use

1. Enter your API key in .env (recommended).
2. Start the app.
3. Upload a PDF.
4. Click Extract Q&A Pairs.
5. View results:
   - Table: best for scanning lots of Q&A
   - Cards: best for reading
   - JSON: best for debugging
6. Use the Export section to download CSV or JSON.

## Model Configuration

The processor uses a default model name in processor.py. If you change the model name, your key/project must have access to that model, otherwise the request will fail.


### Model response is not valid JSON

The app expects JSON and will error if the model returns extra text.
If this happens consistently:
- Try a smaller, simpler PDF
- Reduce temperature in the sidebar
- We can also add a retry pass in processor.py to enforce JSON-only output

### Permission/model errors

If you see errors like "model not found" or "permission denied":
- Switch back to a model your project can access
- Confirm model availability in Google AI Studio for your project

## Development Notes

- processor.py writes the uploaded PDF to a temporary file and deletes it after the request.
- Results are stored in Streamlit session state for the current session.

