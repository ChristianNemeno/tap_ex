# PDF Q&A Extractor

A modular Streamlit application that extracts multiple-choice questions from PDFs using Google's Gemini AI and integrates with a backend API for quiz creation.

## Features

-  **PDF Upload**: Support for text-based and scanned PDFs (up to 200 pages)
-  **AI Extraction**: Extracts MCQs in backend-compatible CreateQuizDto format
-  **JSON Editor**: Review and edit extracted quiz data before submission
-  **Auto-Login**: Caches JWT tokens with automatic re-authentication
-  **Validation**: Ensures 2-6 choices per question with exactly one correct answer
-  **Progress Tracking**: Real-time progress for large PDF processing
-  **Export**: Download results as JSON or CSV
-  **Chunked Processing**: Splits large PDFs into 20-page chunks for optimal processing

## Architecture

The application follows a modular architecture with clear separation of concerns:

```
tap_ex/
├── core/           # Business logic (validation, config, logging)
├── backend/        # API integration (auth, quiz creation)
├── extraction/     # AI & PDF processing (Gemini, PDF utils)
├── ui/             # Streamlit components (sidebar, upload, results)
└── app.py          # Main orchestrator (84 lines)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Requirements

- Windows PowerShell (recommended)
- Python 3.8+ (installed via python.org or Windows py launcher)
- Google Gemini API key
- Backend API URL (optional, for quiz creation)

## Project Structure

### Core Modules
- **core/config.py**: Session state management
- **core/logging_utils.py**: Safe logging with API key fingerprinting
- **core/validation.py**: CreateQuizDto validation

### Backend Integration
- **backend/auth.py**: JWT authentication and token management
- **backend/quiz_api.py**: Quiz creation endpoint integration

### Extraction Layer
- **extraction/gemini.py**: Gemini AI integration for MCQ extraction
- **extraction/pdf_utils.py**: PDF manipulation and splitting

### UI Components
- **ui/components.py**: Common UI elements (header, footer, CSS)
- **ui/sidebar.py**: Configuration sidebar
- **ui/upload.py**: File upload and processing
- **ui/results.py**: Results display with JSON editor
- **ui/export.py**: Export functionality

### Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md): Architecture details and design patterns
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md): Refactoring summary
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md): Migration instructions
- [STRUCTURE.md](STRUCTURE.md): Project structure overview

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
python -m pip install "cryptography>=3.1"  # For encrypted PDF support
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

