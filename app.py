"""
PDF Q&A Extractor - AI-Powered Document Processing
Extract questions and answers from PDF documents using Google's Gemini 1.5 Flash
"""

import streamlit as st
from dotenv import load_dotenv
import os
import json
import pandas as pd
from datetime import datetime
import tempfile
import time
from typing import Any, Dict, Optional
import logging
import traceback

import requests

from processor import GeminiConfig, extract_create_quiz_dto_from_pdf_bytes

# Load environment variables
load_dotenv()


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("tap_ex")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), "tap_ex.log")

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Also emit to console (shows up in Streamlit logs)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


LOGGER = _get_logger()


def _safe_key_fingerprint(key: str) -> str:
    if not isinstance(key, str) or not key:
        return "<empty>"
    tail = key[-4:] if len(key) >= 4 else key
    return f"len={len(key)} tail=***{tail}"


def _looks_like_auth_error(message: str) -> bool:
    m = (message or "").lower()
    return any(
        s in m
        for s in [
            "api key",
            "invalid api key",
            "invalid key",
            "unauthorized",
            "permission denied",
            "forbidden",
            "401",
            "403",
        ]
    )

# Page Configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="PDF Q&A Extractor",
    page_icon="⬜",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ChristianNemeno/tap_ex.git',
        'Report a bug': 'https://github.com/ChristianNemeno/tap_ex/issues',
        'About': """
        # PDF Q&A Extractor
        
        Extract questions and answers from PDF documents using Google's Gemini AI.
        
        **Version:** 1.0.0  
        **Powered by:** Gemini 1.5 Flash
        """
    }
)


def load_css():
    """Load custom CSS styling"""
    try:
        with open('assets/styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # CSS file is optional, continue without it
        pass


def initialize_session_state():
    """Initialize all session state variables with default values"""
    
    defaults = {
        # Results storage
        'results': None,
        'quiz_json_editor': None,
        'quiz_json_validated': None,
        'quiz_json_validation_errors': None,
        'uploaded_file_name': None,
        'processing_complete': False,

        # Backend API settings
        'backend_base_url': os.getenv('TAPCET_API_BASE_URL', 'https://localhost:7237'),
        'backend_email': os.getenv('TAPCET_API_EMAIL', ''),
        'backend_password': os.getenv('TAPCET_API_PASSWORD', ''),
        'backend_verify_tls': os.getenv('TAPCET_API_VERIFY_TLS', 'true').strip().lower() not in {'0', 'false', 'no'},

        # Backend auth cache
        'backend_token': None,
        'backend_token_expires_at': None,
        'backend_user_email': None,
        'backend_user_name': None,
        'backend_auth_error': None,
        'backend_last_login_attempt': 0.0,
        'backend_auto_login': True,
        
        # API configuration
        'api_key_valid': False,
        'api_key': os.getenv('GEMINI_API_KEY', ''),
        
        # Model settings
        'temperature': 0.2,
        'max_tokens': 8192,
        
        # Extraction options
        'include_page_numbers': True,
        'include_confidence': True,
        'extract_metadata': False,
        
        # UI state
        'show_examples': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _normalize_base_url(url: str) -> str:
    s = (url or '').strip()
    while s.endswith('/'):
        s = s[:-1]
    return s


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # Handle common Zulu suffix.
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _clamp_text(value: Any, *, min_len: int, max_len: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    if len(s) < min_len:
        return None
    return s


def _normalize_choices(choices: Any, errors: list[str], *, q_index: int) -> Optional[list[dict]]:
    if not isinstance(choices, list):
        errors.append(f"Question {q_index}: choices must be a list")
        return None

    items: list[dict] = []
    for c in choices:
        if not isinstance(c, dict):
            continue
        text = _clamp_text(c.get('text'), min_len=1, max_len=500)
        if not text:
            continue
        items.append({'text': text, 'isCorrect': bool(c.get('isCorrect'))})

    if len(items) < 2:
        errors.append(f"Question {q_index}: must have at least 2 choices")
        return None
    if len(items) > 6:
        # Trim extra choices to satisfy backend rule
        items = items[:6]

    correct_indexes = [i for i, c in enumerate(items) if c.get('isCorrect') is True]
    if len(correct_indexes) == 1:
        return items
    if len(correct_indexes) == 0:
        # Default: mark first option correct (user can fix in editor)
        items[0]['isCorrect'] = True
        errors.append(f"Question {q_index}: no correct choice marked; defaulted first choice to correct")
        return items

    keep = correct_indexes[0]
    for i in correct_indexes[1:]:
        items[i]['isCorrect'] = False
    items[keep]['isCorrect'] = True
    errors.append(f"Question {q_index}: multiple correct choices; kept first as correct")
    return items


def validate_and_normalize_create_quiz_dto(payload: Any) -> tuple[Optional[dict], list[str]]:
    """Validate+normalize a CreateQuizDto-like dict before POSTing to the backend.

    Returns (normalized_payload, errors). If normalized_payload is None, it's not safe to POST.
    """

    errors: list[str] = []
    if not isinstance(payload, dict):
        return None, ["Payload must be a JSON object"]

    title = _clamp_text(payload.get('title'), min_len=3, max_len=200)
    if not title:
        errors.append("title is required (3-200 chars)")

    description_raw = payload.get('description', None)
    description: Optional[str]
    if description_raw is None:
        description = None
    else:
        description = _clamp_text(description_raw, min_len=0, max_len=2000)

    questions_raw = payload.get('questions')
    if not isinstance(questions_raw, list) or len(questions_raw) < 1:
        errors.append("questions is required (at least 1 question)")
        questions_raw = []

    normalized_questions: list[dict] = []
    for i, q in enumerate(questions_raw, start=1):
        if not isinstance(q, dict):
            errors.append(f"Question {i}: must be an object")
            continue

        q_text = _clamp_text(q.get('text'), min_len=5, max_len=100)
        if not q_text:
            errors.append(f"Question {i}: text is required (5-100 chars)")
            continue

        explanation_raw = q.get('explanation', None)
        explanation: Optional[str]
        if explanation_raw is None:
            explanation = None
        else:
            explanation = _clamp_text(explanation_raw, min_len=0, max_len=300)

        image_url = q.get('imageUrl', None)
        if isinstance(image_url, str):
            image_url = image_url.strip()
            if not (image_url.startswith('http://') or image_url.startswith('https://')):
                errors.append(f"Question {i}: imageUrl must be a valid URL")
                image_url = None
        else:
            image_url = None

        choices = _normalize_choices(q.get('choices'), errors, q_index=i)
        if choices is None:
            continue

        out_q = {
            'text': q_text,
            'explanation': explanation,
            'imageUrl': image_url,
            'choices': choices,
        }
        # drop None fields
        out_q = {k: v for k, v in out_q.items() if v is not None}
        normalized_questions.append(out_q)

    if not normalized_questions:
        errors.append("No valid questions found")

    if errors and (not title or not normalized_questions):
        return None, errors

    normalized = {
        'title': title,
        'description': description,
        'questions': normalized_questions,
    }
    normalized = {k: v for k, v in normalized.items() if v is not None}
    return normalized, errors


def _backend_token_is_valid() -> bool:
    token = st.session_state.get('backend_token')
    expires_at = st.session_state.get('backend_token_expires_at')
    if not token or not isinstance(token, str):
        return False
    if not isinstance(expires_at, datetime):
        return True
    if expires_at.tzinfo is None:
        return datetime.now() < expires_at
    return datetime.now(expires_at.tzinfo) < expires_at


def backend_login(*, timeout_s: int = 20) -> bool:
    """Login to backend and cache JWT token in session state."""

    base_url = _normalize_base_url(st.session_state.get('backend_base_url', ''))
    email = (st.session_state.get('backend_email') or '').strip()
    password = st.session_state.get('backend_password') or ''

    st.session_state.backend_auth_error = None

    if not base_url:
        st.session_state.backend_auth_error = 'Missing backend base URL'
        return False
    if not email or not password:
        st.session_state.backend_auth_error = 'Missing backend email/password'
        return False

    # Simple throttle to avoid hammering during reruns.
    now_ts = time.time()
    last_attempt = float(st.session_state.get('backend_last_login_attempt') or 0.0)
    if now_ts - last_attempt < 3.0:
        return False
    st.session_state.backend_last_login_attempt = now_ts

    url = f"{base_url}/api/auth/login"
    try:
        resp = requests.post(
            url,
            json={"email": email, "password": password},
            timeout=timeout_s,
            verify=bool(st.session_state.get('backend_verify_tls', True)),
        )
    except requests.RequestException as e:
        st.session_state.backend_auth_error = f"Login request failed: {e}"
        return False

    if resp.status_code != 200:
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        msg = None
        if isinstance(payload, dict):
            msg = payload.get('message')
        st.session_state.backend_auth_error = msg or f"Login failed (HTTP {resp.status_code})"
        return False

    try:
        data = resp.json()
    except ValueError:
        st.session_state.backend_auth_error = 'Login response was not JSON'
        return False

    token = data.get('token')
    if not isinstance(token, str) or not token.strip():
        st.session_state.backend_auth_error = 'Login response missing token'
        return False

    st.session_state.backend_token = token.strip()
    st.session_state.backend_token_expires_at = _parse_iso_datetime(data.get('expiresAt'))
    st.session_state.backend_user_email = data.get('email') if isinstance(data.get('email'), str) else None
    st.session_state.backend_user_name = data.get('userName') if isinstance(data.get('userName'), str) else None
    st.session_state.backend_auth_error = None
    return True


def ensure_backend_token() -> bool:
    """Ensure we have a valid backend token, optionally auto-login."""
    if _backend_token_is_valid():
        return True
    if not st.session_state.get('backend_auto_login', True):
        return False
    return backend_login()


def reset_session():
    """Reset session state to defaults (preserves API key and settings)"""
    keys_to_keep = ['api_key', 'temperature', 'max_tokens', 'include_page_numbers', 
                    'include_confidence', 'extract_metadata']
    
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    
    initialize_session_state()


def clear_results():
    """Clear only the results data"""
    st.session_state.results = None
    st.session_state.processing_complete = False
    st.session_state.uploaded_file_name = None


# Initialize session state
initialize_session_state()

# Load custom CSS
load_css()


def render_sidebar():
    """Render the sidebar configuration panel"""
    
    with st.sidebar:
        # Sidebar Header
        st.title("Configuration")
        st.markdown("---")
        
        # API Key Section
        st.subheader("API Credentials")
        
        # Check if API key is loaded from .env
        env_api_key = os.getenv('GEMINI_API_KEY', '')
        if env_api_key and env_api_key.startswith("AIza"):
            st.session_state.api_key = env_api_key
            st.session_state.api_key_valid = True
            st.success("API key loaded from .env")
            
            # Show option to override
            with st.expander("Override API Key (Optional)"):
                api_key_input = st.text_input(
                    "Enter different API Key",
                    type="password",
                    placeholder="AIzaSy...",
                    help="Leave blank to use .env key"
                )
                
                if api_key_input:
                    if api_key_input.startswith("AIza"):
                        st.session_state.api_key = api_key_input
                        st.success("Using override key")
                    else:
                        st.error("Invalid API key format")
        else:
            # No .env key, require manual input
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.api_key,
                help="Enter your Gemini API key. Get one from https://makersuite.google.com/app/apikey",
                placeholder="AIzaSy..."
            )
            
            # Validate API key
            if api_key_input:
                if api_key_input.startswith("AIza"):
                    st.success("API key format looks valid")
                    st.session_state.api_key = api_key_input
                    st.session_state.api_key_valid = True
                else:
                    st.error("Invalid API key format")
                    st.session_state.api_key_valid = False
            else:
                st.warning("API key required (add to .env or enter above)")
                st.session_state.api_key_valid = False
        
        st.markdown("---")
        
        # Model Configuration
        st.subheader("Model Settings")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Lower values = more focused/deterministic. Higher values = more creative/diverse."
        )
        
        max_tokens = st.select_slider(
            "Max Output Tokens",
            options=[2048, 4096, 8192],
            value=st.session_state.max_tokens,
            help="Maximum number of tokens in the response"
        )
        
        st.session_state.temperature = temperature
        st.session_state.max_tokens = max_tokens
        
        st.markdown("---")
        
        # Extraction Settings
        st.subheader("Extraction Options")
        
        include_page_numbers = st.checkbox(
            "Include Page Numbers",
            value=st.session_state.include_page_numbers,
            help="Show the page number where each Q&A was found"
        )
        
        include_confidence = st.checkbox(
            "Show Confidence Scores",
            value=st.session_state.include_confidence,
            help="Display AI's confidence level for each extraction"
        )
        
        extract_metadata = st.checkbox(
            "Extract Metadata",
            value=st.session_state.extract_metadata,
            help="Include additional context like question type, difficulty, etc."
        )
        
        st.session_state.include_page_numbers = include_page_numbers
        st.session_state.include_confidence = include_confidence
        st.session_state.extract_metadata = extract_metadata

        st.markdown("---")

        # Backend Configuration
        st.subheader("Backend API")

        st.session_state.backend_base_url = st.text_input(
            "Base URL",
            value=st.session_state.backend_base_url,
            help="Example: https://localhost:7237",
        )

        st.session_state.backend_verify_tls = st.checkbox(
            "Verify TLS",
            value=bool(st.session_state.get('backend_verify_tls', True)),
            help="Disable only for local dev if Python cannot validate the HTTPS dev certificate.",
        )

        # Normalize after input
        st.session_state.backend_base_url = _normalize_base_url(st.session_state.backend_base_url)
        st.session_state.backend_email = st.text_input(
            "Email",
            value=st.session_state.backend_email,
            placeholder="admin@tapcet.com",
        )
        st.session_state.backend_password = st.text_input(
            "Password",
            value=st.session_state.backend_password,
            type="password",
        )

        st.session_state.backend_auto_login = st.checkbox(
            "Auto-login",
            value=bool(st.session_state.backend_auto_login),
            help="Automatically log in and refresh token when needed",
        )

        colA, colB = st.columns(2)
        with colA:
            if st.button("Login", use_container_width=True, type="secondary"):
                ok = backend_login()
                if ok:
                    st.success("Logged in")
                else:
                    st.error(st.session_state.get('backend_auth_error') or 'Login failed')
        with colB:
            if st.button("Logout", use_container_width=True, type="secondary"):
                st.session_state.backend_token = None
                st.session_state.backend_token_expires_at = None
                st.session_state.backend_user_email = None
                st.session_state.backend_user_name = None
                st.session_state.backend_auth_error = None

        # Attempt auto-login if enabled and creds exist
        _ = ensure_backend_token()
        if _backend_token_is_valid():
            st.success("Backend authorized")
        else:
            err = st.session_state.get('backend_auth_error')
            if err:
                st.warning(err)
            else:
                st.info("Not authorized")
        
        st.markdown("---")
        
        # Action Buttons
        st.subheader("Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Reset", use_container_width=True, type="secondary"):
                reset_session()
                st.rerun()
        
        with col2:
            if st.button("Clear", use_container_width=True, type="secondary"):
                clear_results()
                st.rerun()
        
        st.markdown("---")
        
        # Help Section
        with st.expander("Help & Tips"):
            st.markdown("""
            **Getting Started:**
            1. Enter your Gemini API key above
            2. Upload a PDF document
            3. Click 'Extract Q&A' button
            
            **Best Practices:**
            - Use high-quality scans (300+ DPI)
            - Smaller files process faster
            - Review confidence scores
            
            **Supported Formats:**
            - Text-based PDFs ✓
            - Scanned/Image PDFs ✓
            - Multi-page documents ✓
            
            **Need Help?**
            [Documentation](https://github.com/ChristianNemeno/tap_ex) | [GitHub Issues](https://github.com/ChristianNemeno/tap_ex/issues)
            """)
        
        # Version Info
        st.markdown("---")
        st.caption("v1.0.0 | Powered by Gemini 1.5 Flash")


def render_header():
    """Render the application header"""
    st.title("PDF Q&A Extractor")
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0; color: #666;'>
        Extract questions and answers from any PDF document using Google's Gemini 1.5 Flash AI
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats if results exist
    results = st.session_state.results
    if isinstance(results, dict) and isinstance(results.get('questions'), list):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Questions", len(results.get('questions') or []))
        with col2:
            st.metric("Title", results.get('title') or 'N/A')
        with col3:
            st.metric("File", st.session_state.uploaded_file_name or "N/A")
    
    st.divider()


def process_document(uploaded_file):
    """Process the uploaded document with Gemini API"""

    with st.spinner("Analyzing your document... This may take 30-120 seconds."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            st.session_state.processing_complete = False
            st.session_state.results = None

            status_text.text("Step 1/4: Reading file...")
            progress_bar.progress(15)
            pdf_bytes = uploaded_file.getvalue()

            status_text.text("Step 2/4: Preparing request...")
            progress_bar.progress(35)

            cfg = GeminiConfig(
                api_key=st.session_state.api_key,
                temperature=st.session_state.temperature,
                max_output_tokens=st.session_state.max_tokens,
            )

            status_text.text("Step 3/4: Extracting MCQs with Gemini...")
            progress_bar.progress(65)

            results = extract_create_quiz_dto_from_pdf_bytes(
                pdf_bytes=pdf_bytes,
                filename=uploaded_file.name,
                cfg=cfg,
            )

            status_text.text("Step 4/4: Finalizing...")
            progress_bar.progress(95)

            st.session_state.results = results
            # Seed JSON editor with extracted payload
            st.session_state.quiz_json_editor = json.dumps(results, indent=2)
            st.session_state.quiz_json_validated = None
            st.session_state.quiz_json_validation_errors = None
            st.session_state.processing_complete = True
            progress_bar.progress(100)
            q_count = len(results.get('questions') or []) if isinstance(results, dict) else 0
            st.success(f"Successfully extracted {q_count} questions.")

        except Exception as e:
            st.session_state.processing_complete = False
            st.session_state.results = None
            st.session_state.quiz_json_validated = None
            st.session_state.quiz_json_validation_errors = None

            err_text = str(e) or e.__class__.__name__
            LOGGER.error(
                "Gemini extraction failed | file=%s | model=%s | temp=%.1f | max_tokens=%s | api_key=%s",
                getattr(uploaded_file, 'name', 'unknown'),
                getattr(cfg, 'model_name', 'unknown') if 'cfg' in locals() else 'unknown',
                float(st.session_state.get('temperature', 0.0) or 0.0),
                st.session_state.get('max_tokens'),
                _safe_key_fingerprint(st.session_state.get('api_key', '')),
            )
            LOGGER.error("Exception: %s", err_text)
            LOGGER.error(traceback.format_exc())

            # If it looks like an auth error, invalidate the key so UI prompts clearly.
            if _looks_like_auth_error(err_text):
                st.session_state.api_key_valid = False
                st.error("Gemini request failed (API key/auth issue).")
                st.info("Re-check GEMINI_API_KEY in .env or paste a valid key in the sidebar.")
                st.caption("Details were written to tap_ex.log")
            else:
                st.error(f"An error occurred: {err_text}")
                st.info("Details were written to tap_ex.log")

        finally:
            progress_bar.empty()
            status_text.empty()


def render_upload_section():
    """Render the file upload section"""
    
    st.subheader("Upload Document")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Maximum file size: 20MB. Supports both text-based and scanned PDFs.",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # Store filename
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # File information
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Convert to MB
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info(f"**File:** {uploaded_file.name}")
        
        with col2:
            st.info(f"**Size:** {file_size:.2f} MB")
        
        with col3:
            file_type = "Text PDF" if file_size < 5 else "Large PDF"
            st.info(f"**Type:** {file_type}")
        
        # Warning for large files
        if file_size > 20:
            st.error("File size exceeds 20MB. Please upload a smaller file or contact support for large file processing.")
            return None
        elif file_size > 10:
            st.warning("Large file detected. Processing may take longer (2-5 minutes).")
        
        # File preview option
        with st.expander("Preview File Info"):
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {file_size:.2f} MB ({len(uploaded_file.getvalue())} bytes)")
            st.write(f"**Type:** {uploaded_file.type}")
            st.write(f"**Uploaded:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("")  # Spacing
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if not st.session_state.api_key_valid:
                st.error("Please enter a valid API key in the sidebar")
                st.button("Extract Quiz", disabled=True, use_container_width=True, type="primary")
            else:
                if st.button("Extract Quiz", use_container_width=True, type="primary"):
                    process_document(uploaded_file)
        
        return uploaded_file
    
    else:
        # Empty state with helpful message
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px; border: 2px dashed #ccc;'>
            <h3 style='color: #666;'>No file selected</h3>
            <p style='color: #999;'>
                Drag and drop a PDF file here, or click to browse<br>
                <small>Supports both text-based and scanned PDFs</small>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return None


def render_footer():
    """Render the application footer"""
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #999; padding: 2rem 0;'>
        <p>Made with Streamlit and Google Gemini AI</p>
        <p style='font-size: 12px;'>
            <a href='https://github.com/ChristianNemeno/tap_ex' target='_blank'>GitHub</a> | 
            <a href='https://github.com/ChristianNemeno/tap_ex/blob/main/PROJECT_PLAN.md' target='_blank'>Documentation</a> | 
            <a href='https://github.com/ChristianNemeno/tap_ex/issues' target='_blank'>Report Issue</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_results_section():
    """Render extracted quiz results (CreateQuizDto)."""

    results = st.session_state.get('results')
    if not isinstance(results, dict) or not isinstance(results.get('questions'), list):
        return

    st.markdown("---")
    st.subheader("Extraction Results")

    st.write(f"Title: {results.get('title') or 'N/A'}")
    if results.get('description'):
        st.write(f"Description: {results.get('description')}")

    questions = results.get('questions') or []
    st.metric("Questions", len(questions))

    search_query = st.text_input(
        "Search",
        placeholder="Search in questions",
        label_visibility="collapsed",
    )

    filtered = questions
    if search_query:
        q = search_query.strip().lower()
        if q:
            filtered = [qq for qq in questions if q in str(qq.get('text', '')).lower()]
            st.info(f"Matches: {len(filtered)}")

    for idx, q in enumerate(filtered, start=1):
        q_text = str(q.get('text', '')).strip()
        header = f"Q{idx}: {q_text[:90]}" if q_text else f"Q{idx}"
        with st.expander(header, expanded=(idx == 1)):
            st.markdown("**Question**")
            st.write(q_text)

            expl = q.get('explanation')
            if expl:
                st.markdown("**Explanation**")
                st.write(expl)

            st.markdown("**Choices**")
            for choice in (q.get('choices') or []):
                c_text = str(choice.get('text', '')).strip()
                suffix = " (correct)" if choice.get('isCorrect') else ""
                st.write(f"- {c_text}{suffix}")

    st.markdown("---")
    st.subheader("Edit & Validate JSON")

    if st.session_state.get('quiz_json_editor') is None:
        st.session_state.quiz_json_editor = json.dumps(results, indent=2)

    st.caption("Edit the CreateQuizDto JSON. Validate before creating the quiz.")
    edited_text = st.text_area(
        "CreateQuizDto JSON",
        value=st.session_state.quiz_json_editor,
        height=280,
        key="quiz_json_editor",
    )

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Validate JSON", type="secondary", use_container_width=True):
            try:
                obj = json.loads(edited_text or "")
            except json.JSONDecodeError as e:
                st.session_state.quiz_json_validated = None
                st.session_state.quiz_json_validation_errors = [f"Invalid JSON: {e}"]
            else:
                normalized, errs = validate_and_normalize_create_quiz_dto(obj)
                st.session_state.quiz_json_validated = normalized
                st.session_state.quiz_json_validation_errors = errs
                if normalized is not None:
                    # Reformat the editor with normalized JSON for convenience
                    st.session_state.quiz_json_editor = json.dumps(normalized, indent=2)

    with colB:
        if st.button("Reset to Extracted", type="secondary", use_container_width=True):
            st.session_state.quiz_json_editor = json.dumps(results, indent=2)
            st.session_state.quiz_json_validated = None
            st.session_state.quiz_json_validation_errors = None

    with colC:
        st.download_button(
            label="Download Edited JSON",
            data=edited_text or "",
            file_name=f"create_quiz_edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            type="secondary",
        )

    errs = st.session_state.get('quiz_json_validation_errors')
    if errs:
        # These are mostly warnings unless the payload is invalid (validated None)
        if st.session_state.get('quiz_json_validated') is None:
            st.error("Validation failed. Fix the issues below.")
        else:
            st.warning("Validated with warnings:")
        for msg in errs:
            st.write(f"- {msg}")
    elif st.session_state.get('quiz_json_validated') is not None:
        st.success("JSON validated and ready to send")

    st.markdown("---")
    st.subheader("Backend")

    if _backend_token_is_valid():
        st.success("Ready to create quiz in backend")
    else:
        st.info("Login in the sidebar to create quiz")

    if st.button("Create Quiz in Backend", type="primary", use_container_width=True):
        if not ensure_backend_token():
            st.error(st.session_state.get('backend_auth_error') or 'Not authorized')
        else:
            # Always parse+validate the current editor content before sending.
            current_text = st.session_state.get('quiz_json_editor') or ''
            try:
                obj = json.loads(current_text)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return

            payload, errs = validate_and_normalize_create_quiz_dto(obj)
            st.session_state.quiz_json_validated = payload
            st.session_state.quiz_json_validation_errors = errs
            if payload is None:
                st.error("Fix JSON validation errors before creating the quiz.")
                return

            base_url = _normalize_base_url(st.session_state.get('backend_base_url', ''))
            url = f"{base_url}/api/quiz"
            token = st.session_state.get('backend_token')
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                    verify=bool(st.session_state.get('backend_verify_tls', True)),
                )
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                return

            if resp.status_code == 201:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = None
                st.success("Quiz created")
                if payload is not None:
                    st.session_state.created_quiz_response = payload
                    st.json(payload)
            elif resp.status_code == 401:
                st.session_state.backend_token = None
                st.error("Unauthorized (token expired or invalid)")
            else:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = resp.text
                st.error(f"Create quiz failed (HTTP {resp.status_code})")
                st.write(payload)


def render_export_section():
    """Render export buttons for extracted quiz data"""

    results = st.session_state.get('results')
    if not isinstance(results, dict) or not isinstance(results.get('questions'), list):
        return

    st.markdown("---")
    st.subheader("Export")

    # Flatten questions for CSV
    rows = []
    for idx, q in enumerate(results.get('questions') or [], start=1):
        choices = q.get('choices') or []
        correct_text = next((c.get('text') for c in choices if c.get('isCorrect')), None)
        rows.append(
            {
                "question_number": idx,
                "question": q.get('text'),
                "correct_choice": correct_text,
                "choices": "; ".join([str(c.get('text', '')).strip() for c in choices]),
            }
        )

    df = pd.DataFrame(rows)
    csv_data = df.to_csv(index=False).encode('utf-8')
    json_data = json.dumps(results, indent=2)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"quiz_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
            type="secondary",
        )
    with col2:
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"quiz_{ts}.json",
            mime="application/json",
            use_container_width=True,
            type="secondary",
        )


def main():
    """Main application entry point"""
    
    # Render sidebar
    render_sidebar()
    
    # Render header
    render_header()
    
    # Render upload section
    uploaded_file = render_upload_section()

    # Render results and export
    if st.session_state.get('processing_complete') and st.session_state.get('results'):
        render_results_section()
        render_export_section()
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
