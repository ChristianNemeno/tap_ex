"""Application configuration and session state management."""

import os
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_session_state_defaults() -> Dict[str, Any]:
    """Return default values for all session state variables."""
    return {
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
        'extraction_mode': 'Chunked (Fast)',  # or 'Page-by-Page (Detailed)'
        
        # UI state
        'show_examples': False,
    }


def initialize_session_state() -> None:
    """Initialize all session state variables with default values."""
    import streamlit as st
    
    defaults = get_session_state_defaults()
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
