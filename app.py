"""Main Streamlit application for PDF MCQ Extraction.

This application extracts multiple-choice questions from PDFs using Google's Gemini AI
and integrates with a backend API for quiz creation.
"""

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

import streamlit as st

from core.config import initialize_session_state, get_session_state_defaults
from core.logging_utils import get_logger
from ui.components import load_css, render_header, render_footer
from ui.sidebar import render_sidebar
from ui.upload import render_upload_section
from ui.filtered_pages import render_filtered_pages_section
from ui.results import render_results_section
from ui.export import render_export_section


# Initialize logger
logger = get_logger()


def reset_session():
    """Reset session state to defaults (preserves API key and settings)."""
    keys_to_keep = ['api_key', 'temperature', 'max_tokens', 'include_page_numbers', 
                    'include_confidence', 'extract_metadata']
    
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    
    initialize_session_state()


def clear_results():
    """Clear only the results data."""
    st.session_state.results = None
    st.session_state.processing_complete = False
    st.session_state.uploaded_file_name = None
    st.session_state.quiz_json_editor = None
    st.session_state.quiz_json_validated = None
    st.session_state.quiz_json_validation_errors = []
    st.session_state.created_quiz_response = None


def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="PDF Q&A Extractor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load custom CSS
    load_css()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    render_header()
    
    # Upload section
    render_upload_section()
    
    # Filtered pages review section (after OCR)
    if st.session_state.ocr_filtering_complete:
        render_filtered_pages_section()
    
    # Results section
    if st.session_state.processing_complete:
        render_results_section()
        render_export_section()
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()
