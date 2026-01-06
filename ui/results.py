"""Results section UI with JSON editor and backend integration."""

import json

import streamlit as st

from backend.auth import backend_token_is_valid
from backend.quiz_api import create_quiz_in_backend


def render_results_section():
    """Render the results section with JSON editor and backend integration."""
    
    results = st.session_state.results
    
    if not isinstance(results, dict):
        st.info("No results yet. Upload a PDF and click 'Extract Quiz' to get started.")
        return
    
    st.header("📊 Extraction Results")
    
    # Display title and description
    title = results.get('title', 'Untitled Quiz')
    description = results.get('description', '')
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(title)
        if description:
            st.caption(description)
    
    with col2:
        questions = results.get('questions', [])
        st.metric("Questions", len(questions))
    
    st.markdown("---")
    
    # JSON Editor
    st.subheader("JSON Editor")
    st.caption("Review and edit the extracted quiz data before sending to backend")
    
    # Initialize editor content if not set
    if 'quiz_json_editor' not in st.session_state or st.session_state.quiz_json_editor is None:
        st.session_state.quiz_json_editor = json.dumps(results, indent=2)
    
    # Text area for JSON editing
    edited_json = st.text_area(
        "Quiz JSON",
        value=st.session_state.quiz_json_editor,
        height=400,
        help="Edit the JSON directly. Click Validate to check for errors.",
        label_visibility="collapsed"
    )
    
    st.session_state.quiz_json_editor = edited_json
    
    # Validation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Validate JSON", use_container_width=True):
            try:
                obj = json.loads(edited_json)
                from core.validation import validate_and_normalize_create_quiz_dto
                payload, errs = validate_and_normalize_create_quiz_dto(obj)
                st.session_state.quiz_json_validated = payload
                st.session_state.quiz_json_validation_errors = errs
                
                if payload is None:
                    st.error("Validation failed")
                elif errs:
                    st.warning("Validated with warnings")
                else:
                    st.success("JSON valid")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
    
    with col2:
        if st.button("Reset to Original", use_container_width=True):
            st.session_state.quiz_json_editor = json.dumps(results, indent=2)
            st.session_state.quiz_json_validated = None
            st.session_state.quiz_json_validation_errors = []
            st.rerun()
    
    with col3:
        if st.button("Download JSON", use_container_width=True):
            st.download_button(
                label="Download",
                data=edited_json,
                file_name=f"{title.replace(' ', '_')}.json",
                mime="application/json"
            )
    
    # Display validation results
    errs = st.session_state.get('quiz_json_validation_errors', [])
    if errs:
        payload = st.session_state.get('quiz_json_validated')
        if payload is None:
            st.error("Validation failed. Fix the issues below.")
        else:
            st.warning("Validated with warnings:")
        for msg in errs:
            st.write(f"- {msg}")
    elif st.session_state.get('quiz_json_validated') is not None:
        st.success("JSON validated and ready to send")

    st.markdown("---")
    
    # Backend integration
    st.subheader("Backend")

    if backend_token_is_valid():
        st.success("Ready to create quiz in backend")
    else:
        st.info("Login in the sidebar to create quiz")

    if st.button("Create Quiz in Backend", type="primary", use_container_width=True):
        success, error_msg, response_data = create_quiz_in_backend(edited_json)
        
        if success:
            st.success("Quiz created successfully!")
            if response_data:
                st.session_state.created_quiz_response = response_data
                with st.expander("View Response"):
                    st.json(response_data)
        else:
            st.error(error_msg or "Failed to create quiz")
            
            # Show validation errors if any
            errs = st.session_state.get('quiz_json_validation_errors', [])
            if errs:
                with st.expander("Validation Errors"):
                    for msg in errs:
                        st.write(f"- {msg}")
