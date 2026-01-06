"""Results section UI with JSON editor and backend integration."""

import json

import streamlit as st

from backend.auth import backend_token_is_valid
from backend.quiz_api import create_quiz_in_backend


def _flatten_page_by_page_results(results: dict) -> dict:
    """Flatten page-by-page results into standard CreateQuizDto format.
    
    Combines all questions from all pages into a single questions array
    for backend compatibility.
    """
    all_questions = []
    pages = results.get('pages', [])
    
    for page_data in pages:
        questions = page_data.get('questions', [])
        all_questions.extend(questions)
    
    return {
        'title': results.get('title', 'Untitled Quiz'),
        'description': results.get('description', ''),
        'questions': all_questions,
    }


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
    
    # Check if this is page-by-page results
    is_page_by_page = 'pages' in results and isinstance(results.get('pages'), list)
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader(title)
        if description:
            st.caption(description)
    
    with col2:
        if is_page_by_page:
            total_questions = sum(page.get('questionCount', 0) for page in results.get('pages', []))
            st.metric("Total Questions", total_questions)
        else:
            questions = results.get('questions', [])
            st.metric("Questions", len(questions))
    
    with col3:
        if is_page_by_page:
            st.metric("Pages", results.get('totalPages', 0))
    
    st.markdown("---")
    
    # Display page-by-page breakdown if available
    if is_page_by_page:
        st.subheader("📄 Questions by Page")
        
        pages = results.get('pages', [])
        for page_data in pages:
            page_num = page_data.get('pageNumber', 0)
            questions = page_data.get('questions', [])
            question_count = page_data.get('questionCount', 0)
            
            with st.expander(f"Page {page_num} ({question_count} question{'s' if question_count != 1 else ''})", expanded=question_count > 0):
                if question_count == 0:
                    st.info("No questions found on this page")
                else:
                    for idx, question in enumerate(questions, 1):
                        st.markdown(f"**Q{idx}: {question.get('text', 'N/A')}**")
                        
                        choices = question.get('choices', [])
                        for choice in choices:
                            is_correct = choice.get('isCorrect', False)
                            icon = "✅" if is_correct else "⭕"
                            st.markdown(f"{icon} {choice.get('text', 'N/A')}")
                        
                        if question.get('explanation'):
                            st.caption(f"💡 {question.get('explanation')}")
                        
                        if idx < len(questions):
                            st.markdown("---")
        
        st.markdown("---")
    
    # JSON Editor
    st.subheader("JSON Editor")
    st.caption("Review and edit the extracted quiz data before sending to backend")
    
    # Initialize editor content if not set
    if 'quiz_json_editor' not in st.session_state or st.session_state.quiz_json_editor is None:
        # For page-by-page results, flatten for backend compatibility
        if is_page_by_page:
            flattened = _flatten_page_by_page_results(results)
            st.session_state.quiz_json_editor = json.dumps(flattened, indent=2)
        else:
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
            if is_page_by_page:
                flattened = _flatten_page_by_page_results(results)
                st.session_state.quiz_json_editor = json.dumps(flattened, indent=2)
            else:
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
