"""Common UI components and utilities."""

import streamlit as st


def render_header():
    """Render the application header."""
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


def render_footer():
    """Render the application footer."""
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


def load_css():
    """Load custom CSS styles."""
    st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Headers */
        h1 {
            color: #1f77b4;
            font-weight: 600;
        }
        
        h2 {
            color: #2c3e50;
            margin-top: 2rem;
        }
        
        /* Cards */
        .stAlert {
            border-radius: 8px;
        }
        
        /* Buttons */
        .stButton button {
            border-radius: 6px;
            font-weight: 500;
        }
        
        /* File uploader */
        .stFileUploader {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 2rem;
        }
        
        /* Text input */
        .stTextInput input {
            border-radius: 6px;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Metrics */
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
