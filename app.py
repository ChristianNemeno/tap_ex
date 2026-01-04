"""
PDF Q&A Extractor - AI-Powered Document Processing
Extract questions and answers from PDF documents using Google's Gemini 1.5 Flash
"""

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import pandas as pd
from datetime import datetime
import tempfile
import time

from processor import GeminiConfig, extract_qa_pairs_from_pdf_bytes

# Load environment variables
load_dotenv()

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
        'uploaded_file_name': None,
        'processing_complete': False,
        
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
    if st.session_state.results:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Q&A", len(st.session_state.results))
        
        with col2:
            high_conf = sum(1 for r in st.session_state.results if r.get('confidence') == 'high')
            st.metric("High Confidence", high_conf)
        
        with col3:
            pages = set(r.get('page_number', 'N/A') for r in st.session_state.results)
            st.metric("Pages Covered", len(pages))
        
        with col4:
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

            status_text.text("Step 3/4: Extracting Q&A with Gemini...")
            progress_bar.progress(65)

            results = extract_qa_pairs_from_pdf_bytes(
                pdf_bytes=pdf_bytes,
                filename=uploaded_file.name,
                cfg=cfg,
            )

            status_text.text("Step 4/4: Finalizing...")
            progress_bar.progress(95)

            st.session_state.results = results
            st.session_state.processing_complete = True
            progress_bar.progress(100)
            st.success(f"Successfully extracted {len(results)} Q&A pairs.")

        except Exception as e:
            st.session_state.processing_complete = False
            st.session_state.results = None
            st.error(f"An error occurred: {str(e)}")
            st.info("Check your API key and try again.")

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
                st.button("Extract Q&A Pairs", disabled=True, use_container_width=True, type="primary")
            else:
                if st.button("Extract Q&A Pairs", use_container_width=True, type="primary"):
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
    """Render extracted Q&A results"""

    results = st.session_state.get('results')
    if not results:
        return

    st.markdown("---")
    st.subheader("Extraction Results")

    total_qa = len(results)
    high_conf = sum(1 for r in results if r.get('confidence') == 'high')
    unique_pages = len(set(r.get('page_number', 'N/A') for r in results))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Q&A", total_qa)
    with col2:
        st.metric("High Confidence", high_conf)
    with col3:
        st.metric("Unique Pages", unique_pages)

    st.markdown("")

    col1, col2 = st.columns([3, 2])
    with col1:
        view_mode = st.radio(
            "View Mode",
            options=["Table", "Cards", "JSON"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with col2:
        search_query = st.text_input(
            "Search",
            placeholder="Search in questions and answers",
            label_visibility="collapsed",
        )

    filtered = results
    if search_query:
        q = search_query.strip().lower()
        if q:
            filtered = [
                r for r in results
                if q in str(r.get('question', '')).lower() or q in str(r.get('answer', '')).lower()
            ]
            st.info(f"Matches: {len(filtered)}")

    if view_mode == "Table":
        df = pd.DataFrame(filtered)

        column_order = ['question_number', 'question', 'answer']
        if st.session_state.get('include_page_numbers', True):
            column_order.append('page_number')
        if st.session_state.get('include_confidence', True):
            column_order.append('confidence')

        df = df[[c for c in column_order if c in df.columns]]
        rename_map = {
            'question_number': 'Question #',
            'page_number': 'Page',
            'confidence': 'Confidence',
            'question': 'Question',
            'answer': 'Answer',
        }
        df = df.rename(columns=rename_map)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Question #': st.column_config.NumberColumn('Question #', width='small'),
                'Page': st.column_config.NumberColumn('Page', width='small'),
            },
        )

    elif view_mode == "Cards":
        if not filtered:
            st.info("No results to display.")
            return

        for idx, item in enumerate(filtered, start=1):
            qn = item.get('question_number', idx)
            question = str(item.get('question', '')).strip()
            answer = str(item.get('answer', '')).strip()
            page_number = item.get('page_number', 'N/A')
            confidence = item.get('confidence', 'N/A')

            header = f"Q{qn}: {question[:90]}" if question else f"Q{qn}"
            with st.expander(header, expanded=(idx == 1)):
                if question:
                    st.markdown("**Question**")
                    st.write(question)
                if answer:
                    st.markdown("**Answer**")
                    st.write(answer)

                meta_parts = []
                if st.session_state.get('include_page_numbers', True):
                    meta_parts.append(f"Page: {page_number}")
                if st.session_state.get('include_confidence', True):
                    meta_parts.append(f"Confidence: {confidence}")
                if meta_parts:
                    st.caption(" | ".join(meta_parts))

    else:
        st.code(json.dumps(filtered, indent=2), language='json')


def render_export_section():
    """Render export buttons for extracted data"""

    results = st.session_state.get('results')
    if not results:
        return

    st.markdown("---")
    st.subheader("Export")

    df = pd.DataFrame(results)
    csv_data = df.to_csv(index=False).encode('utf-8')
    json_data = json.dumps(results, indent=2)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"qa_pairs_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
            type="secondary",
        )
    with col2:
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"qa_pairs_{ts}.json",
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
