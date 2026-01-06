"""Upload section UI and document processing."""

import io
from datetime import datetime

import streamlit as st
from pypdf import PdfReader

from extraction.gemini import GeminiConfig, extract_create_quiz_dto_from_pdf_bytes, extract_create_quiz_dto_by_page


def render_upload_section():
    """Render the file upload section."""
    
    st.header("📄 Upload Document")
    
    # Extraction mode selection
    extraction_mode = st.radio(
        "Extraction Mode",
        options=["Chunked (Fast)", "Page-by-Page (Detailed)"],
        help="Chunked: Groups pages into 20-page chunks. Page-by-Page: Processes each page individually with rate limiting (slower but more precise)",
        horizontal=True
    )
    st.session_state.extraction_mode = extraction_mode
    
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
        
        # Validate page count
        try:
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            page_count = len(reader.pages)
            st.info(f"**Pages:** {page_count}")
            
            if page_count > 200:
                st.error("PDF has more than 200 pages. Please split your PDF or upload a smaller document.")
                return None
            elif page_count > 100:
                st.warning("Large document detected. Processing may take 3-10 minutes.")
        except Exception:
            pass
        
        # Warning for large files
        if file_size > 20:
            st.error("File size exceeds 20MB. Please upload a smaller file.")
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
        # Empty state
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


def process_document(uploaded_file):
    """Process the uploaded document with Gemini API."""

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

            status_text.text("Step 3/5: Validating PDF...")
            progress_bar.progress(40)

            def update_progress(msg: str, pct: float):
                status_text.text(msg)
                # Map pct (0.0-1.0) to progress bar range 40-95
                bar_pct = int(40 + (pct * 55))
                progress_bar.progress(bar_pct)

            # Choose extraction method based on mode
            if st.session_state.get('extraction_mode') == "Page-by-Page (Detailed)":
                results = extract_create_quiz_dto_by_page(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded_file.name,
                    cfg=cfg,
                    progress_callback=update_progress,
                )
            else:
                results = extract_create_quiz_dto_from_pdf_bytes(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded_file.name,
                    cfg=cfg,
                    progress_callback=update_progress,
                )

            status_text.text("Step 5/5: Finalizing...")
            progress_bar.progress(95)

            st.session_state.results = results
            st.session_state.processing_complete = True

            progress_bar.progress(100)
            status_text.text("✅ Extraction complete!")

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.session_state.processing_complete = False
            st.session_state.results = None
