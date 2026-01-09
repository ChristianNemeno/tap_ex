"""Upload section UI and document processing."""

import io
from datetime import datetime

import streamlit as st
from pypdf import PdfReader

from extraction.gemini import GeminiConfig, filter_pages_with_questions
from extraction.pdf_utils import get_page_count


def render_upload_section():
    """Render the file upload section."""
    
    st.header("📄 Upload Document")
    
    # Only show upload if not already filtered
    if st.session_state.ocr_filtering_complete:
        st.info(f"✅ OCR filtering complete. Review filtered pages below.")
        if st.button("🔄 Upload Different PDF"):
            st.session_state.ocr_filtering_complete = False
            st.session_state.filtered_pages = []
            st.session_state.filtered_page_numbers = []
            st.session_state.total_pages = 0
            st.rerun()
        return None
    
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
                st.button("Scan for Questions (OCR)", disabled=True, use_container_width=True, type="primary")
            else:
                if st.button("🔍 Scan for Questions (OCR)", use_container_width=True, type="primary"):
                    scan_document_ocr(uploaded_file)
        
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


def scan_document_ocr(uploaded_file):
    """Scan document with OCR to filter pages with questions."""

    with st.spinner("Scanning document with OCR... This takes 0.5-1 second per page."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Reset state
            st.session_state.ocr_filtering_complete = False
            st.session_state.filtered_pages = []
            st.session_state.filtered_page_numbers = []
            st.session_state.processing_complete = False
            st.session_state.results = None

            status_text.text("Reading PDF...")
            progress_bar.progress(5)
            pdf_bytes = uploaded_file.getvalue()
            
            # Get total page count
            total_pages = get_page_count(pdf_bytes)
            st.session_state.total_pages = total_pages

            status_text.text(f"Starting OCR scan on {total_pages} pages...")
            progress_bar.progress(10)

            def update_progress(msg: str, pct: float):
                status_text.text(msg)
                # Map pct (0.0-1.0) to progress bar range 10-95
                bar_pct = int(10 + (pct * 85))
                progress_bar.progress(min(bar_pct, 95))

            # Run OCR filtering
            filtered_page_numbers, filtered_page_pdfs = filter_pages_with_questions(
                pdf_bytes=pdf_bytes,
                progress_callback=update_progress,
            )

            # Store in session state
            st.session_state.filtered_pages = filtered_page_pdfs
            st.session_state.filtered_page_numbers = filtered_page_numbers
            st.session_state.ocr_filtering_complete = True

            progress_bar.progress(100)
            status_text.text(f"✅ Found {len(filtered_page_pdfs)} pages with questions!")
            
            st.success(f"OCR scan complete! Found **{len(filtered_page_pdfs)}** pages with questions out of **{total_pages}** total pages.")
            st.balloons()

        except Exception as e:
            st.error(f"Error during OCR scanning: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.session_state.ocr_filtering_complete = False
