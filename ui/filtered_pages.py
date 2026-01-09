"""UI component for displaying and managing filtered PDF pages."""

import streamlit as st
from io import BytesIO


def render_filtered_pages_section():
    """Render the filtered pages review section."""
    
    if not st.session_state.ocr_filtering_complete:
        return
    
    st.header("📋 Filtered Pages Review")
    
    filtered_count = len(st.session_state.filtered_pages)
    total_pages = st.session_state.total_pages
    
    if filtered_count == 0:
        st.warning(f"No pages with questions detected out of {total_pages} total pages.")
        st.info("The OCR filter looks for multiple choice options, numbered questions, answer keys, and question patterns.")
        return
    
    st.success(f"Found **{filtered_count}** pages with questions out of **{total_pages}** total pages")
    
    # Summary info
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.metric("Pages with Questions", filtered_count)
    
    with col2:
        st.metric("Total Pages", total_pages)
    
    with col3:
        reduction = int((1 - filtered_count / total_pages) * 100) if total_pages > 0 else 0
        st.metric("AI Cost Reduction", f"{reduction}%")
    
    st.markdown("---")
    
    # Display filtered pages
    st.subheader("Review and Edit Filtered Pages")
    st.caption("Remove any pages that don't contain questions before processing with AI")
    
    # Create columns for page display
    cols_per_row = 3
    pages_data = list(zip(
        st.session_state.filtered_page_numbers,
        st.session_state.filtered_pages
    ))
    
    for idx in range(0, len(pages_data), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx, col in enumerate(cols):
            page_idx = idx + col_idx
            if page_idx >= len(pages_data):
                break
            
            page_num, page_pdf = pages_data[page_idx]
            
            with col:
                with st.container(border=True):
                    st.markdown(f"**Page {page_num}**")
                    
                    # Show PDF preview (note: st doesn't have native PDF preview, so show info)
                    st.caption(f"Size: {len(page_pdf) / 1024:.1f} KB")
                    
                    # Download button for this page
                    st.download_button(
                        label="📥 Download",
                        data=page_pdf,
                        file_name=f"page_{page_num}.pdf",
                        mime="application/pdf",
                        key=f"download_page_{page_num}",
                        use_container_width=True,
                    )
                    
                    # Remove button
                    if st.button(
                        "🗑️ Remove",
                        key=f"remove_page_{page_num}",
                        use_container_width=True,
                        type="secondary"
                    ):
                        remove_page(page_idx)
                        st.rerun()
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🔄 Re-scan PDF", use_container_width=True):
            # Clear filtered pages and allow re-upload
            st.session_state.ocr_filtering_complete = False
            st.session_state.filtered_pages = []
            st.session_state.filtered_page_numbers = []
            st.session_state.total_pages = 0
            st.rerun()
    
    with col2:
        if len(st.session_state.filtered_pages) > 0:
            if st.button(
                f"🤖 Process {len(st.session_state.filtered_pages)} Pages with AI",
                use_container_width=True,
                type="primary"
            ):
                process_with_ai()
        else:
            st.button(
                "🤖 Process with AI (No pages selected)",
                disabled=True,
                use_container_width=True
            )
    
    with col3:
        if st.button("❌ Clear All", use_container_width=True):
            clear_all_pages()
            st.rerun()


def remove_page(page_idx: int):
    """Remove a page from the filtered list."""
    if 0 <= page_idx < len(st.session_state.filtered_pages):
        st.session_state.filtered_pages.pop(page_idx)
        st.session_state.filtered_page_numbers.pop(page_idx)
        st.toast(f"Removed page from selection", icon="🗑️")


def clear_all_pages():
    """Clear all filtered pages."""
    st.session_state.filtered_pages = []
    st.session_state.filtered_page_numbers = []
    st.session_state.ocr_filtering_complete = False
    st.toast("Cleared all filtered pages", icon="❌")


def process_with_ai():
    """Process the filtered pages with Gemini AI."""
    from extraction.gemini import GeminiConfig, process_filtered_pages_with_ai
    
    with st.spinner("Processing with Gemini AI... This may take 1-5 minutes."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            cfg = GeminiConfig(
                api_key=st.session_state.api_key,
                temperature=st.session_state.temperature,
                max_output_tokens=st.session_state.max_tokens,
            )
            
            def update_progress(msg: str, pct: float):
                status_text.text(msg)
                progress_bar.progress(min(int(pct * 100), 100))
            
            results = process_filtered_pages_with_ai(
                filtered_page_pdfs=st.session_state.filtered_pages,
                filtered_page_numbers=st.session_state.filtered_page_numbers,
                filename=st.session_state.uploaded_file_name or "document.pdf",
                cfg=cfg,
                progress_callback=update_progress,
            )
            
            # Store results
            st.session_state.results = results
            st.session_state.processing_complete = True
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Successfully extracted questions from {len(st.session_state.filtered_pages)} pages!")
            st.balloons()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during AI processing: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
