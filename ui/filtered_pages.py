"""UI component for displaying and managing filtered pages with OCR text."""

import streamlit as st


def render_filtered_pages_section():
    """Render the filtered pages review section."""
    
    if not st.session_state.ocr_filtering_complete:
        return
    
    st.header("📋 Filtered Pages Review")
    
    filtered_count = len(st.session_state.filtered_pages)
    total_pages = st.session_state.total_pages
    
    if filtered_count == 0:
        st.warning(f"No pages extracted from PDF.")
        return
    
    st.success(f"Extracted text from **{filtered_count}** pages (all pages)")
    
    # Summary info with better visibility
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;'>
            <h3 style='color: #0068c9; margin: 0;'>{filtered_count}</h3>
            <p style='color: #262730; margin: 0;'>Pages Extracted</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;'>
            <h3 style='color: #0068c9; margin: 0;'>{total_pages}</h3>
            <p style='color: #262730; margin: 0;'>Total Pages</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display filtered pages
    st.subheader("Review Extracted Text")
    st.caption("Remove any pages you don't want to process with AI")
    
    # Create columns for page display
    cols_per_row = 2  # Reduced to 2 columns for better text readability
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
            
            page_num, page_text = pages_data[page_idx]
            
            with col:
                with st.container(border=True):
                    st.markdown(f"**Page {page_num}**")
                    
                    # Debug: Show if text is empty
                    if not page_text or len(page_text.strip()) == 0:
                        st.warning("⚠️ No text extracted from this page")
                        st.caption("OCR may have failed or page contains only images")
                    else:
                        # Show text preview (first 500 characters)
                        preview_text = page_text[:500] + "..." if len(page_text) > 500 else page_text
                        st.text_area(
                            "Content Preview",
                            value=preview_text,
                            height=200,
                            key=f"preview_{page_num}",
                            label_visibility="collapsed"
                        )
                        
                        st.caption(f"Text length: {len(page_text)} characters")
                    
                    # Action buttons
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        # View full text
                        if st.button(
                            "👁️ View Full",
                            key=f"view_page_{page_num}",
                            use_container_width=True,
                            help="View complete text"
                        ):
                            st.session_state[f"show_full_{page_num}"] = True
                    
                    with btn_col2:
                        # Remove button
                        if st.button(
                            "🗑️ Remove",
                            key=f"remove_page_{page_num}",
                            use_container_width=True,
                            type="secondary",
                            help="Remove this page"
                        ):
                            remove_page(page_idx)
                            st.rerun()
                    
                    # Show full text in expander if requested
                    if st.session_state.get(f"show_full_{page_num}", False):
                        with st.expander("Full Page Text", expanded=True):
                            st.text_area(
                                f"Full text from page {page_num}",
                                value=page_text,
                                height=400,
                                key=f"full_{page_num}",
                                label_visibility="collapsed"
                            )
                            if st.button("Close", key=f"close_{page_num}"):
                                st.session_state[f"show_full_{page_num}"] = False
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
    from extraction.gemini import GeminiConfig, process_pages_with_ai
    
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
            
            # Get original PDF bytes from session state
            pdf_bytes = st.session_state.get('uploaded_pdf_bytes', None)
            
            results = process_pages_with_ai(
                ocr_texts=st.session_state.filtered_pages,
                page_numbers=st.session_state.filtered_page_numbers,
                filename=st.session_state.uploaded_file_name or "document.pdf",
                cfg=cfg,
                progress_callback=update_progress,
                pdf_bytes=pdf_bytes,
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
