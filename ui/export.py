"""Export section UI for downloading results."""

import json
from datetime import datetime

import streamlit as st
import pandas as pd


def render_export_section():
    """Render the export section with download options."""
    
    results = st.session_state.results
    
    if not isinstance(results, dict):
        return
    
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON export
        st.subheader("JSON Format")
        json_str = json.dumps(results, indent=2)
        
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # CSV export
        st.subheader("CSV Format")
        
        # Convert questions to DataFrame
        questions = results.get('questions', [])
        if questions:
            rows = []
            for q in questions:
                choices_text = "; ".join([
                    f"{c.get('text', '')} {'(correct)' if c.get('isCorrect') else ''}"
                    for c in q.get('choices', [])
                ])
                rows.append({
                    'Question': q.get('text', ''),
                    'Explanation': q.get('explanation', ''),
                    'Choices': choices_text,
                    'ImageUrl': q.get('imageUrl', '')
                })
            
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
