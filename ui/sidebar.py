"""Streamlit sidebar component for configuration."""

import os
import streamlit as st
from backend.auth import backend_login, normalize_base_url


def render_sidebar():
    """Render the sidebar with API key, model settings, and backend config."""
    
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key Section
        st.subheader("API Key")
        
        # Check for .env key
        env_key = os.getenv('GEMINI_API_KEY')
        if env_key:
            fingerprint = f"***{env_key[-6:]}" if len(env_key) >= 6 else "***"
            st.success(f"Using key from .env: {fingerprint}")
            st.session_state.api_key = env_key
            st.session_state.api_key_valid = True
            
            if st.button("🔄 Reset API Key", help="Clear .env key and enter manually"):
                st.session_state.api_key = ""
                st.session_state.api_key_valid = False
                st.rerun()
        else:
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.api_key,
                help="Enter your Gemini API key. Get one from https://makersuite.google.com/app/apikey",
                placeholder="AIzaSy..."
            )
            
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
            help="Lower values = more focused/deterministic"
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
        
        st.session_state.include_page_numbers = st.checkbox(
            "Include Page Numbers",
            value=st.session_state.include_page_numbers,
            help="Show the page number where each Q&A was found"
        )
        
        st.session_state.include_confidence = st.checkbox(
            "Show Confidence Scores",
            value=st.session_state.include_confidence,
            help="Display AI's confidence level for each extraction"
        )
        
        st.session_state.extract_metadata = st.checkbox(
            "Extract Metadata",
            value=st.session_state.extract_metadata,
            help="Include additional context like question type, difficulty, etc."
        )

        st.markdown("---")

        # Backend Configuration
        st.subheader("Backend API")

        st.session_state.backend_base_url = st.text_input(
            "Base URL",
            value=st.session_state.backend_base_url,
            help="Example: https://localhost:7237",
        )

        st.session_state.backend_verify_tls = st.checkbox(
            "Verify TLS",
            value=bool(st.session_state.get('backend_verify_tls', True)),
            help="Disable only for local dev with self-signed certs",
        )

        st.session_state.backend_base_url = normalize_base_url(st.session_state.backend_base_url)
        
        st.session_state.backend_email = st.text_input(
            "Email",
            value=st.session_state.backend_email,
            placeholder="admin@tapcet.com",
        )
        
        st.session_state.backend_password = st.text_input(
            "Password",
            type="password",
            value=st.session_state.backend_password,
        )

        st.session_state.backend_auto_login = st.checkbox(
            "Auto-login",
            value=bool(st.session_state.get('backend_auto_login', True)),
            help="Automatically login when token expires",
        )

        if st.button("Login to Backend", type="primary", use_container_width=True):
            if backend_login():
                st.success("Logged in successfully")
                st.rerun()
            else:
                error = st.session_state.get('backend_auth_error', 'Login failed')
                st.error(error)

        # Show login status
        if st.session_state.get('backend_token'):
            user = st.session_state.get('backend_user_email', 'Unknown')
            st.success(f"✓ Logged in as {user}")
        
        st.markdown("---")
        
        # Actions
        st.subheader("Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                # Import at use time to avoid circular import
                import importlib
                import sys
                if 'app' in sys.modules:
                    app = sys.modules['app']
                    app.reset_session()
                    st.rerun()
        
        with col2:
            if st.button("Clear", use_container_width=True, type="secondary"):
                # Import at use time to avoid circular import
                import importlib
                import sys
                if 'app' in sys.modules:
                    app = sys.modules['app']
                    app.clear_results()
                    st.rerun()
