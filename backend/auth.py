"""Backend API authentication and token management."""

import time
from datetime import datetime
from typing import Any, Optional

import requests
import streamlit as st


def normalize_base_url(url: str) -> str:
    """Normalize backend base URL by stripping trailing slashes."""
    s = (url or '').strip()
    while s.endswith('/'):
        s = s[:-1]
    return s


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    """Parse ISO datetime string with optional Zulu suffix."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # Handle common Zulu suffix
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def backend_token_is_valid() -> bool:
    """Check if current backend token is valid and not expired."""
    token = st.session_state.get('backend_token')
    expires_at = st.session_state.get('backend_token_expires_at')
    if not token or not isinstance(token, str):
        return False
    if not isinstance(expires_at, datetime):
        return True
    if expires_at.tzinfo is None:
        return datetime.now() < expires_at
    return datetime.now(expires_at.tzinfo) < expires_at


def backend_login(*, timeout_s: int = 20) -> bool:
    """Login to backend and cache JWT token in session state.
    
    Returns True if login successful, False otherwise.
    Stores auth error in st.session_state.backend_auth_error if login fails.
    """

    base_url = normalize_base_url(st.session_state.get('backend_base_url', ''))
    email = (st.session_state.get('backend_email') or '').strip()
    password = st.session_state.get('backend_password') or ''

    st.session_state.backend_auth_error = None

    if not base_url:
        st.session_state.backend_auth_error = 'Missing backend base URL'
        return False
    if not email or not password:
        st.session_state.backend_auth_error = 'Missing backend email/password'
        return False

    # Simple throttle to avoid hammering during reruns
    now_ts = time.time()
    last_attempt = float(st.session_state.get('backend_last_login_attempt') or 0.0)
    if now_ts - last_attempt < 3.0:
        return False
    st.session_state.backend_last_login_attempt = now_ts

    url = f"{base_url}/api/auth/login"
    try:
        resp = requests.post(
            url,
            json={"email": email, "password": password},
            timeout=timeout_s,
            verify=bool(st.session_state.get('backend_verify_tls', True)),
        )
    except requests.RequestException as e:
        st.session_state.backend_auth_error = f"Login request failed: {e}"
        return False

    if resp.status_code != 200:
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        msg = None
        if isinstance(payload, dict):
            msg = payload.get('message')
        st.session_state.backend_auth_error = msg or f"Login failed (HTTP {resp.status_code})"
        return False

    try:
        data = resp.json()
    except ValueError:
        st.session_state.backend_auth_error = 'Login response was not JSON'
        return False

    token = data.get('token')
    if not isinstance(token, str) or not token.strip():
        st.session_state.backend_auth_error = 'Login response missing token'
        return False

    st.session_state.backend_token = token.strip()
    st.session_state.backend_token_expires_at = parse_iso_datetime(data.get('expiresAt'))
    st.session_state.backend_user_email = data.get('email') if isinstance(data.get('email'), str) else None
    st.session_state.backend_user_name = data.get('userName') if isinstance(data.get('userName'), str) else None
    st.session_state.backend_auth_error = None
    return True


def ensure_backend_token() -> bool:
    """Ensure we have a valid backend token, optionally auto-login.
    
    Returns True if token is valid (or successfully obtained), False otherwise.
    """
    if backend_token_is_valid():
        return True
    if not st.session_state.get('backend_auto_login', True):
        return False
    return backend_login()
