"""Backend API integration for quiz creation."""

import json
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

from backend.auth import normalize_base_url, ensure_backend_token
from core.validation import validate_and_normalize_create_quiz_dto


def create_quiz_in_backend(json_text: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Create quiz in backend from JSON text.
    
    Args:
        json_text: JSON string of CreateQuizDto payload
        
    Returns:
        Tuple of (success, error_message, response_data)
        - success: True if quiz created successfully
        - error_message: Error message if failed, None if success
        - response_data: Response JSON from backend if successful, None otherwise
    """
    
    # Ensure we have valid token
    if not ensure_backend_token():
        error = st.session_state.get('backend_auth_error') or 'Not authorized'
        return False, error, None
    
    # Parse and validate JSON
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None
    
    payload, errs = validate_and_normalize_create_quiz_dto(obj)
    
    # Store validation results in session state
    st.session_state.quiz_json_validated = payload
    st.session_state.quiz_json_validation_errors = errs
    
    if payload is None:
        error_msg = "Validation failed:\n" + "\n".join(f"- {err}" for err in errs)
        return False, error_msg, None
    
    # Make POST request
    base_url = normalize_base_url(st.session_state.get('backend_base_url', ''))
    url = f"{base_url}/api/quiz"
    token = st.session_state.get('backend_token')
    
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
            verify=bool(st.session_state.get('backend_verify_tls', True)),
        )
    except requests.RequestException as e:
        return False, f"Request failed: {e}", None
    
    # Handle response
    if resp.status_code == 201:
        try:
            response_data = resp.json()
        except ValueError:
            response_data = None
        return True, None, response_data
    
    elif resp.status_code == 401:
        # Token expired or invalid
        st.session_state.backend_token = None
        return False, "Unauthorized (token expired or invalid)", None
    
    else:
        try:
            error_payload = resp.json()
        except ValueError:
            error_payload = resp.text
        
        error_msg = f"Create quiz failed (HTTP {resp.status_code})"
        if isinstance(error_payload, dict):
            if 'message' in error_payload:
                error_msg += f": {error_payload['message']}"
        
        return False, error_msg, error_payload
