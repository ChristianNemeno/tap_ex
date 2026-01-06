"""DEPRECATED: Legacy compatibility module.

This module is deprecated and kept only for backwards compatibility.
All functionality has been moved to the extraction package.

New code should import from:
    - extraction.gemini for GeminiConfig and extract_create_quiz_dto_from_pdf_bytes
    - extraction.pdf_utils for PDF manipulation utilities

This file will be removed in a future version.
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "The 'processor' module is deprecated. "
    "Please import from 'extraction.gemini' instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from extraction modules for backwards compatibility
from extraction.gemini import (
    DEFAULT_MODEL_NAME,
    MCQ_CREATE_QUIZ_PROMPT,
    REPAIR_TO_JSON_PROMPT_TEMPLATE,
    GeminiConfig,
    extract_create_quiz_dto_from_pdf_bytes,
)
from extraction.pdf_utils import (
    split_pdf_into_chunks,
    get_page_count,
)

# Make all exports available
__all__ = [
    'DEFAULT_MODEL_NAME',
    'MCQ_CREATE_QUIZ_PROMPT',
    'REPAIR_TO_JSON_PROMPT_TEMPLATE',
    'GeminiConfig',
    'extract_create_quiz_dto_from_pdf_bytes',
    'split_pdf_into_chunks',
    'get_page_count',
]
