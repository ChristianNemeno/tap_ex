"""Gemini processing utilities for PDF Q&A extraction."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import google.generativeai as genai


DEFAULT_MODEL_NAME = "gemini-3-flash-preview"


EXTRACTION_PROMPT = """
You are an expert document analyst.

Task:
Extract all question and answer pairs from the attached PDF.

Rules:
- Only use information explicitly present in the document.
- If an answer is not explicitly stated, return "Answer not found in document".
- Include page_number when available.
- Return only valid JSON. Do not include any extra text.

Output format:
[
  {
    "question_number": 1,
    "question": "...",
    "answer": "...",
    "page_number": 1,
    "confidence": "high"
  }
]

Confidence values must be one of: high, medium, low.
""".strip()


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = 0.2
    max_output_tokens: int = 8192


def extract_qa_pairs_from_pdf_bytes(
    *,
    pdf_bytes: bytes,
    filename: str,
    cfg: GeminiConfig,
) -> List[Dict[str, Any]]:
    """Send a PDF to Gemini and return a list of extracted Q&A objects."""

    if not cfg.api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    genai.configure(api_key=cfg.api_key)

    generation_config: Dict[str, Any] = {
        "temperature": float(cfg.temperature),
        "max_output_tokens": int(cfg.max_output_tokens),
        # Ask the SDK/model to return JSON when possible.
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(model_name=cfg.model_name, generation_config=generation_config)

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = tmp.name

        uploaded = genai.upload_file(path=tmp_path, display_name=filename)
        resp = model.generate_content([uploaded, EXTRACTION_PROMPT])

        text = getattr(resp, "text", None)
        if not text:
            # Some SDK versions provide candidates/parts; fall back to string.
            text = str(resp)

        return _parse_and_normalize_response(text)

    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _parse_and_normalize_response(raw: str) -> List[Dict[str, Any]]:
    data = _safe_json_load(raw)

    if not isinstance(data, list):
        raise ValueError("Gemini response JSON must be a list")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue

        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue

        qn = item.get("question_number", idx)
        page_number = item.get("page_number", "N/A")
        confidence = item.get("confidence", "medium")

        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        normalized.append(
            {
                "question_number": qn,
                "question": question.strip(),
                "answer": answer.strip(),
                "page_number": page_number,
                "confidence": confidence,
            }
        )

    if not normalized:
        raise ValueError("No valid Q&A items found in model response")

    return normalized


def _safe_json_load(raw: str) -> Any:
    """Try hard to parse JSON from the model response."""

    s = raw.strip()

    # Remove markdown fences if present.
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()

    # Try direct parse.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Try to extract the first JSON array.
    match = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to extract the first JSON object.
    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Failed to parse JSON from model response")
