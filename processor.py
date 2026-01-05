"""Gemini processing utilities for PDF extraction.

This module extracts quiz-ready MCQs directly in the backend's CreateQuizDto JSON shape.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import google.generativeai as genai


DEFAULT_MODEL_NAME = "gemini-3-flash-preview"


MCQ_CREATE_QUIZ_PROMPT = """
You are an expert at extracting multiple-choice quizzes from PDFs.

Task:
Extract the quiz content from the attached PDF and output a single JSON object matching this exact shape (CreateQuizDto):

{
    "title": "...",
    "description": "...", 
    "questions": [
        {
            "text": "...",
            "explanation": "...",
            "imageUrl": "...",
            "choices": [
                { "text": "...", "isCorrect": true },
                { "text": "...", "isCorrect": false }
            ]
        }
    ]
}

Rules (must follow):
- Return ONLY valid JSON. No markdown. No extra text.
- Questions must be multiple-choice.
- Each question must have 2 to 6 choices.
- Each question must have EXACTLY ONE choice where isCorrect=true.
- Use only information explicitly present in the PDF to determine the correct answer.
- If the PDF does not clearly indicate the correct answer, choose the most likely correct option and set explanation to "Correct answer not clearly indicated".
- title: 3-200 chars.
- description: optional (null or empty ok), max 2000 chars.
- question.text: 5-100 chars.
- question.explanation: optional, max 300 chars.
- imageUrl: optional; include only if explicitly provided and is a valid URL.

Output strictly as JSON.
""".strip()


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = 0.2
    max_output_tokens: int = 8192


def extract_create_quiz_dto_from_pdf_bytes(
    *,
    pdf_bytes: bytes,
    filename: str,
    cfg: GeminiConfig,
) -> Dict[str, Any]:
    """Send a PDF to Gemini and return a CreateQuizDto-shaped dict."""

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
        resp = model.generate_content([uploaded, MCQ_CREATE_QUIZ_PROMPT])

        text = getattr(resp, "text", None)
        if not text:
            # Some SDK versions provide candidates/parts; fall back to string.
            text = str(resp)

        return _parse_and_normalize_create_quiz_dto(text)

    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _parse_and_normalize_create_quiz_dto(raw: str) -> Dict[str, Any]:
    data = _safe_json_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Gemini response JSON must be an object for CreateQuizDto")

    title = data.get("title")
    description = data.get("description", None)
    questions = data.get("questions")

    if not isinstance(title, str):
        raise ValueError("CreateQuizDto.title must be a string")
    title = _clamp_text(title.strip(), min_len=3, max_len=200)
    if not title:
        raise ValueError("CreateQuizDto.title is required")

    if description is None:
        normalized_description: Optional[str] = None
    elif isinstance(description, str):
        normalized_description = _clamp_text(description.strip(), min_len=0, max_len=2000)
    else:
        normalized_description = None

    if not isinstance(questions, list) or not questions:
        raise ValueError("CreateQuizDto.questions must be a non-empty list")

    normalized_questions: List[Dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue

        q_text = q.get("text")
        if not isinstance(q_text, str):
            continue
        q_text = _clamp_text(q_text.strip(), min_len=5, max_len=100)
        if not q_text:
            continue

        q_expl = q.get("explanation", None)
        if isinstance(q_expl, str):
            q_expl_norm: Optional[str] = _clamp_text(q_expl.strip(), min_len=0, max_len=300)
        else:
            q_expl_norm = None

        image_url = q.get("imageUrl", None)
        if isinstance(image_url, str):
            image_url = image_url.strip()
            if not _looks_like_url(image_url):
                image_url = None
        else:
            image_url = None

        choices = q.get("choices")
        if not isinstance(choices, list):
            continue
        choices_norm = _normalize_choices(choices)
        if choices_norm is None:
            continue

        normalized_questions.append(
            {
                "text": q_text,
                "explanation": q_expl_norm,
                "imageUrl": image_url,
                "choices": choices_norm,
            }
        )

    if not normalized_questions:
        raise ValueError("No valid questions found in model response")

    result: Dict[str, Any] = {
        "title": title,
        "description": normalized_description,
        "questions": normalized_questions,
    }

    return _drop_none_fields(result)


def _normalize_choices(choices: List[Any]) -> Optional[List[Dict[str, Any]]]:
    raw_items: List[Dict[str, Any]] = []
    for c in choices:
        if not isinstance(c, dict):
            continue
        text = c.get("text")
        is_correct = c.get("isCorrect")
        if not isinstance(text, str):
            continue
        text = _clamp_text(text.strip(), min_len=1, max_len=500)
        if not text:
            continue
        raw_items.append({"text": text, "isCorrect": bool(is_correct)})

    # Enforce 2-6 choices
    if len(raw_items) < 2:
        return None
    if len(raw_items) > 6:
        raw_items = raw_items[:6]

    # Enforce exactly one correct answer
    correct_indices = [i for i, c in enumerate(raw_items) if c.get("isCorrect") is True]
    if len(correct_indices) == 1:
        return raw_items

    if len(correct_indices) == 0:
        raw_items[0]["isCorrect"] = True
        return raw_items

    # Too many marked correct: keep the first correct, set the rest false
    keep = correct_indices[0]
    for i in correct_indices[1:]:
        raw_items[i]["isCorrect"] = False
    raw_items[keep]["isCorrect"] = True
    return raw_items


def _looks_like_url(value: str) -> bool:
    return bool(re.match(r"^https?://", value, flags=re.IGNORECASE))


def _clamp_text(value: str, *, min_len: int, max_len: int) -> str:
    if not isinstance(value, str):
        return ""
    s = value.strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    if len(s) < min_len:
        return ""
    return s


def _drop_none_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _drop_none_fields(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none_fields(v) for v in obj]
    return obj


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
