"""Gemini processing utilities for PDF extraction.

This module extracts quiz-ready MCQs directly in the backend's CreateQuizDto JSON shape.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
import io

import google.generativeai as genai
from pypdf import PdfReader

from extraction.pdf_utils import split_pdf_into_chunks, split_pdf_into_single_pages, get_page_count


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
- There are times that there is no answer key sheet but the answer is highlighted or marked in the text. Use that to determine the correct answer.
- If theres no answer key at all provide an answer based on the question context.
Output strictly as JSON.
""".strip()


REPAIR_TO_JSON_PROMPT_TEMPLATE = """
You returned output that was not valid JSON.

Task:
Convert the text below into ONE valid JSON object that matches this schema (CreateQuizDto):

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

Rules:
- Output ONLY valid JSON. No markdown, no commentary.
- Ensure: 2-6 choices per question and EXACTLY ONE correct choice.

Text to convert:
""".strip()


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = 0.2
    max_output_tokens: int = 8192
    pages_per_chunk: int = 20
    max_pages: int = 200


def extract_create_quiz_dto_from_pdf_bytes(
    *,
    pdf_bytes: bytes,
    filename: str,
    cfg: GeminiConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Split PDF into chunks, process in batches, and merge results."""

    if not cfg.api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    # Validate page count
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    if total_pages > cfg.max_pages:
        raise ValueError(f"PDF has {total_pages} pages (max allowed: {cfg.max_pages})")

    if progress_callback:
        progress_callback(f"Splitting {total_pages} pages into chunks...", 0.05)

    # Split into chunks of pages_per_chunk
    chunk_pdfs = split_pdf_into_chunks(pdf_bytes, pages_per_chunk=cfg.pages_per_chunk)
    if not chunk_pdfs:
        raise ValueError("Failed to split PDF into chunks")

    if progress_callback:
        progress_callback(f"Created {len(chunk_pdfs)} chunks ({cfg.pages_per_chunk} pages each)", 0.1)

    genai.configure(api_key=cfg.api_key)
    generation_config: Dict[str, Any] = {
        "temperature": float(cfg.temperature),
        "max_output_tokens": int(cfg.max_output_tokens),
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(model_name=cfg.model_name, generation_config=generation_config)

    # Process chunks in batches (Gemini file upload limit is 10)
    all_questions: List[Dict[str, Any]] = []
    batch_size = 10
    num_batches = (len(chunk_pdfs) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(chunk_pdfs))
        batch_chunks = chunk_pdfs[batch_start:batch_end]

        if progress_callback:
            progress_callback(
                f"Processing batch {batch_idx + 1}/{num_batches} (chunks {batch_start + 1}-{batch_end})...",
                0.1 + (0.8 * (batch_idx / num_batches)),
            )

        batch_questions = _process_chunk_batch(batch_chunks, model, filename, batch_start)
        all_questions.extend(batch_questions)

    if progress_callback:
        progress_callback(f"Merging {len(all_questions)} questions...", 0.95)

    # Build final CreateQuizDto
    title = f"Quiz from {filename}"
    if len(title) > 200:
        title = title[:197] + "..."

    result = {
        "title": title,
        "description": f"Extracted from {total_pages}-page PDF",
        "questions": all_questions,
    }

    return result


def extract_create_quiz_dto_by_page(
    *,
    pdf_bytes: bytes,
    filename: str,
    cfg: GeminiConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Extract questions page-by-page with rate limiting.
    
    Splits PDF into single-page PDFs, processes in batches of 10 (Gemini limit),
    with 5 RPM rate limiting. Returns results grouped by page.
    
    Args:
        pdf_bytes: Original PDF bytes
        filename: PDF filename
        cfg: Gemini configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with title, description, and pages array containing questions per page
    """
    if not cfg.api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    # Validate page count
    total_pages = get_page_count(pdf_bytes)
    if total_pages > cfg.max_pages:
        raise ValueError(f"PDF has {total_pages} pages (max allowed: {cfg.max_pages})")

    if progress_callback:
        progress_callback(f"Splitting {total_pages} pages...", 0.05)

    # Split into single-page PDFs
    single_page_pdfs = split_pdf_into_single_pages(pdf_bytes)
    if not single_page_pdfs:
        raise ValueError("Failed to split PDF into pages")

    if progress_callback:
        progress_callback(f"Created {len(single_page_pdfs)} single-page PDFs", 0.1)

    genai.configure(api_key=cfg.api_key)
    generation_config: Dict[str, Any] = {
        "temperature": float(cfg.temperature),
        "max_output_tokens": int(cfg.max_output_tokens),
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(model_name=cfg.model_name, generation_config=generation_config)

    # Process pages in batches of 10 (Gemini upload limit) with rate limiting (5 RPM)
    batch_size = 10
    rpm_limit = 5
    seconds_per_request = 60.0 / rpm_limit  # 12 seconds between requests for 5 RPM
    
    num_batches = (len(single_page_pdfs) + batch_size - 1) // batch_size
    pages_with_questions: List[Dict[str, Any]] = []
    last_request_time = 0.0

    for batch_idx in range(num_batches):
        # Rate limiting: ensure we don't exceed 5 RPM
        if batch_idx > 0:
            elapsed = time.time() - last_request_time
            if elapsed < seconds_per_request:
                wait_time = seconds_per_request - elapsed
                if progress_callback:
                    progress_callback(
                        f"Rate limiting: waiting {wait_time:.1f}s (5 RPM limit)...",
                        0.1 + (0.8 * (batch_idx / num_batches)),
                    )
                time.sleep(wait_time)
        
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(single_page_pdfs))
        batch_pages = single_page_pdfs[batch_start:batch_end]

        if progress_callback:
            progress_callback(
                f"Processing pages {batch_start + 1}-{batch_end}/{total_pages}...",
                0.1 + (0.8 * (batch_idx / num_batches)),
            )

        last_request_time = time.time()
        batch_results = _process_pages_batch(
            batch_pages, 
            model, 
            filename, 
            batch_start
        )
        pages_with_questions.extend(batch_results)

    if progress_callback:
        progress_callback(f"Completed processing {total_pages} pages", 0.95)

    # Build final result with page-grouped questions
    title = f"Quiz from {filename}"
    if len(title) > 200:
        title = title[:197] + "..."

    result = {
        "title": title,
        "description": f"Extracted from {total_pages}-page PDF (page-by-page)",
        "pages": pages_with_questions,  # Array of {pageNumber, questions[]}
        "totalPages": total_pages,
    }

    return result


def _process_pages_batch(
    page_pdfs: List[bytes],
    model: Any,
    base_filename: str,
    batch_offset: int,
) -> List[Dict[str, Any]]:
    """Process a batch of single-page PDFs and return questions with page numbers.
    
    Args:
        page_pdfs: List of single-page PDF bytes
        model: Gemini model instance
        base_filename: Original filename
        batch_offset: Starting page index for this batch
        
    Returns:
        List of dicts with pageNumber and questions array
    """
    uploaded_files = []
    tmp_paths: List[str] = []
    results: List[Dict[str, Any]] = []

    try:
        # Upload all pages in this batch
        for i, page_bytes in enumerate(page_pdfs):
            page_num = batch_offset + i + 1  # 1-indexed page numbers
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(page_bytes)
                tmp.flush()
                tmp_paths.append(tmp.name)
                uploaded = genai.upload_file(
                    path=tmp.name, 
                    display_name=f"{base_filename}_page{page_num}"
                )
                uploaded_files.append((uploaded, page_num))

        # Process each page individually to maintain page-to-questions mapping
        for uploaded_file, page_num in uploaded_files:
            prompt_parts = [uploaded_file, MCQ_CREATE_QUIZ_PROMPT]
            resp = model.generate_content(prompt_parts)
            text = getattr(resp, "text", None) or str(resp)

            try:
                dto = _parse_and_normalize_create_quiz_dto(text)
            except ValueError:
                # Retry with repair prompt
                repair_prompt = f"{REPAIR_TO_JSON_PROMPT_TEMPLATE}\n\n{text}\n"
                resp2 = model.generate_content([repair_prompt])
                text2 = getattr(resp2, "text", None) or str(resp2)
                dto = _parse_and_normalize_create_quiz_dto(text2)

            # Extract questions and associate with page number
            questions = dto.get("questions", [])
            if isinstance(questions, list) and questions:
                results.append({
                    "pageNumber": page_num,
                    "questions": questions,
                    "questionCount": len(questions),
                })
            else:
                # No questions found on this page
                results.append({
                    "pageNumber": page_num,
                    "questions": [],
                    "questionCount": 0,
                })

    finally:
        # Cleanup temp files
        for tmp_path in tmp_paths:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return results


def _process_chunk_batch(
    chunk_pdfs: List[bytes],
    model: Any,
    base_filename: str,
    batch_offset: int,
) -> List[Dict[str, Any]]:
    """Upload chunk PDFs to Gemini, extract questions, return normalized list."""
    uploaded_files = []
    tmp_paths: List[str] = []

    try:
        # Upload all chunks in this batch
        for i, chunk_bytes in enumerate(chunk_pdfs):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(chunk_bytes)
                tmp.flush()
                tmp_paths.append(tmp.name)
                uploaded = genai.upload_file(path=tmp.name, display_name=f"{base_filename}_chunk{batch_offset + i}")
                uploaded_files.append(uploaded)

        # Send all uploaded chunks + prompt in one request
        prompt_parts = uploaded_files + [MCQ_CREATE_QUIZ_PROMPT]
        resp = model.generate_content(prompt_parts)
        text = getattr(resp, "text", None) or str(resp)

        try:
            dto = _parse_and_normalize_create_quiz_dto(text)
        except ValueError:
            # Retry with repair prompt
            repair_prompt = f"{REPAIR_TO_JSON_PROMPT_TEMPLATE}\n\n{text}\n"
            resp2 = model.generate_content([repair_prompt])
            text2 = getattr(resp2, "text", None) or str(resp2)
            dto = _parse_and_normalize_create_quiz_dto(text2)

        # Extract just the questions array
        questions = dto.get("questions", [])
        if not isinstance(questions, list):
            return []
        return questions

    finally:
        # Cleanup temp files
        for tmp_path in tmp_paths:
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

    preview = re.sub(r"\s+", " ", s)
    if len(preview) > 800:
        preview = preview[:800] + "..."
    raise ValueError(f"Failed to parse JSON from model response. Preview: {preview}")
