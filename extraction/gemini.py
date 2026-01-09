"""Gemini processing utilities for PDF extraction.

This module extracts quiz-ready MCQs directly in the backend's CreateQuizDto JSON shape.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
import io

import google.generativeai as genai
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract

if os.name == 'nt':  # Windows
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Configure Poppler path for pdf2image
    poppler_paths = [
        r'C:\poppler\Release-25.12.0-0\poppler-25.12.0\Library\bin',
        r'C:\Program Files\poppler\bin',
        r'C:\poppler\bin',
    ]
    POPPLER_PATH = None
    for path in poppler_paths:
        if os.path.exists(path):
            POPPLER_PATH = path
            break
    

from extraction.pdf_utils import split_pdf_into_chunks, split_pdf_into_single_pages, get_page_count


DEFAULT_MODEL_NAME = "gemini-3-flash-preview"


def extract_ocr_text_from_all_pages(
    pdf_bytes: bytes,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[List[int], List[str]]:
    """Extract OCR text from all PDF pages without filtering.
    
    Args:
        pdf_bytes: Original PDF bytes
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (page_numbers, ocr_texts) for all pages
    """
    if progress_callback:
        progress_callback("Splitting PDF into pages for OCR extraction...", 0.0)
    
    # Split into single pages
    single_page_pdfs = split_pdf_into_single_pages(pdf_bytes)
    total_pages = len(single_page_pdfs)
    
    if progress_callback:
        progress_callback(f"Extracting text from {total_pages} pages...", 0.05)
    
    page_numbers = []
    ocr_texts = []
    
    for idx, page_pdf in enumerate(single_page_pdfs):
        page_num = idx + 1
        
        if progress_callback and idx % 5 == 0:  # Update every 5 pages
            progress = 0.05 + (0.90 * (idx / total_pages))
            progress_callback(f"Extracting text from page {page_num}/{total_pages}...", progress)
        
        # Extract OCR text from page
        try:
            # Convert PDF page to image
            convert_kwargs = {'dpi': 200}
            if os.name == 'nt' and POPPLER_PATH:
                convert_kwargs['poppler_path'] = POPPLER_PATH
            
            images = convert_from_bytes(page_pdf, **convert_kwargs)
            if images:
                # OCR the page
                text = pytesseract.image_to_string(images[0])
                print(f"[OCR DEBUG] Page {page_num}: Extracted {len(text)} chars")
            else:
                text = ""
                print(f"[OCR DEBUG] Page {page_num}: No image extracted")
        except Exception as e:
            text = ""
            print(f"[OCR DEBUG] Page {page_num}: Error - {str(e)}")
        
        # Include all pages regardless of content
        page_numbers.append(page_num)
        ocr_texts.append(text)
    
    if progress_callback:
        progress_callback(
            f"OCR extraction complete: {total_pages} pages processed",
            1.0
        )
    
    return page_numbers, ocr_texts


def process_pages_with_ai(
    ocr_texts: List[str],
    page_numbers: List[int],
    filename: str,
    cfg: GeminiConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Process OCR text from pages with Gemini AI.
    
    Args:
        ocr_texts: List of OCR text from pages
        page_numbers: Corresponding page numbers
        filename: Original PDF filename
        cfg: Gemini configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with title, description, and pages array containing questions per page
    """
    if not cfg.api_key:
        raise ValueError("Missing GEMINI_API_KEY")
    
    if not ocr_texts:
        return {
            "title": f"Quiz from {filename}",
            "description": "No pages to process",
            "pages": [],
            "totalPages": 0,
        }
    
    if progress_callback:
        progress_callback(
            f"Processing {len(ocr_texts)} pages with AI...",
            0.0
        )

    # Configure Gemini
    genai.configure(api_key=cfg.api_key)
    generation_config: Dict[str, Any] = {
        "temperature": float(cfg.temperature),
        "max_output_tokens": int(cfg.max_output_tokens),
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(model_name=cfg.model_name, generation_config=generation_config)

    # Process pages with rate limiting (5 RPM for free tier)
    rpm_limit = 5
    seconds_per_request = 60.0 / rpm_limit  # 12 seconds between requests
    
    pages_with_questions: List[Dict[str, Any]] = []
    last_request_time = 0.0

    for idx, (page_text, page_num) in enumerate(zip(ocr_texts, page_numbers)):
        # Rate limiting
        if idx > 0:
            elapsed = time.time() - last_request_time
            if elapsed < seconds_per_request:
                wait_time = seconds_per_request - elapsed
                if progress_callback:
                    progress_callback(
                        f"Rate limiting: waiting {wait_time:.1f}s (5 RPM limit)...",
                        idx / len(ocr_texts),
                    )
                time.sleep(wait_time)

        if progress_callback:
            progress_callback(
                f"AI processing page {page_num} ({idx + 1}/{len(ocr_texts)})...",
                idx / len(ocr_texts),
            )

        last_request_time = time.time()
        
        # Send OCR text as plain text prompt
        prompt = f"{MCQ_CREATE_QUIZ_PROMPT}\n\nPage {page_num} content:\n\n{page_text}"
        
        try:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None) or str(resp)

            try:
                dto = _parse_and_normalize_create_quiz_dto(text)
            except ValueError:
                # Retry with repair prompt
                repair_prompt = f"{REPAIR_TO_JSON_PROMPT_TEMPLATE}\n\n{text}\n"
                resp2 = model.generate_content(repair_prompt)
                text2 = getattr(resp2, "text", None) or str(resp2)
                dto = _parse_and_normalize_create_quiz_dto(text2)

            # Extract questions and associate with page number
            questions = dto.get("questions", [])
            if isinstance(questions, list) and questions:
                pages_with_questions.append({
                    "pageNumber": page_num,
                    "questions": questions,
                    "questionCount": len(questions),
                })
            else:
                # No questions found on this page
                pages_with_questions.append({
                    "pageNumber": page_num,
                    "questions": [],
                    "questionCount": 0,
                })
        except Exception as e:
            # Log error but continue
            pages_with_questions.append({
                "pageNumber": page_num,
                "questions": [],
                "questionCount": 0,
                "error": str(e)[:200],
            })

    if progress_callback:
        progress_callback(f"Completed: {len(ocr_texts)} pages processed", 1.0)

    # Build final result
    title = f"Quiz from {filename}"
    if len(title) > 200:
        title = title[:197] + "..."

    result = {
        "title": title,
        "description": f"Extracted from {len(ocr_texts)} pages",
        "pages": pages_with_questions,
        "totalPages": len(ocr_texts),
    }

    return result


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
    """Extract text via OCR, process in chunks, and merge results into a flat question list."""

    if not cfg.api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    # Validate page count
    total_pages = get_page_count(pdf_bytes)
    if total_pages > cfg.max_pages:
        raise ValueError(f"PDF has {total_pages} pages (max allowed: {cfg.max_pages})")

    if progress_callback:
        progress_callback(f"Starting OCR extraction on {total_pages} pages...", 0.0)

    # STEP 1: Extract OCR text from all pages
    page_numbers, ocr_texts = extract_ocr_text_from_all_pages(
        pdf_bytes,
        progress_callback=progress_callback
    )
    
    if not ocr_texts:
        return {
            "title": f"Quiz from {filename}",
            "description": f"No pages extracted from {total_pages}-page PDF",
            "questions": [],
        }

    if progress_callback:
        progress_callback(f"Processing {len(ocr_texts)} pages with AI...", 0.5)

    # STEP 2: Configure Gemini
    genai.configure(api_key=cfg.api_key)
    generation_config: Dict[str, Any] = {
        "temperature": float(cfg.temperature),
        "max_output_tokens": int(cfg.max_output_tokens),
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(model_name=cfg.model_name, generation_config=generation_config)

    # STEP 3: Process pages in chunks with rate limiting (5 RPM)
    rpm_limit = 5
    seconds_per_request = 60.0 / rpm_limit
    
    all_questions: List[Dict[str, Any]] = []
    last_request_time = 0.0
    
    # Group pages into chunks
    chunk_size = cfg.pages_per_chunk
    num_chunks = (len(ocr_texts) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        # Rate limiting
        if chunk_idx > 0:
            elapsed = time.time() - last_request_time
            if elapsed < seconds_per_request:
                wait_time = seconds_per_request - elapsed
                if progress_callback:
                    progress_callback(
                        f"Rate limiting: waiting {wait_time:.1f}s (5 RPM limit)...",
                        0.5 + (0.4 * (chunk_idx / num_chunks)),
                    )
                time.sleep(wait_time)

        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(ocr_texts))
        chunk_texts = ocr_texts[chunk_start:chunk_end]
        chunk_page_nums = page_numbers[chunk_start:chunk_end]

        if progress_callback:
            progress_callback(
                f"AI processing chunk {chunk_idx + 1}/{num_chunks} (pages {chunk_start + 1}-{chunk_end})...",
                0.5 + (0.4 * (chunk_idx / num_chunks)),
            )

        last_request_time = time.time()
        
        # Combine chunk texts into one prompt
        combined_text = "\n\n".join([
            f"Page {page_num}:\n{text}" 
            for page_num, text in zip(chunk_page_nums, chunk_texts)
        ])
        
        prompt = f"{MCQ_CREATE_QUIZ_PROMPT}\n\nContent:\n\n{combined_text}"
        
        try:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None) or str(resp)

            try:
                dto = _parse_and_normalize_create_quiz_dto(text)
            except ValueError:
                # Retry with repair prompt
                repair_prompt = f"{REPAIR_TO_JSON_PROMPT_TEMPLATE}\n\n{text}\n"
                resp2 = model.generate_content(repair_prompt)
                text2 = getattr(resp2, "text", None) or str(resp2)
                dto = _parse_and_normalize_create_quiz_dto(text2)

            # Extract questions and merge into flat list
            questions = dto.get("questions", [])
            if isinstance(questions, list):
                all_questions.extend(questions)
        except Exception as e:
            # Log error but continue
            print(f"[ERROR] Chunk {chunk_idx + 1} failed: {str(e)[:200]}")

    if progress_callback:
        progress_callback(f"Completed: {len(all_questions)} questions extracted", 1.0)

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
    """Extract questions page-by-page using OCR text extraction.
    
    Workflow:
    1. OCR all pages to extract text
    2. Process pages with Gemini AI using text prompts (15 RPM, cheap)
    
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

    # Get total page count
    total_pages = get_page_count(pdf_bytes)
    if total_pages > cfg.max_pages:
        raise ValueError(f"PDF has {total_pages} pages (max allowed: {cfg.max_pages})")

    if progress_callback:
        progress_callback(f"Starting OCR extraction on {total_pages} pages...", 0.0)

    # STEP 1: OCR extract text from all pages
    page_numbers, ocr_texts = extract_ocr_text_from_all_pages(
        pdf_bytes,
        progress_callback=progress_callback
    )
    
    if not ocr_texts:
        # No pages extracted
        if progress_callback:
            progress_callback("No pages extracted", 1.0)
        return {
            "title": f"Quiz from {filename}",
            "description": f"No pages extracted from {total_pages}-page PDF",
            "pages": [],
            "totalPages": total_pages,
        }
    
    # STEP 2: Process all pages with Gemini AI using text prompts
    result = process_pages_with_ai(
        ocr_texts,
        page_numbers,
        filename,
        cfg,
        progress_callback
    )
    
    result["totalPages"] = total_pages
    return result


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
