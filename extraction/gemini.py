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
from pdf2image import convert_from_bytes
import pytesseract

if os.name == 'nt':  # Windows
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    

from extraction.pdf_utils import split_pdf_into_chunks, split_pdf_into_single_pages, get_page_count


DEFAULT_MODEL_NAME = "gemini-3-flash-preview"


def detect_questions_in_page(page_bytes: bytes) -> Dict[str, Any]:
    """Use OCR to detect if a page contains questions or answer keys.
    
    Args:
        page_bytes: Single-page PDF bytes
        
    Returns:
        Dictionary with:
        - has_questions: bool
        - has_answer_key: bool
        - question_count_estimate: int
        - patterns_found: list of detected patterns
    """
    try:
        # Convert PDF page to image
        images = convert_from_bytes(page_bytes, dpi=200)
        if not images:
            return {"has_questions": False, "has_answer_key": False, "question_count_estimate": 0, "patterns_found": []}
        
        # OCR the first (and only) page
        text = pytesseract.image_to_string(images[0])
        text_lower = text.lower()
        
        patterns_found = []
        has_questions = False
        has_answer_key = False
        question_count = 0
        
        # Pattern 1: Multiple choice options (A), B), C), D)
        mc_pattern = re.compile(r'[A-F]\s*[.):]', re.IGNORECASE)
        mc_matches = mc_pattern.findall(text)
        if len(mc_matches) >= 3:  # At least 3 options suggest a question
            patterns_found.append("multiple_choice_letters")
            has_questions = True
            # Rough estimate: divide by 4 (assuming 4 options per question)
            question_count += len(mc_matches) // 4
        
        # Pattern 2: Numbered questions (1., 2., 3. or 1), 2), 3))
        numbered_pattern = re.compile(r'^\s*\d{1,3}\s*[.):]', re.MULTILINE)
        numbered_matches = numbered_pattern.findall(text)
        if len(numbered_matches) >= 2:
            patterns_found.append("numbered_questions")
            has_questions = True
            question_count = max(question_count, len(numbered_matches))
        
        # Pattern 3: Answer key indicators
        answer_key_keywords = [
            'answer key', 'answer sheet', 'correct answers', 'solutions',
            'answer guide', 'key:', 'answers:'
        ]
        for keyword in answer_key_keywords:
            if keyword in text_lower:
                patterns_found.append(f"answer_key_keyword: {keyword}")
                has_answer_key = True
                break
        
        # Pattern 4: Grid-style answer key (e.g., "1. A  2. B  3. C")
        answer_grid_pattern = re.compile(r'\d+\s*[.)]\s*[A-F]', re.IGNORECASE)
        answer_grid_matches = answer_grid_pattern.findall(text)
        if len(answer_grid_matches) >= 5:  # 5+ answers in grid format
            patterns_found.append("answer_grid")
            has_answer_key = True
        
        # Pattern 5: Question stems ("What is", "Which of", "Select", etc.)
        question_stems = [
            'what is', 'what are', 'which of', 'select the', 'choose the',
            'identify', 'find the', 'calculate', 'determine'
        ]
        stem_count = sum(1 for stem in question_stems if stem in text_lower)
        if stem_count >= 2:
            patterns_found.append("question_stems")
            has_questions = True
        
        return {
            "has_questions": has_questions,
            "has_answer_key": has_answer_key,
            "question_count_estimate": question_count,
            "patterns_found": patterns_found,
        }
        
    except Exception as e:
        # If OCR fails, assume page might have content (fail-safe)
        return {
            "has_questions": True,  # Fail-safe: include page if OCR fails
            "has_answer_key": False,
            "question_count_estimate": 0,
            "patterns_found": [f"ocr_error: {str(e)}"],
        }


def filter_pages_with_questions(
    pdf_bytes: bytes,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[List[int], List[bytes]]:
    """Filter PDF pages to find those with questions or answer keys using OCR.
    
    Args:
        pdf_bytes: Original PDF bytes
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (page_numbers, page_pdfs) for pages that contain questions/answers
    """
    if progress_callback:
        progress_callback("Splitting PDF into pages for OCR filtering...", 0.0)
    
    # Split into single pages
    single_page_pdfs = split_pdf_into_single_pages(pdf_bytes)
    total_pages = len(single_page_pdfs)
    
    if progress_callback:
        progress_callback(f"OCR scanning {total_pages} pages for questions...", 0.05)
    
    filtered_page_numbers = []
    filtered_page_pdfs = []
    
    for idx, page_pdf in enumerate(single_page_pdfs):
        page_num = idx + 1
        
        if progress_callback and idx % 5 == 0:  # Update every 5 pages
            progress = 0.05 + (0.25 * (idx / total_pages))
            progress_callback(f"OCR scanning page {page_num}/{total_pages}...", progress)
        
        # Use OCR to detect questions
        detection = detect_questions_in_page(page_pdf)
        
        # Include page if it has questions or answer keys
        if detection["has_questions"] or detection["has_answer_key"]:
            filtered_page_numbers.append(page_num)
            filtered_page_pdfs.append(page_pdf)
    
    if progress_callback:
        progress_callback(
            f"OCR filtering complete: {len(filtered_page_numbers)}/{total_pages} pages contain questions",
            1.0
        )
    
    return filtered_page_numbers, filtered_page_pdfs


def process_filtered_pages_with_ai(
    filtered_page_pdfs: List[bytes],
    filtered_page_numbers: List[int],
    filename: str,
    cfg: GeminiConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Process pre-filtered pages with Gemini AI.
    
    Args:
        filtered_page_pdfs: List of filtered page PDFs (bytes)
        filtered_page_numbers: Corresponding page numbers
        filename: Original PDF filename
        cfg: Gemini configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with title, description, and pages array containing questions per page
    """
    if not cfg.api_key:
        raise ValueError("Missing GEMINI_API_KEY")
    
    if not filtered_page_pdfs:
        return {
            "title": f"Quiz from {filename}",
            "description": "No pages to process",
            "pages": [],
            "filteredPages": 0,
        }
    
    if progress_callback:
        progress_callback(
            f"Processing {len(filtered_page_pdfs)} pages with AI...",
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

    # Process pages in batches of 10 (Gemini upload limit) with rate limiting (5 RPM)
    batch_size = 10
    rpm_limit = 5
    seconds_per_request = 60.0 / rpm_limit  # 12 seconds between requests for 5 RPM
    
    num_batches = (len(filtered_page_pdfs) + batch_size - 1) // batch_size
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
                        (batch_idx / num_batches),
                    )
                time.sleep(wait_time)
        
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(filtered_page_pdfs))
        batch_pages = filtered_page_pdfs[batch_start:batch_end]
        batch_page_numbers = filtered_page_numbers[batch_start:batch_end]

        if progress_callback:
            progress_callback(
                f"AI processing pages {batch_start + 1}-{batch_end}/{len(filtered_page_pdfs)}...",
                (batch_idx / num_batches),
            )

        last_request_time = time.time()
        batch_results = _process_pages_batch_with_page_numbers(
            batch_pages, 
            batch_page_numbers,
            model, 
            filename
        )
        pages_with_questions.extend(batch_results)

    if progress_callback:
        progress_callback(f"Completed: {len(filtered_page_pdfs)} pages processed", 1.0)

    # Build final result
    title = f"Quiz from {filename}"
    if len(title) > 200:
        title = title[:197] + "..."

    result = {
        "title": title,
        "description": f"Extracted from {len(filtered_page_pdfs)} pages with questions",
        "pages": pages_with_questions,
        "filteredPages": len(filtered_page_pdfs),
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
    """Extract questions page-by-page with OCR pre-filtering and rate limiting.
    
    Workflow:
    1. OCR all pages to detect questions/answer keys (fast, cheap)
    2. Filter to only relevant pages
    3. Process filtered pages with Gemini AI (slow, expensive)
    
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
        progress_callback(f"Starting OCR pre-filtering on {total_pages} pages...", 0.0)

    # STEP 1: OCR filter to find pages with questions (0.0 - 0.3 progress)
    filtered_page_numbers, filtered_page_pdfs = filter_pages_with_questions(
        pdf_bytes,
        progress_callback=progress_callback
    )
    
    if not filtered_page_pdfs:
        # No pages with questions found
        if progress_callback:
            progress_callback("No pages with questions detected", 1.0)
        return {
            "title": f"Quiz from {filename}",
            "description": f"No questions found in {total_pages}-page PDF",
            "pages": [],
            "totalPages": total_pages,
            "filteredPages": 0,
        }
    
    if progress_callback:
        progress_callback(
            f"Processing {len(filtered_page_pdfs)} pages with AI (filtered from {total_pages})...",
            0.35
        )

    # STEP 2: Process filtered pages with Gemini AI (0.35 - 0.95 progress)
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
    
    num_batches = (len(filtered_page_pdfs) + batch_size - 1) // batch_size
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
                        0.35 + (0.6 * (batch_idx / num_batches)),
                    )
                time.sleep(wait_time)
        
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(filtered_page_pdfs))
        batch_pages = filtered_page_pdfs[batch_start:batch_end]
        batch_page_numbers = filtered_page_numbers[batch_start:batch_end]

        if progress_callback:
            progress_callback(
                f"AI processing filtered pages {batch_start + 1}-{batch_end}/{len(filtered_page_pdfs)}...",
                0.35 + (0.6 * (batch_idx / num_batches)),
            )

        last_request_time = time.time()
        batch_results = _process_pages_batch_with_page_numbers(
            batch_pages, 
            batch_page_numbers,
            model, 
            filename
        )
        pages_with_questions.extend(batch_results)

    if progress_callback:
        progress_callback(f"Completed: {len(filtered_page_pdfs)} pages processed", 0.95)

    # Build final result with page-grouped questions
    title = f"Quiz from {filename}"
    if len(title) > 200:
        title = title[:197] + "..."

    result = {
        "title": title,
        "description": f"Extracted from {total_pages}-page PDF ({len(filtered_page_pdfs)} pages with questions)",
        "pages": pages_with_questions,  # Array of {pageNumber, questions[]}
        "totalPages": total_pages,
        "filteredPages": len(filtered_page_pdfs),
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
    page_numbers = [batch_offset + i + 1 for i in range(len(page_pdfs))]
    return _process_pages_batch_with_page_numbers(page_pdfs, page_numbers, model, base_filename)


def _process_pages_batch_with_page_numbers(
    page_pdfs: List[bytes],
    page_numbers: List[int],
    model: Any,
    base_filename: str,
) -> List[Dict[str, Any]]:
    """Process a batch of single-page PDFs with explicit page numbers.
    
    Args:
        page_pdfs: List of single-page PDF bytes
        page_numbers: Corresponding page numbers for each PDF
        model: Gemini model instance
        base_filename: Original filename
        
    Returns:
        List of dicts with pageNumber and questions array
    """
    uploaded_files = []
    tmp_paths: List[str] = []
    results: List[Dict[str, Any]] = []

    try:
        # Upload all pages in this batch
        for i, page_bytes in enumerate(page_pdfs):
            page_num = page_numbers[i]
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
