"""CreateQuizDto validation and normalization for backend API."""

from typing import Any, Dict, List, Optional, Tuple


def clamp_text(value: Any, *, min_len: int, max_len: int) -> Optional[str]:
    """Clamp text to min/max length constraints."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    if len(s) < min_len:
        return None
    return s


def normalize_choices(choices: Any, errors: List[str], *, q_index: int) -> Optional[List[Dict]]:
    """Normalize and validate choices array for a question."""
    if not isinstance(choices, list):
        errors.append(f"Question {q_index}: choices must be a list")
        return None

    items: List[Dict] = []
    for c in choices:
        if not isinstance(c, dict):
            continue
        text = clamp_text(c.get('text'), min_len=1, max_len=500)
        if not text:
            continue
        items.append({'text': text, 'isCorrect': bool(c.get('isCorrect'))})

    if len(items) < 2:
        errors.append(f"Question {q_index}: must have at least 2 choices")
        return None
    if len(items) > 6:
        # Trim extra choices to satisfy backend rule
        items = items[:6]

    correct_indexes = [i for i, c in enumerate(items) if c.get('isCorrect') is True]
    if len(correct_indexes) == 1:
        return items
    if len(correct_indexes) == 0:
        # Default: mark first option correct (user can fix in editor)
        items[0]['isCorrect'] = True
        errors.append(f"Question {q_index}: no correct choice marked; defaulted first choice to correct")
        return items

    keep = correct_indexes[0]
    for i in correct_indexes[1:]:
        items[i]['isCorrect'] = False
    items[keep]['isCorrect'] = True
    errors.append(f"Question {q_index}: multiple correct choices; kept first as correct")
    return items


def validate_and_normalize_create_quiz_dto(payload: Any) -> Tuple[Optional[Dict], List[str]]:
    """Validate and normalize a CreateQuizDto-like dict before POSTing to the backend.

    Returns (normalized_payload, errors). If normalized_payload is None, it's not safe to POST.
    """

    errors: List[str] = []
    if not isinstance(payload, dict):
        return None, ["Payload must be a JSON object"]

    title = clamp_text(payload.get('title'), min_len=3, max_len=200)
    if not title:
        errors.append("title is required (3-200 chars)")

    description_raw = payload.get('description', None)
    description: Optional[str]
    if description_raw is None:
        description = None
    else:
        description = clamp_text(description_raw, min_len=0, max_len=2000)

    questions_raw = payload.get('questions')
    if not isinstance(questions_raw, list) or len(questions_raw) < 1:
        errors.append("questions is required (at least 1 question)")
        questions_raw = []

    normalized_questions: List[Dict] = []
    for i, q in enumerate(questions_raw, start=1):
        if not isinstance(q, dict):
            errors.append(f"Question {i}: must be an object")
            continue

        q_text = clamp_text(q.get('text'), min_len=5, max_len=100)
        if not q_text:
            errors.append(f"Question {i}: text is required (5-100 chars)")
            continue

        explanation_raw = q.get('explanation', None)
        explanation: Optional[str]
        if explanation_raw is None:
            explanation = None
        else:
            explanation = clamp_text(explanation_raw, min_len=0, max_len=300)

        image_url = q.get('imageUrl', None)
        if isinstance(image_url, str):
            image_url = image_url.strip()
            if not (image_url.startswith('http://') or image_url.startswith('https://')):
                errors.append(f"Question {i}: imageUrl must be a valid URL")
                image_url = None
        else:
            image_url = None

        choices = normalize_choices(q.get('choices'), errors, q_index=i)
        if choices is None:
            continue

        out_q = {
            'text': q_text,
            'explanation': explanation,
            'imageUrl': image_url,
            'choices': choices,
        }
        # drop None fields
        out_q = {k: v for k, v in out_q.items() if v is not None}
        normalized_questions.append(out_q)

    if not normalized_questions:
        errors.append("No valid questions found")

    if errors and (not title or not normalized_questions):
        return None, errors

    normalized = {
        'title': title,
        'description': description,
        'questions': normalized_questions,
    }
    normalized = {k: v for k, v in normalized.items() if v is not None}
    return normalized, errors
