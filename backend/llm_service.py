import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

# from emergentintegrations.llm.chat import (
#     LlmChat,
#     UserMessage,
#     FileContentWithMimeType,
# )

logger = logging.getLogger(__name__)


def _api_key() -> str:
    return os.environ.get("EMERGENT_LLM_KEY", "")


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _mime_for(filename: str) -> Optional[str]:
    ext = Path(filename).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext)


async def extract_text_from_file(file_path: str, filename: str) -> str:
    """Use Gemini to extract text from PDF/image/text files."""
    mime = _mime_for(filename)
    if not mime:
        raise ValueError(f"Unsupported file type: {filename}")

    # Plain text: read directly
    if mime == "text/plain":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # PDF or image: send to Gemini for extraction
    chat = LlmChat(
        api_key=_api_key(),
        session_id=f"extract-{filename}",
        system_message=(
            "You extract the full textual content from documents and images. "
            "Return the extracted text preserving headings, lists, and paragraph structure. "
            "Do NOT summarize, do NOT add commentary. Return only the extracted text."
        ),
    ).with_model("gemini", GEMINI_MODEL)

    file_attach = FileContentWithMimeType(file_path=file_path, mime_type=mime)
    prompt = "Extract all text content from the attached file. Preserve structure."
    response = await chat.send_message(
        UserMessage(text=prompt, file_contents=[file_attach])
    )
    return str(response).strip()


def _extract_json_objects(text: str) -> List[Dict]:
    """
    Bracket-match complete top-level JSON objects from an array body.
    Tolerates truncated final object, stray text, code fences.
    """
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    # Skip past the opening '['
    start = text.find("[")
    if start != -1:
        text = text[start + 1:]

    objects: List[Dict] = []
    i, n = 0, len(text)
    while i < n:
        # skip separators / whitespace
        while i < n and text[i] in ", \n\r\t":
            i += 1
        if i >= n or text[i] != "{":
            break
        obj_start = i
        depth = 0
        in_str = False
        esc = False
        closed = False
        while i < n:
            c = text[i]
            if esc:
                esc = False
            elif in_str:
                if c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        obj_str = text[obj_start:i + 1]
                        try:
                            objects.append(json.loads(obj_str))
                        except json.JSONDecodeError:
                            # Try lenient fixups: strip trailing commas
                            fixed = re.sub(r",(\s*[}\]])", r"\1", obj_str)
                            try:
                                objects.append(json.loads(fixed))
                            except json.JSONDecodeError:
                                pass
                        i += 1
                        closed = True
                        break
            i += 1
        if not closed:
            # Truncated final object — stop; keep what we've salvaged so far
            break
    return objects


def _parse_json_array(text: str) -> List[Dict]:
    """Robust JSON array parsing from LLM output — tolerant of truncation."""
    stripped = text.strip()
    body = re.sub(r"^```(?:json)?\s*", "", stripped)
    body = re.sub(r"\s*```$", "", body)
    start = body.find("[")
    end = body.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = body[start:end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        # Try trailing-comma fixup
        try:
            fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
            data = json.loads(fixed)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    # Fallback: salvage individual objects (handles truncation)
    return _extract_json_objects(text)


MAX_MCQS = 40


def compute_target_bounds(total_chars: int, requested: int) -> tuple[int, int]:
    """
    Returns (target_min, target_max).
    - requested == 0 -> AUTO: fully text-driven, roughly ~1 MCQ per ~80 characters,
      minimum 8, maximum MAX_MCQS. The min gives the model room; the max is a hard ceiling.
    - requested > 0  -> user's number is the MINIMUM target; extra concepts may push it higher,
      capped at min(MAX_MCQS, max(requested*2, requested+5)).
    """
    if requested <= 0:
        auto_max = max(8, min(MAX_MCQS, total_chars // 80))
        auto_min = max(5, auto_max - 5)
        return auto_min, auto_max
    upper = min(MAX_MCQS, max(requested + 5, requested * 2))
    return requested, upper


async def generate_mcqs(
    note_texts: List[Dict[str, str]], count: int, allow_outside_knowledge: bool = False
) -> List[Dict]:
    """
    note_texts: list of {"chapter": str, "content": str}
    count: user-requested target. 0 = fully auto (concept-driven).
    Returns list of MCQ dicts: {question, options: {A,B,C,D}, correct_option, explanation, chapter}
    """
    if not note_texts:
        return []

    source_block = "\n\n".join(
        f"=== CHAPTER: {n['chapter']} ===\n{n['content'][:20000]}"
        for n in note_texts
    )
    total_chars = sum(len(n["content"]) for n in note_texts)
    target_min, target_max = compute_target_bounds(total_chars, count)

    rule = (
        "You MAY use general knowledge only when strictly necessary."
        if allow_outside_knowledge
        else "You MUST use ONLY the provided notes as source. Do NOT introduce outside information. "
             "If a fact is not in the notes, do NOT invent it."
    )

    system_msg = (
        "You are a senior UPSC-style exam question setter. Generate high-quality UPSC-style "
        "CONCEPTUAL and ANALYTICAL multiple-choice questions (MCQs) strictly grounded in the "
        "provided study notes. "
        f"{rule} "
        "STYLE RULES (UPSC): "
        "- Prefer conceptual, analytical, and application-based questions over direct factual recall. "
        "- Use UPSC-style phrasings such as: 'Consider the following statements... Which of the "
        "statements given above is/are correct?', 'Which of the following best explains...', "
        "'Which of the following is/are correct regarding...', 'Assertion (A) and Reason (R)...', "
        "'Arrange the following in the correct order...', 'Match the following...'. "
        "- Test cause-and-effect, comparison, inference, correct/incorrect statement sets, "
        "sequencing, and application of the concepts in the notes. "
        "- AVOID simple 'What is the definition of X?' or 'Who invented Y?' style direct fact "
        "questions unless the fact can only be tested that way. Prefer wrapping such facts inside "
        "statement-analysis or best-explanation formats. "
        "- Distractors must be plausible and derived from the same context (not obviously wrong); "
        "prefer close/related concepts as wrong options. "
        "LANGUAGE RULE: Detect the language of the provided notes. If the notes are primarily in "
        "Hindi (Devanagari script), generate the ENTIRE MCQ — question stem, all 4 options, and "
        "the explanation — in Hindi (Devanagari). If the notes are primarily in English, generate "
        "everything in English. For mixed content, match the dominant language of each chapter. "
        "The option KEYS must always remain the Latin letters A, B, C, D, and the chapter name "
        "must be copied verbatim from the provided chapter header. "
        "COVERAGE RULE: Aim to cover every important concept, fact, definition, date, name, "
        "cause-effect, comparison and key point in the notes. Do NOT produce duplicate or "
        "near-duplicate questions about the same point — vary the angle (conceptual, analytical, "
        "application, comparison) even when re-touching a topic. "
        "Each question must have exactly 4 options labeled A, B, C, D with only one correct. "
        "Provide a short (1-2 sentence) explanation of WHY the correct answer is correct AND, "
        "where helpful, why the closest distractor is wrong, in the same language as the question. "
        "Return ONLY a valid JSON array. No prose, no markdown fences."
    )

    prompt = f"""Generate UPSC-style CONCEPTUAL and ANALYTICAL MCQs from the notes below.

QUANTITY POLICY:
- Produce at least {target_min} MCQs.
- If the notes contain more distinct important concepts, produce more MCQs to cover them — up to a maximum of {target_max}.
- The count is driven by the actual concept-density of the notes, not by a fixed number.
- Skip trivial filler; do NOT pad with duplicates just to hit a number.
- Each MCQ must test a DIFFERENT angle — a different fact, concept, cause-effect, comparison, application, sequencing, or statement-analysis derived from the notes.

STYLE (UPSC):
- Favor formats like "Consider the following statements ... Which of the statements given above is/are correct?", "Which of the following best explains ...", "Which of the following is/are correct regarding ...", assertion-reason, matching, sequencing, and best-inference questions.
- Avoid plain "What is X?" / "Who did Y?" direct recall style unless truly unavoidable — instead wrap the same fact inside an analytical or evaluative frame.
- Distractors must be plausible and drawn from the same subject context.

Return a JSON array where each element has this exact shape:
{{
  "question": "string (in the same language as the source chapter, UPSC-style phrasing where possible)",
  "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
  "correct_option": "A" | "B" | "C" | "D",
  "explanation": "short explanation string in the same language as the question; explain WHY the answer is correct and briefly why the closest distractor is not",
  "chapter": "chapter name this question came from"
}}

Rules:
- Between {target_min} and {target_max} items in the array (inclusive).
- No two questions may test the same point from the same angle.
- Do NOT include markdown code fences or any text outside the JSON array.
- Exactly 4 options; only one correct.
- Grounded strictly in the provided notes (no external facts).
- Match the source language: Hindi notes -> Hindi MCQ; English notes -> English MCQ.

NOTES:
{source_block}
"""

    chat = LlmChat(
        api_key=_api_key(),
        session_id=f"mcq-gen-{os.urandom(4).hex()}",
        system_message=system_msg,
    ).with_model("gemini", GEMINI_MODEL).with_params(max_tokens=16000)

    response = await chat.send_message(UserMessage(text=prompt))
    raw = str(response)

    items = _parse_json_array(raw)
    if not items:
        logger.error("Parser returned no items. Raw prefix: %s", raw[:600])
        raise ValueError("AI response was empty or unparseable. Please try again.")

    cleaned = []
    seen_questions = set()
    for it in items:
        try:
            if not isinstance(it, dict):
                continue
            opts = it.get("options")
            if not isinstance(opts, dict):
                continue
            q_text = str(it.get("question", "")).strip()
            if not q_text:
                continue
            # Options must have all four keys with non-empty text
            option_map = {}
            skip = False
            for L in "ABCD":
                v = opts.get(L) or opts.get(L.lower())
                if v is None:
                    skip = True
                    break
                option_map[L] = str(v).strip()
                if not option_map[L]:
                    skip = True
                    break
            if skip:
                continue

            # Normalize correct_option: accept 'A'/'a'/'1'/... etc.
            co_raw = str(it.get("correct_option", "")).strip()
            co = ""
            if co_raw:
                first = co_raw[0].upper()
                if first in {"A", "B", "C", "D"}:
                    co = first
                elif first in {"1", "2", "3", "4"}:
                    co = "ABCD"[int(first) - 1]
            if co not in {"A", "B", "C", "D"}:
                continue

            key = " ".join(q_text.lower().split())
            if key in seen_questions:
                continue
            seen_questions.add(key)

            cleaned.append({
                "question": q_text,
                "options": option_map,
                "correct_option": co,
                "explanation": str(it.get("explanation", "")).strip(),
                "chapter": str(it.get("chapter", "")).strip(),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not cleaned:
        logger.error("No valid MCQs after cleaning. Item count=%d, raw prefix=%s",
                     len(items), raw[:600])
        raise ValueError("AI returned no valid MCQs. Please try again.")
    return cleaned
