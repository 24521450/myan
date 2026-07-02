from __future__ import annotations
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING
import nltk
from nltk.stem import WordNetLemmatizer

if TYPE_CHECKING:
    from src.deck_builder.build_notes import BuiltCard

# Initialize NLTK WordNet Lemmatizer
try:
    lemmatizer = WordNetLemmatizer()
    # Test call to ensure it works
    lemmatizer.lemmatize("testing", pos="v")
except Exception as e:
    raise ImportError(f"Failed to initialize NLTK WordNet Lemmatizer: {e}")


def load_synonym_overrides(path: Path | str | None) -> dict[str, list[dict]]:
    """Loads manual synonym example overrides from a JSONL file grouped by card GUID."""
    overrides: dict[str, list[dict]] = {}
    if path is None:
        return overrides

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Synonym overrides file not found at: {path}")

    seen_overrides = set()
    with p.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                raise ValueError(f"Error parsing line {line_idx} of synonym overrides file: {e}")

            guid = r.get("guid")
            word = r.get("word")
            pos = r.get("pos")
            cefr = r.get("cefr")
            original_example = r.get("original_example")
            action = r.get("action")

            if not all([guid, word, pos, cefr, original_example, action]):
                raise ValueError(
                    f"Error line {line_idx}: missing one of required fields "
                    f"(guid, word, pos, cefr, original_example, action)"
                )

            guid = guid.strip()
            word = word.strip()
            pos = pos.strip()
            cefr = cefr.strip()
            original_example = original_example.strip()
            action = action.strip().lower()

            if action not in ("annotate", "skip"):
                raise ValueError(f"Error line {line_idx}: invalid action {action!r}")

            if action == "annotate":
                annotated_example = r.get("annotated_example")
                if not annotated_example or not annotated_example.strip():
                    raise ValueError(f"Error line {line_idx}: missing annotated_example for action 'annotate'")
                source_example = r.get("source_example")
                if not source_example or not source_example.strip():
                    raise ValueError(f"Error line {line_idx}: missing source_example for action 'annotate'")
            else:
                reason = r.get("reason")
                if not reason or not reason.strip():
                    raise ValueError(f"Error line {line_idx}: missing reason for action 'skip'")

            key = (guid, clean_for_matching(original_example))
            if key in seen_overrides:
                raise ValueError(f"Duplicate override for GUID {guid!r} and example {original_example!r} at line {line_idx}")
            seen_overrides.add(key)

            overrides.setdefault(guid, []).append(r)

    return overrides


def clean_for_matching(text: str) -> str:
    """Normalize text for matching (case-insensitive, clean quotes/smart symbols, clean whitespaces)."""
    cleaned = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(cleaned.lower().split())


def strip_synonym_annotations(chunk: str, synonyms: list[str]) -> str:
    """Removes parenthesized synonym annotations that match synonyms from a list."""
    if not synonyms:
        return chunk

    syn_lower_set = {s.lower().strip() for s in synonyms}

    def replace_fn(match):
        content = match.group(1).lower()
        words = [w.strip() for w in content.split(",") if w.strip()]
        if words and all(w in syn_lower_set for w in words):
            return ""  # strip the parenthesis and leading space
        return match.group(0)

    # Match " (content)" where content is inside parentheses
    return re.sub(r"\s+\(([^)]+)\)", replace_fn, chunk)


def is_already_annotated(chunk: str, synonyms: list[str]) -> bool:
    """Checks if any synonym in the list is already annotated in parentheses."""
    if not synonyms:
        return True
    chunk_lower = chunk.lower()
    for syn in synonyms:
        # Look for "(syn)" or "(..., syn, ...)"
        pattern = r"\(\s*[^)]*" + re.escape(syn.lower().strip()) + r"[^)]*\)"
        if re.search(pattern, chunk_lower):
            return True
    return False


def matches_lemma(word_in_chunk: str, headword: str) -> bool:
    """Lemmatizes word_in_chunk and compares it to headword."""
    w_lower = word_in_chunk.lower().strip()
    h_lower = headword.lower().strip()
    if w_lower == h_lower:
        return True
    
    # Try different POS lemmatizations
    for p in ("n", "v", "a", "r"):
        try:
            lemma = lemmatizer.lemmatize(w_lower, pos=p)
            if lemma == h_lower:
                return True
        except Exception:
            pass
    return False


def get_synonym_particles(synonyms: list[str]) -> set[str]:
    """Extracts common particle words used in phrasal verb synonyms."""
    particles = {"off", "out", "up", "down", "away", "in", "on", "back", "about", "over"}
    found = set()
    for syn in synonyms:
        words = [w.strip().lower() for w in re.split(r"[\s\-]+", syn) if w.strip()]
        for w in words:
            if w in particles:
                found.add(w)
    return found


def annotate_chunk_auto(chunk: str, headword: str, synonyms: list[str]) -> str | None:
    """Tries to automatically annotate the chunk with synonyms. Returns None if unresolved."""
    if not synonyms:
        return chunk

    cleaned = strip_synonym_annotations(chunk, synonyms)
    if is_already_annotated(cleaned, synonyms):
        return cleaned

    syns_clean = [s.strip() for s in synonyms if s.strip()]
    if not syns_clean:
        return cleaned

    syn_str = f" ({', '.join(syns_clean)})"
    syn_particles = get_synonym_particles(syns_clean)

    # Priority 1: Substring word-boundary match of the headword
    # Escape headword for safe regex
    if syn_particles:
        pattern = r"\b" + re.escape(headword.lower().strip()) + r"(?:\s+(?:" + "|".join(re.escape(p) for p in syn_particles) + r"))?\b"
    else:
        pattern = r"\b" + re.escape(headword.lower().strip()) + r"\b"

    match = re.search(pattern, cleaned.lower())
    if match:
        idx = match.end()
        return cleaned[:idx] + syn_str + cleaned[idx:]

    # Priority 2: Lemmatize each word in the chunk
    words_iter = list(re.finditer(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", cleaned))
    for idx_w, m in enumerate(words_iter):
        word_text = m.group()
        if matches_lemma(word_text, headword):
            end_idx = m.end()
            # Check if followed by a particle
            if idx_w + 1 < len(words_iter):
                next_word = words_iter[idx_w + 1]
                next_word_lower = next_word.group().lower()
                if next_word_lower in syn_particles:
                    # Check that between_text is only whitespace
                    between_text = cleaned[m.end():next_word.start()]
                    if not between_text.strip():
                        end_idx = next_word.end()
            return cleaned[:end_idx] + syn_str + cleaned[end_idx:]

    return None


def _validate_override_synonyms(original: str, annotated: str, allowed_synonyms: list[str]) -> list[str]:
    """Validates that annotated example only adds valid synonyms of the sense."""
    errors = []
    allowed_set = {s.lower().strip() for s in allowed_synonyms}
    
    # Find all parentheses in original and annotated
    orig_parents = set(re.findall(r"\(([^)]+)\)", original.lower()))
    annotated_parents = re.findall(r"\(([^)]+)\)", annotated.lower())
    
    for p in annotated_parents:
        if p in orig_parents:
            continue
        # Split by comma and verify each word is in allowed synonyms
        words = [w.strip() for w in p.split(",") if w.strip()]
        if not words:
            errors.append(f"Empty parenthesized annotation found in: {annotated!r}")
            continue
        for w in words:
            if w not in allowed_set:
                errors.append(
                    f"Synonym {w!r} inside parenthesis in manual mapping is not associated with this sense's synonyms: {allowed_synonyms}"
                )
    return errors


def annotate_card_examples(
    card: BuiltCard,
    specs: list[dict],
    overrides: dict[str, list[dict]]
) -> tuple[str, list[str]]:
    """Annotates card examples with synonyms of correct senses, checking overrides.
    
    Returns (annotated_example_field, list_of_errors).
    """
    errors: list[str] = []
    if not card.example.strip():
        return "", errors

    chunks = [ch.strip() for ch in card.example.split("|")]
    annotated_chunks: list[str] = []

    # Get union of all synonyms of all specs to safely clean chunks
    all_syns = set()
    for spec in specs:
        for s in spec.get("synonyms") or []:
            all_syns.add(s.strip())
    all_syns_list = list(all_syns)

    cleaned_chunks = [strip_synonym_annotations(ch, all_syns_list) for ch in chunks]

    # Resolve each chunk
    card_overrides = overrides.get(card.guid, [])
    used_overrides = set()

    # Validate card identity on all overrides for this card
    actual_word_clean = re.sub(r"\s*\(.*?\)\s*", "", card.word).strip().lower()
    actual_pos = card.pos.strip().lower()
    actual_cefr = card.cefr.strip().upper()
    for entry in card_overrides:
        expected_word = entry.get("word", "").strip().lower()
        expected_pos = entry.get("pos", "").strip().lower()
        expected_cefr = entry.get("cefr", "").strip().upper()
        if actual_word_clean != expected_word or actual_pos != expected_pos or actual_cefr != expected_cefr:
            errors.append(
                f"Card identity mismatch for override GUID {card.guid!r}: "
                f"override expected ({expected_word!r}, {expected_pos!r}, {expected_cefr!r}), "
                f"got ({actual_word_clean!r}, {actual_pos!r}, {actual_cefr!r})"
            )

    for i, cl_chunk in enumerate(cleaned_chunks):
        original_chunk = chunks[i]
        
        # Check manual overrides
        matched_override = None
        for idx, entry in enumerate(card_overrides):
            if idx in used_overrides:
                continue
            if clean_for_matching(entry.get("original_example") or "") == clean_for_matching(cl_chunk):
                matched_override = entry
                used_overrides.add(idx)
                break
        
        if matched_override:
            # We found a manual override!
            action = matched_override.get("action", "").strip().lower()

            # Find matching spec for original chunk if any
            spec_idx = None
            chunk_clean = clean_for_matching(cl_chunk)
            for j, spec in enumerate(specs):
                if clean_for_matching(spec["text"]) == chunk_clean:
                    spec_idx = j
                    break

            if action == "skip":
                if spec_idx is not None:
                    spec = specs[spec_idx]
                    if spec.get("synonyms"):
                        errors.append(
                            f"Skip action not allowed for exact sense with synonyms on card {card.word} ({card.guid}) "
                            f"for example {original_chunk!r}"
                        )
                annotated_chunks.append(cl_chunk)
                continue

            # For action == "annotate"
            source_ex = matched_override.get("source_example")
            if not source_ex or not source_ex.strip():
                errors.append(
                    f"Missing source_example in annotate override for card {card.word} ({card.guid}) "
                    f"for example {original_chunk!r}"
                )
                annotated_chunks.append(original_chunk)
                continue

            matched_spec_idx = None
            source_ex_clean = clean_for_matching(source_ex)
            for j, spec in enumerate(specs):
                if clean_for_matching(spec["text"]) == source_ex_clean:
                    matched_spec_idx = j
                    break

            if matched_spec_idx is None:
                errors.append(
                    f"source_example {source_ex!r} in override does not match any Oxford spec for card {card.word} ({card.guid})"
                )
                annotated_chunks.append(original_chunk)
                continue

            allowed_syns = specs[matched_spec_idx].get("synonyms") or []
            if not allowed_syns:
                errors.append(
                    f"source_example {source_ex!r} has no synonyms on card {card.word} ({card.guid})"
                )
                annotated_chunks.append(original_chunk)
                continue

            annotated_ex = matched_override.get("annotated_example") or ""
            # Validate that base text is not modified
            stripped_annotated = strip_synonym_annotations(annotated_ex, allowed_syns)
            if stripped_annotated.strip() != cl_chunk.strip():
                errors.append(
                    f"Base text modified in override for {card.word} ({card.guid}): "
                    f"original {cl_chunk!r}, stripped override {stripped_annotated!r}"
                )

            # Validate override synonyms
            validation_errors = _validate_override_synonyms(cl_chunk, annotated_ex, allowed_syns)
            if validation_errors:
                errors.extend([f"Manual override validation failed for {card.word} ({card.guid}): {err}" for err in validation_errors])

            annotated_chunks.append(annotated_ex)
            continue

        # No manual override, resolve via mapped spec
        spec_idx = None
        chunk_clean = clean_for_matching(cl_chunk)
        for j, spec in enumerate(specs):
            if clean_for_matching(spec["text"]) == chunk_clean:
                spec_idx = j
                break

        if spec_idx is not None:
            spec = specs[spec_idx]
            syns = spec.get("synonyms") or []
            if syns:
                annotated = annotate_chunk_auto(cl_chunk, actual_word_clean, syns)
                if annotated is None:
                    errors.append(
                        f"Unresolved auto-annotation for {card.word} ({card.guid}) chunk: {original_chunk!r}. "
                        f"Synonyms of sense: {syns}. Please add a manual override."
                    )
                    annotated_chunks.append(original_chunk)
                else:
                    annotated_chunks.append(annotated)
            else:
                # No synonyms for this sense
                annotated_chunks.append(cl_chunk)
        else:
            # We cannot map the chunk to any spec
            has_any_syns = any(spec.get("synonyms") for spec in specs)
            if has_any_syns:
                errors.append(
                    f"Unresolved alignment for {card.word} ({card.guid}) chunk: {original_chunk!r}. "
                    f"Could not map chunk to any Oxford sense. Please add a manual override."
                )
            annotated_chunks.append(original_chunk)

    # Check for unused overrides for this card
    for idx, entry in enumerate(card_overrides):
        if idx not in used_overrides:
            errors.append(
                f"Unused manual override for {card.word} ({card.guid}) with example: {entry.get('original_example')!r}. "
                f"Does not match any cleaned example chunk."
            )

    return "|".join(annotated_chunks), errors


def get_synonyms_specs_for_card(card: BuiltCard, senses_index: dict) -> list[dict]:
    """Helper to resolve synonym specs from senses_index based on card word, pos, cefr."""
    pos_parts = [p.strip().lower() for p in card.pos.split(",") if p.strip()]
    specs = []
    word_lower = card.word.lower()
    cefr = card.cefr or 'UNCLASSIFIED'
    
    for p in pos_parts:
        key = (word_lower, p, cefr)
        if key in senses_index:
            for ms in senses_index[key]:
                for ex_dict in ms.examples or []:
                    ex_text = (ex_dict.get("text") or "").strip()
                    syns = ex_dict.get("synonyms") or []
                    if ex_text:
                        specs.append({"text": ex_text, "synonyms": syns})
    return specs
