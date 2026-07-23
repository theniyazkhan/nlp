"""
Reusable entity-masking evaluation module.
Identifies verbatim and transliterated entity copies in translation hypotheses and computes entity-masked BLEU and chrF++ scores.
Masking is applied symmetrically to both hypothesis and reference.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List

import sacrebleu

try:
    from unidecode import unidecode as _unidecode
    _UNIDECODE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _UNIDECODE_AVAILABLE = False


def _word_boundary_pattern(text: str) -> re.Pattern:
    escaped = re.escape(text)
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE | re.UNICODE)


def _remove_spans(text: str, spans: List[tuple]) -> str:
    if not spans:
        return text
    spans = sorted(set(spans))
    result = []
    prev = 0
    for start, end in spans:
        if start > prev:
            result.append(text[prev:start])
        prev = end
    result.append(text[prev:])
    return " ".join("".join(result).split())


def find_copied_entities(
    hypothesis: str,
    entities: List[Dict],
    policy: str = "verbatim",
) -> List[Dict]:
    """
    Return entity dicts from `entities` that appear in `hypothesis`.
    """
    if not entities:
        return []

    copied = []
    for ent in entities:
        ent_text = ent.get("text", "")
        if not ent_text:
            continue

        pattern = _word_boundary_pattern(ent_text)
        if pattern.search(hypothesis):
            copied.append(ent)
            continue

        if policy == "translit":
            if not _UNIDECODE_AVAILABLE:
                raise ImportError(
                    "Package 'unidecode' is required for policy='translit'."
                )
            romanised = _unidecode(hypothesis)
            for token in romanised.split():
                ratio = SequenceMatcher(
                    None, token.lower(), ent_text.lower()
                ).ratio()
                if ratio > 0.8:
                    copied.append(ent)
                    break

    return copied


def mask_text(text: str, entities: List[Dict]) -> str:
    """
    Remove spans of copied entity text from `text`, normalise whitespace.
    """
    if not entities or not text:
        return text

    spans_to_remove = []
    for ent in entities:
        ent_text = ent.get("text", "")
        if not ent_text:
            continue
        pattern = _word_boundary_pattern(ent_text)
        for m in pattern.finditer(text):
            spans_to_remove.append((m.start(), m.end()))

    return _remove_spans(text, spans_to_remove)


def masked_scores(
    hypotheses: List[str],
    references: List[str],
    entities_per_sentence: List[List[Dict]],
    policy: str = "verbatim",
) -> Dict[str, float]:
    """
    Compute normal and entity-masked BLEU / chrF++ scores.
    """
    if len(hypotheses) != len(references) or len(hypotheses) != len(entities_per_sentence):
        raise ValueError(
            "hypotheses, references, and entities_per_sentence must have the same length."
        )

    if not hypotheses:
        return {
            "normal_bleu": 0.0,
            "masked_bleu": 0.0,
            "inflation_bleu": 0.0,
            "normal_chrf": 0.0,
            "masked_chrf": 0.0,
            "inflation_chrf": 0.0,
        }

    normal_bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    normal_chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score

    masked_hyps: List[str] = []
    masked_refs: List[str] = []

    for hyp, ref, entities in zip(hypotheses, references, entities_per_sentence):
        copied = find_copied_entities(hyp, entities, policy=policy)
        masked_hyps.append(mask_text(hyp, copied))
        masked_refs.append(mask_text(ref, copied))

    masked_bleu = sacrebleu.corpus_bleu(masked_hyps, [masked_refs]).score
    masked_chrf = sacrebleu.corpus_chrf(masked_hyps, [masked_refs], word_order=2).score

    return {
        "normal_bleu": round(normal_bleu, 4),
        "masked_bleu": round(masked_bleu, 4),
        "inflation_bleu": round(normal_bleu - masked_bleu, 4),
        "normal_chrf": round(normal_chrf, 4),
        "masked_chrf": round(masked_chrf, 4),
        "inflation_chrf": round(normal_chrf - masked_chrf, 4),
    }
