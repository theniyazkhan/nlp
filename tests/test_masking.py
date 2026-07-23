"""
tests/test_masking.py – pytest tests for src/masking.py
Run: pytest tests/test_masking.py -v
"""

import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from masking import find_copied_entities, mask_text, masked_scores


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / shared data
# ─────────────────────────────────────────────────────────────────────────────

LONDON = {"text": "London", "label": "GPE"}
BOB_DYLAN = {"text": "Bob Dylan", "label": "PERSON"}
YEAR_2016 = {"text": "2016", "label": "DATE"}
DALHOUSIE = {"text": "Dalhousie University", "label": "ORG"}


# ─────────────────────────────────────────────────────────────────────────────
# find_copied_entities  tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFindCopiedEntities:

    def test_exact_match(self):
        """Entity text appears verbatim in hypothesis."""
        result = find_copied_entities("Bob Dylan won the prize.", [BOB_DYLAN])
        assert result == [BOB_DYLAN]

    def test_case_mismatch(self):
        """Match is case-insensitive."""
        result = find_copied_entities("bob dylan won the prize.", [BOB_DYLAN])
        assert result == [BOB_DYLAN]

    def test_multi_word_entity(self):
        """Multi-word entity is matched as a whole phrase."""
        result = find_copied_entities("He studied at Dalhousie University.", [DALHOUSIE])
        assert result == [DALHOUSIE]

    def test_entity_not_present(self):
        """Entity text absent from hypothesis → empty list."""
        result = find_copied_entities("He attended the local university.", [DALHOUSIE])
        assert result == []

    def test_empty_entity_list(self):
        """Empty entity list → empty result."""
        result = find_copied_entities("Some text here.", [])
        assert result == []

    def test_partial_word_not_matched(self):
        """'Land' should not match entity 'London'."""
        result = find_copied_entities("He visited Landshire.", [LONDON])
        assert result == []

    def test_multiple_entities_partial_match(self):
        """Only the entities present in hypothesis are returned."""
        result = find_copied_entities("London hosted the event.", [LONDON, BOB_DYLAN])
        assert result == [LONDON]


# ─────────────────────────────────────────────────────────────────────────────
# mask_text  tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskText:

    def test_basic_mask(self):
        result = mask_text("London is a great city.", [LONDON])
        assert "London" not in result
        assert "is a great city" in result

    def test_multi_word_mask(self):
        result = mask_text("Bob Dylan won the 2016 Nobel Prize.", [BOB_DYLAN, YEAR_2016])
        assert "Bob Dylan" not in result
        assert "2016" not in result
        assert "won the Nobel Prize" in result

    def test_empty_entities_no_change(self):
        text = "Some original text."
        result = mask_text(text, [])
        assert result == text

    def test_whitespace_normalised(self):
        """Removing spans should not leave double spaces."""
        result = mask_text("A London B", [LONDON])
        assert "  " not in result
        assert result == "A B"


# ─────────────────────────────────────────────────────────────────────────────
# Bengali reference test (non-Latin script)
# ─────────────────────────────────────────────────────────────────────────────

class TestNonLatinReferences:

    def test_bengali_reference_entity_not_present(self):
        """English entity 'London' is not present in Bengali text; nothing masked."""
        bengali_ref = "লন্ডন একটি শহর।"   # Romanised would be "Landan ekti sahar."
        result = find_copied_entities(bengali_ref, [LONDON])
        # verbatim 'London' is NOT in the Bengali string
        assert result == []

    def test_bengali_hypothesis_unchanged(self):
        """Masking with entity not present in hypothesis leaves it unchanged."""
        bengali_hyp = "লন্ডন একটি শহর।"
        result = mask_text(bengali_hyp, [LONDON])
        assert result == bengali_hyp

    def test_arabic_reference_entity_not_present(self):
        """English entity 'London' is not in Arabic text."""
        arabic_ref = "لندن مدينة كبيرة."
        result = find_copied_entities(arabic_ref, [LONDON])
        assert result == []

    def test_arabic_hypothesis_unchanged(self):
        arabic_hyp = "لندن مدينة كبيرة."
        result = mask_text(arabic_hyp, [LONDON])
        assert result == arabic_hyp


# ─────────────────────────────────────────────────────────────────────────────
# masked_scores  tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskedScores:

    def test_return_keys(self):
        hyps = ["The cat sat on the mat."]
        refs = ["The cat sat on the mat."]
        ents = [[]]
        result = masked_scores(hyps, refs, ents)
        expected_keys = {
            "normal_bleu", "masked_bleu", "inflation_bleu",
            "normal_chrf", "masked_chrf", "inflation_chrf",
        }
        assert set(result.keys()) == expected_keys

    def test_no_entities_zero_inflation(self):
        """With no entities, masked == normal → inflation == 0."""
        hyps = ["The prize was awarded in Stockholm."]
        refs = ["The prize was awarded in Stockholm."]
        ents = [[]]
        result = masked_scores(hyps, refs, ents)
        assert result["inflation_bleu"] == pytest.approx(0.0, abs=1e-3)
        assert result["inflation_chrf"] == pytest.approx(0.0, abs=1e-3)

    def test_inflation_positive_when_entity_copied(self):
        """Inflation should be ≥ 0 when named entity is copied into hypothesis."""
        # hypothesis == English source → maximally copied
        hyps = ["Bob Dylan won the Nobel Prize in Literature."]
        refs = ["Bob Dylan a câștigat Premiul Nobel pentru Literatură."]  # Romanian ref
        ents = [[BOB_DYLAN]]
        result = masked_scores(hyps, refs, ents)
        assert result["inflation_chrf"] >= 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            masked_scores(["a"], ["b", "c"], [[]])

    def test_empty_lists(self):
        result = masked_scores([], [], [])
        assert result["normal_bleu"] == pytest.approx(0.0, abs=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Translit policy test
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslitPolicy:

    def test_translit_detects_romanised_entity(self):
        """
        unidecode('Лондон') == 'London' → fuzzy-matches entity 'London'.
        With verbatim policy it should NOT match; with translit it should.
        """
        cyrillic_london = "Лондон является городом."   # Russian: "London is a city."
        verbatim_result = find_copied_entities(cyrillic_london, [LONDON], policy="verbatim")
        assert verbatim_result == []   # verbatim: 'London' not in Cyrillic string

        translit_result = find_copied_entities(cyrillic_london, [LONDON], policy="translit")
        assert translit_result == [LONDON]   # translit: unidecode('Лондон') ≈ 'London'
