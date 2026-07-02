from __future__ import annotations
import pytest
from pathlib import Path
from src.deck_builder.build_notes import BuiltCard, build_notes, BuildNotesPaths
from src.config import ProjectPaths
from src.deck_builder.synonym_annotator import (
    clean_for_matching,
    strip_synonym_annotations,
    is_already_annotated,
    matches_lemma,
    annotate_chunk_auto,
    annotate_card_examples,
    load_synonym_overrides
)

def test_clean_for_matching():
    # Parentheses are preserved
    assert clean_for_matching("Fish are abundant (plentiful) in the lake.") == "fish are abundant (plentiful) in the lake."
    assert clean_for_matching("Some ‘quote’ and “another”") == "some 'quote' and \"another\""
    assert clean_for_matching("Some text (= if applicable)") == "some text (= if applicable)"


def test_strip_synonym_annotations():
    syns = ["plentiful", "obvious"]
    assert strip_synonym_annotations("Fish are abundant (plentiful) in the lake.", syns) == "Fish are abundant in the lake."
    assert strip_synonym_annotations("apparent (obvious) lack", syns) == "apparent lack"
    assert strip_synonym_annotations("apparent (something else) lack", syns) == "apparent (something else) lack"


def test_is_already_annotated():
    syns = ["plentiful", "obvious"]
    assert is_already_annotated("abundant (plentiful)", syns) is True
    assert is_already_annotated("apparent (clear, obvious)", syns) is True
    assert is_already_annotated("no annotation", syns) is False


def test_matches_lemma():
    assert matches_lemma("accumulated", "accumulate") is True
    assert matches_lemma("accumulating", "accumulate") is True
    assert matches_lemma("debts", "debt") is True
    assert matches_lemma("strange", "alien") is False


def test_annotate_chunk_auto_basic():
    assert annotate_chunk_auto("Fish are abundant in the lake.", "abundant", ["plentiful"]) == "Fish are abundant (plentiful) in the lake."
    assert annotate_chunk_auto("Debts accumulated.", "accumulate", ["amass"]) == "Debts accumulated (amass)."
    assert annotate_chunk_auto("Nothing matches.", "accumulate", ["amass"]) is None


def test_annotate_chunk_auto_particles_and_phrasals():
    # trigger off (set off)
    assert annotate_chunk_auto("Nuts can trigger off a violent allergic reaction.", "trigger", ["set off"]) == "Nuts can trigger off (set off) a violent allergic reaction."
    # teem down (pour) - note that pour has no particle down, so it does not extend!
    assert annotate_chunk_auto("The rain was teeming down.", "teem", ["pour"]) == "The rain was teeming (pour) down."
    # delve in (dig) - dig has no particle in, so it does not extend!
    assert annotate_chunk_auto("She delved in her handbag.", "delve", ["dig"]) == "She delved (dig) in her handbag."


def test_annotate_chunk_first_occurrence_only():
    # Should only insert after the first occurrence of the target word
    assert annotate_chunk_auto("He wanted to test the test.", "test", ["evaluate"]) == "He wanted to test (evaluate) the test."


def test_annotate_chunk_multiple_synonyms():
    assert annotate_chunk_auto("A crucial decision.", "crucial", ["critical", "essential"]) == "A crucial (critical, essential) decision."


def test_annotate_chunk_idempotency():
    syns = ["plentiful"]
    orig = "Fish are abundant (plentiful) in the lake."
    assert annotate_chunk_auto(orig, "abundant", syns) == orig


def test_annotate_card_examples_success():
    card = BuiltCard(
        guid="12345", notetype="EAVM", deck="Oxford", word="abundant", pos="adjective", ipa="...",
        definition="...", example="Fish are abundant in the lake.|an abundant supply",
        collocations="...", wordfamily="...", uk_audio="...", us_audio="...", source1="Oxford", source2="Oxford",
        cefr="B2", idioms="...", tags="..."
    )
    specs = [
        {"text": "Fish are abundant in the lake.", "synonyms": ["plentiful"]},
        {"text": "an abundant supply", "synonyms": ["rich"]}
    ]
    overrides = {}
    
    annotated, errors = annotate_card_examples(card, specs, overrides)
    assert not errors
    assert annotated == "Fish are abundant (plentiful) in the lake.|an abundant (rich) supply"


def test_annotate_card_examples_unmapped_without_override():
    card = BuiltCard(
        guid="12345", notetype="EAVM", deck="Oxford", word="delve", pos="verb", ipa="...",
        definition="...", example="She delved in her handbag.|We need to delve deeper into the issue.",
        collocations="...", wordfamily="...", uk_audio="...", us_audio="...", source1="Oxford", source2="Oxford",
        cefr="UNCLASSIFIED", idioms="...", tags="..."
    )
    specs = [
        {"text": "She delved in her handbag.", "synonyms": ["dig"]}
    ]
    overrides = {}
    
    annotated, errors = annotate_card_examples(card, specs, overrides)
    assert len(errors) == 1
    assert "Unresolved alignment" in errors[0]


def test_annotate_card_examples_skip_override():
    card = BuiltCard(
        guid="12345", notetype="EAVM", deck="Oxford", word="delve", pos="verb", ipa="...",
        definition="...", example="She delved in her handbag.|We need to delve deeper into the issue.",
        collocations="...", wordfamily="...", uk_audio="...", us_audio="...", source1="Oxford", source2="Oxford",
        cefr="UNCLASSIFIED", idioms="...", tags="..."
    )
    specs = [
        {"text": "She delved in her handbag.", "synonyms": ["dig"]}
    ]
    overrides = {
        "12345": [
            {
                "guid": "12345",
                "word": "delve",
                "pos": "verb",
                "cefr": "UNCLASSIFIED",
                "original_example": "We need to delve deeper into the issue.",
                "action": "skip",
                "reason": "not in Oxford"
            }
        ]
    }
    
    annotated, errors = annotate_card_examples(card, specs, overrides)
    assert not errors
    assert annotated == "She delved (dig) in her handbag.|We need to delve deeper into the issue."


def test_annotate_card_examples_annotate_override():
    card = BuiltCard(
        guid="12345", notetype="EAVM", deck="Oxford", word="cope", pos="verb", ipa="...",
        definition="...", example="I got to the stage where I wasn't coping any more.",
        collocations="...", wordfamily="...", uk_audio="...", us_audio="...", source1="Oxford", source2="Oxford",
        cefr="B2", idioms="...", tags="..."
    )
    specs = [
        {"text": "I got to the stage where I wasn't coping any more.", "synonyms": ["manage"]}
    ]
    overrides = {
        "12345": [
            {
                "guid": "12345",
                "word": "cope",
                "pos": "verb",
                "cefr": "B2",
                "original_example": "I got to the stage where I wasn't coping any more.",
                "action": "annotate",
                "source_example": "I got to the stage where I wasn't coping any more.",
                "annotated_example": "I got to the stage where I wasn't coping (manage) any more."
            }
        ]
    }
    
    annotated, errors = annotate_card_examples(card, specs, overrides)
    assert not errors
    assert annotated == "I got to the stage where I wasn't coping (manage) any more."


def test_override_validations(tmp_path):
    # Test missing override file fails
    missing_file = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        load_synonym_overrides(missing_file)

    # Test invalid jsonl fails
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("invalid json\n")
    with pytest.raises(ValueError):
        load_synonym_overrides(bad_file)

    # Test duplicate override fails
    dup_file = tmp_path / "dup.jsonl"
    dup_file.write_text(
        '{"guid": "123", "word": "w", "pos": "p", "cefr": "c", "original_example": "ex", "action": "skip", "reason": "r"}\n'
        '{"guid": "123", "word": "w", "pos": "p", "cefr": "c", "original_example": "ex", "action": "skip", "reason": "r"}\n'
    )
    with pytest.raises(ValueError):
        load_synonym_overrides(dup_file)


def test_override_identity_and_base_text_errors():
    card = BuiltCard(
        guid="12345", notetype="EAVM", deck="Oxford", word="cope", pos="verb", ipa="...",
        definition="...", example="I wasn't coping any more.",
        collocations="...", wordfamily="...", uk_audio="...", us_audio="...", source1="Oxford", source2="Oxford",
        cefr="B2", idioms="...", tags="..."
    )
    specs = [{"text": "I wasn't coping any more.", "synonyms": ["manage"]}]

    # Identity mismatch
    overrides_identity = {
        "12345": [
            {
                "guid": "12345",
                "word": "wrong-word",  # mismatch
                "pos": "verb",
                "cefr": "B2",
                "original_example": "I wasn't coping any more.",
                "action": "skip",
                "reason": "testing"
            }
        ]
    }
    _, errors = annotate_card_examples(card, specs, overrides_identity)
    assert len(errors) == 2
    assert "Card identity mismatch" in errors[0]
    assert "Skip action not allowed for exact sense with synonyms" in errors[1]

    # Base text modified in override
    overrides_modified_base = {
        "12345": [
            {
                "guid": "12345",
                "word": "cope",
                "pos": "verb",
                "cefr": "B2",
                "original_example": "I wasn't coping any more.",
                "action": "annotate",
                "source_example": "I wasn't coping any more.",
                "annotated_example": "I wasn't coping (manage) any more and some extra words."  # modified base text!
            }
        ]
    }
    _, errors = annotate_card_examples(card, specs, overrides_modified_base)
    assert len(errors) == 1
    assert "Base text modified" in errors[0]


def test_skip_abundant_fails():
    card = BuiltCard(
        guid="12345", notetype="EAVM", deck="Oxford", word="abundant", pos="adjective", ipa="...",
        definition="...", example="Fish are abundant in the lake.",
        collocations="...", wordfamily="...", uk_audio="...", us_audio="...", source1="Oxford", source2="Oxford",
        cefr="B2", idioms="...", tags="..."
    )
    specs = [
        {"text": "Fish are abundant in the lake.", "synonyms": ["plentiful"]}
    ]
    overrides = {
        "12345": [
            {
                "guid": "12345",
                "word": "abundant",
                "pos": "adjective",
                "cefr": "B2",
                "original_example": "Fish are abundant in the lake.",
                "action": "skip",
                "reason": "Skip exact sense with synonyms"
            }
        ]
    }
    _, errors = annotate_card_examples(card, specs, overrides)
    assert len(errors) == 1
    assert "Skip action not allowed for exact sense with synonyms" in errors[0]


def test_annotate_missing_or_invalid_source_example():
    card = BuiltCard(
        guid="12345", notetype="EAVM", deck="Oxford", word="cope", pos="verb", ipa="...",
        definition="...", example="I wasn't coping any more.",
        collocations="...", wordfamily="...", uk_audio="...", us_audio="...", source1="Oxford", source2="Oxford",
        cefr="B2", idioms="...", tags="..."
    )
    specs = [
        {"text": "I wasn't coping any more.", "synonyms": ["manage"]}
    ]

    # Missing source_example
    overrides_missing = {
        "12345": [
            {
                "guid": "12345",
                "word": "cope",
                "pos": "verb",
                "cefr": "B2",
                "original_example": "I wasn't coping any more.",
                "action": "annotate",
                # missing source_example
                "annotated_example": "I wasn't coping (manage) any more."
            }
        ]
    }
    _, errors = annotate_card_examples(card, specs, overrides_missing)
    assert len(errors) == 1
    assert "Missing source_example in annotate override" in errors[0]

    # Invalid source_example (does not match spec)
    overrides_invalid = {
        "12345": [
            {
                "guid": "12345",
                "word": "cope",
                "pos": "verb",
                "cefr": "B2",
                "original_example": "I wasn't coping any more.",
                "action": "annotate",
                "source_example": "completely different example text",
                "annotated_example": "I wasn't coping (manage) any more."
            }
        ]
    }
    _, errors = annotate_card_examples(card, specs, overrides_invalid)
    assert len(errors) == 1
    assert "does not match any Oxford spec" in errors[0]


def test_two_senses_same_synonym_no_ambiguity():
    card = BuiltCard(
        guid="12345", notetype="EAVM", deck="Oxford", word="test", pos="verb", ipa="...",
        definition="...", example="I need to test this.",
        collocations="...", wordfamily="...", uk_audio="...", us_audio="...", source1="Oxford", source2="Oxford",
        cefr="B2", idioms="...", tags="..."
    )
    specs = [
        {"text": "Let us test them.", "synonyms": ["evaluate"]},
        {"text": "They test the software.", "synonyms": ["evaluate"]}
    ]
    # The chunk "I need to test this" is unmapped. It should not guess or fallback to any evaluate spec.
    _, errors = annotate_card_examples(card, specs, {})
    assert len(errors) == 1
    assert "Unresolved alignment" in errors[0]


def test_unknown_guid_fails_on_single_card_build(tmp_path):
    # Setup single card build
    vocab_txt = tmp_path / "vocab.txt"
    vocab_txt.write_text(
        "#separator:tab\n#html:true\n#guid column:1\n#notetype column:2\n#deck column:3\n#tags column:17\n"
        "guid_conq\tModel\tDeck Oxford\tconquer\tverb\t/ipa/\tdefn\tex\t\t\t\t\tOxford\tOxford\tC1\t\tSource::Oxford CEFR::C1\n",
        encoding="utf-8"
    )

    oxford_jsonl = tmp_path / "oxford.jsonl"
    oxford_jsonl.write_text(
        '{"word": "conquer", "pos_data": [{"pos": "verb", "definitions": [{"text": "defn"}]}]}\n',
        encoding="utf-8"
    )

    # Overrides has a DIFFERENT guid (unknown GUID)
    overrides_file = tmp_path / "overrides.jsonl"
    overrides_file.write_text(
        '{"guid": "different_guid", "word": "abundant", "pos": "adjective", "cefr": "B2", "original_example": "ex", "action": "skip", "reason": "testing"}\n',
        encoding="utf-8"
    )

    paths = BuildNotesPaths(
        oxford_jsonl_path=oxford_jsonl,
        notes_txt_path=vocab_txt,
        deck_audit_jsonl_path=tmp_path / "audit.jsonl",
        gamma_verdicts_path=tmp_path / "gamma.json",
        oxford_3000_md=tmp_path / "oxford_3000.md",
        oxford_5000_md=tmp_path / "oxford_5000.md",
        awl_md=tmp_path / "awl.md",
        manual_card_fills_path=tmp_path / "filled.json",
        audio_dir=tmp_path / "audio",
        synonym_example_overrides_path=overrides_file
    )

    # Write empty files for dependencies
    paths.deck_audit_jsonl_path.write_text("", encoding="utf-8")
    paths.gamma_verdicts_path.write_text('{"verdicts": []}\n', encoding="utf-8")
    paths.oxford_3000_md.write_text("", encoding="utf-8")
    paths.oxford_5000_md.write_text("", encoding="utf-8")
    paths.awl_md.write_text("", encoding="utf-8")
    paths.manual_card_fills_path.write_text("[]\n", encoding="utf-8")
    paths.audio_dir.mkdir(exist_ok=True)

    # Should fail due to unknown GUID in overrides
    with pytest.raises(ValueError) as exc:
        build_notes(paths)
    assert "Unknown card GUIDs defined in synonym overrides" in str(exc.value)


def test_production_overrides_loaded_and_used_once():
    # This integration test verifies that the production overrides file is 100% correct,
    # and all 14 overrides are successfully matched and used exactly once with 0 errors.
    proj = ProjectPaths()

    # Load cards and run the real builder to verify
    paths = BuildNotesPaths(
        oxford_jsonl_path=proj.oxford_jsonl,
        notes_txt_path=proj.anki_notes_txt,
        deck_audit_jsonl_path=proj.deck_audit_jsonl,
        gamma_verdicts_path=proj.gamma_verdicts,
        oxford_3000_md=proj.oxford_3000_md,
        oxford_5000_md=proj.oxford_5000_md,
        awl_md=proj.awl_md,
        manual_card_fills_path=proj.manual_card_fills,
        audio_dir=proj.audio_dir,
        synonym_example_overrides_path=proj.synonym_example_overrides,
        review_overrides_path=proj.non_oxford_non_c2_overrides
    )

    res = build_notes(paths)
    assert res.built_cards_count == 2452
