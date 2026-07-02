import pytest
from lxml import html as lxml_html
from src.scraper.oxford import _extract_synonyms
from src.scraper.merge import merge_word_records

def test_extract_synonyms_none():
    html_content = """
    <li class="sense">
        <span class="def">having a lot of something</span>
    </li>
    """
    element = lxml_html.fromstring(html_content)
    assert _extract_synonyms(element) == []

def test_extract_synonyms_single():
    html_content = """
    <li class="sense">
        <span class="def">having a lot of something</span>
        <span class="xrefs" xt="syn" hclass="xrefs">
            <span class="prefix">synonym</span>
            <span class="xh">plentiful</span>
        </span>
    </li>
    """
    element = lxml_html.fromstring(html_content)
    assert _extract_synonyms(element) == ["plentiful"]

def test_extract_synonyms_multiple_and_dedup():
    html_content = """
    <li class="sense">
        <span class="def">having a lot of something</span>
        <span class="xrefs" xt="syn" hclass="xrefs">
            <span class="prefix">synonym</span>
            <span class="xh">plentiful</span>
            <span class="xh">abundant</span>
            <span class="xh">plentiful</span>
        </span>
    </li>
    """
    element = lxml_html.fromstring(html_content)
    assert _extract_synonyms(element) == ["plentiful", "abundant"]

def test_extract_synonyms_exclusion():
    # Opposites (xt="opp") and see-also prefix should be excluded
    html_content = """
    <li class="sense">
        <span class="def">having a lot of something</span>
        <span class="xrefs" xt="opp" hclass="xrefs">
            <span class="prefix">opposite</span>
            <span class="xh">scarce</span>
        </span>
        <span class="xrefs" xt="syn" hclass="xrefs">
            <span class="xh">plentiful</span>
        </span>
    </li>
    """
    element = lxml_html.fromstring(html_content)
    assert _extract_synonyms(element) == ["plentiful"]

def test_merge_union_synonyms():
    # Two records with identical definitions, but different synonyms
    record1 = {
        "word": "abundant",
        "homonym_index": 1,
        "source": "Oxford",
        "pos": ["adjective"],
        "pos_data": [
            {
                "pos": "adjective",
                "definitions": [
                    {
                        "n": 1,
                        "sensenum_local": "1",
                        "text": "existing in large quantities",
                        "cefr": "C1",
                        "topics": [],
                        "collocations": {},
                        "examples": [],
                        "is_phrase": False,
                        "is_idiom": False,
                        "synonyms": ["plentiful"]
                    }
                ]
            }
        ]
    }
    
    record2 = {
        "word": "abundant",
        "homonym_index": 1,
        "source": "Oxford",
        "pos": ["adjective"],
        "pos_data": [
            {
                "pos": "adjective",
                "definitions": [
                    {
                        "n": 1,
                        "sensenum_local": "1",
                        "text": "existing in large quantities",
                        "cefr": "C1",
                        "topics": [],
                        "collocations": {},
                        "examples": [],
                        "is_phrase": False,
                        "is_idiom": False,
                        "synonyms": ["copious", "plentiful"]
                    }
                ]
            }
        ]
    }

    merged = merge_word_records([record1, record2])
    defs = merged["pos_data"][0]["definitions"]
    assert len(defs) == 1
    # union, preserving order of first appearance, no duplicates
    assert defs[0]["synonyms"] == ["plentiful", "copious"]
