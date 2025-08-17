# tests/test_mdt_utils.py
import json
import logging
import pytest
from medmdt.mdt.utils import parse_llm_json


def test_parse_plain_json():
    text = '{"key": "value", "num": 42}'
    result = parse_llm_json(text)
    assert result == {"key": "value", "num": 42}


def test_parse_json_in_markdown_fence():
    text = '```json\n{"diagnosis": "高血压"}\n```'
    result = parse_llm_json(text)
    assert result == {"diagnosis": "高血压"}


def test_parse_json_in_fence_without_language_tag():
    text = '```\n{"a": 1}\n```'
    result = parse_llm_json(text)
    assert result == {"a": 1}


def test_parse_json_with_surrounding_text():
    text = 'Here is the result:\n```json\n{"ok": true}\n```\nDone.'
    result = parse_llm_json(text)
    assert result == {"ok": True}


def test_parse_json_with_leading_trailing_whitespace():
    text = '  \n  {"key": "value"}  \n  '
    result = parse_llm_json(text)
    assert result == {"key": "value"}


def test_parse_invalid_json_raises_decode_error():
    with pytest.raises(json.JSONDecodeError, match="LLM output is not valid JSON"):
        parse_llm_json("this is not json at all")


def test_parse_invalid_json_logs_warning(caplog):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("{bad json")
    assert "Failed to parse LLM JSON output" in caplog.text


def test_parse_invalid_json_in_fence_raises():
    text = '```json\n{not valid}\n```'
    with pytest.raises(json.JSONDecodeError, match="LLM output is not valid JSON"):
        parse_llm_json(text)


def test_parse_nested_json():
    data = {"patient": {"name": "张三", "age": 65}, "vitals": [1, 2, 3]}
    text = f"```json\n{json.dumps(data, ensure_ascii=False)}\n```"
    result = parse_llm_json(text)
    assert result == data
