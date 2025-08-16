# src/medmdt/mdt/utils.py
"""Shared utilities for the MDT engine."""

import json
import logging
import re

logger = logging.getLogger(__name__)


def parse_llm_json(text: str) -> dict:
    """Extract and parse JSON from LLM output.

    Handles markdown code fences (```json ... ```) that LLMs commonly wrap
    around JSON responses.  On parse failure a descriptive error is raised
    after logging a warning with the raw text.

    Args:
        text: Raw LLM output that may contain JSON, optionally inside
              markdown fences.

    Returns:
        Parsed dictionary.

    Raises:
        json.JSONDecodeError: If the text does not contain valid JSON.
    """
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON output: %s", text[:200])
        raise json.JSONDecodeError(
            f"LLM output is not valid JSON: {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc
