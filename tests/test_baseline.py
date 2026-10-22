"""Baseline plumbing with a scripted model (no AWS). Checks parsing and the report, not accuracy."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402
from scripted_model import ScriptedModel  # noqa: E402

from eval.baseline_llm import ask_llm, build_report, parse_llm  # noqa: E402
from eval.cases import CASES  # noqa: E402


def test_parse_llm_extracts_json_from_chatty_reply():
    d = parse_llm('Sure! {"mechanic": "attack", "item": "iron_dagger", "feasible": true} Hope that helps.')
    assert d == {"mechanic": "attack", "item": "iron_dagger", "feasible": True}


def test_parse_llm_rejects_garbage():
    with pytest.raises(ValueError):
        parse_llm("I think the player attacks.")


def test_ask_llm_and_report_roundtrip():
    model = ScriptedModel([("text", '{"mechanic": "attack", "item": "none", "feasible": true}')])
    rows = {c.id: ask_llm(model, c) for c in CASES}
    assert all("parse_error" not in r for r in rows.values())
    text = build_report({"model_id": "scripted"}, rows, "test")
    assert "Tool selection" in text and "Grounding check" in text and "no calibration table" in text
