"""Offline tests: response parsing, policy bands, the router and a full GameMaster turn."""

from __future__ import annotations

import pytest

from decider_patterns import ActionRouter, Decider, Raw, Thresholds, classify, parse_choice
from decider_patterns.client import DEFAULT_URL
from decider_patterns.fake import FakeDecider
from gm.master import GameMaster
from gm.tools import MECHANIC_CRITERIA
from gm.world import Game


def _raw(tool_conf=0.9, p_unlisted=0.0, feasible=0.95) -> Raw:
    return Raw("attack", tool_conf, {"attack": 0.9}, "none", 0.9,
               {"none": 1 - p_unlisted, "unlisted": p_unlisted}, feasible)


def test_decider_none_url_uses_default():
    assert Decider(None).url == DEFAULT_URL.rstrip("/")


def test_parse_choice_matches_verified_server_shape():
    ans = {"type": "choice", "choice": "attack", "probabilities": {"attack": 0.8, "sneak": 0.2}, "confidence": 0.6}
    assert parse_choice(ans, ["attack", "sneak"]) == ("attack", 0.6, {"attack": 0.8, "sneak": 0.2})


def test_parse_choice_fails_loudly_on_unknown_shape():
    with pytest.raises(ValueError, match="unexpected choice answer shape"):
        parse_choice({"answer": "attack"})
    with pytest.raises(ValueError, match="not one of the offered options"):
        parse_choice({"choice": "dance", "probabilities": {}, "confidence": 1.0}, ["attack"])


@pytest.mark.parametrize("raw,verdict", [
    (_raw(0.95), "fast"),
    (_raw(0.70), "fast"),
    (_raw(0.55), "fast_hedged"),
    (_raw(0.20), "escalate"),
    (_raw(0.95, p_unlisted=0.8), "ungrounded"),      # cheating beats confidence
    (_raw(0.95, feasible=0.1), "infeasible"),
    (_raw(0.20, p_unlisted=0.8, feasible=0.1), "ungrounded"),  # ungrounded checked first
])
def test_classify_bands(raw, verdict):
    assert classify(raw, Thresholds()) == verdict


def test_thresholds_roundtrip(tmp_path):
    p = tmp_path / "t.json"
    Thresholds(fast=0.8, hedge=0.3, ungrounded=0.5, feasible_min=0.2).save(p, tuned_on="dev")
    assert Thresholds.load(p) == Thresholds(0.8, 0.3, 0.5, 0.2)


def test_item_options_follow_live_state():
    g = Game()
    assert "healing_potion" in g.item_options()
    g.inventory["healing_potion"]["qty"] = 0
    opts = g.item_options()
    assert "healing_potion" not in opts and "unlisted" in opts and "none" in opts


def test_router_one_request_three_questions():
    d = FakeDecider()
    g = Game()
    dec = ActionRouter(d).decide("x", g.state_text("I cast firebolt at the goblin"),
                                 MECHANIC_CRITERIA, g.item_options())
    assert d.calls == 1
    assert dec.tool == "cast_spell" and dec.verdict in ("fast", "fast_hedged")


def test_game_master_turn_no_llm_enforces_hard_facts():
    gm = GameMaster(Game(seed=1), FakeDecider(), llm=False)
    gm.game.inventory["healing_potion"]["qty"] = 0
    res = gm.turn("I drink a healing potion")
    # The fake sees the live option set; the potion is gone so the code path must not heal.
    assert gm.game.inventory["healing_potion"]["qty"] == 0
    assert gm.game.hp == 12
    assert res.narration and "decider" in res.ticker()


def test_game_master_refuses_unowned_spell():
    gm = GameMaster(Game(seed=1), FakeDecider(), llm=False)
    res = gm.turn("I cast lightning bolt at the goblin")
    assert res.path == "ungrounded" and res.resolutions == []
    assert gm.game.slots == 2, "no spell slot may be spent on a refused action"


def test_escalate_without_llm_uses_top_pick():
    gm = GameMaster(Game(seed=1), FakeDecider(), llm=False)
    res = gm.turn("hmm")
    assert res.path == "escalated" and "--no-llm" in res.note and len(res.resolutions) == 1
