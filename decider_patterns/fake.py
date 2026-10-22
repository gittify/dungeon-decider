"""KEYWORD STAND-IN for the Decider server -- for plumbing tests and offline UI work ONLY.

It returns schema-valid answers (same shape as the real server, see client.py) from crude
keyword rules. It is NOT the model: any accuracy, calibration or latency figure produced with
it is meaningless and must never go in the article. The eval script refuses to report numbers
from it unless you pass --fake, and then prints a banner on every table.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

BANNER = "*** FAKE KEYWORD DECIDER -- PLUMBING TEST ONLY, NUMBERS ARE MEANINGLESS ***"

_MECH_RULES: list[tuple[str, str]] = [
    ("cast_spell", r"\b(cast|spell|firebolt|fireball|lightning|incantation|chant)\b"),
    ("use_item", r"\b(drink|quaff|use|light|equip|apply)\b"),
    ("check_inventory", r"\b(inventory|carrying|what do i have|my pack|my hp|status)\b"),
    ("disarm_trap", r"\b(trap|pressure plate|disarm|jam)\b"),
    ("intimidate", r"\b(threaten|intimidate|scare|menace|bully|growl)\b"),
    ("persuade", r"\b(persuade|bribe|charm|bargain|negotiate|convince|sweet-talk|offer)\b"),
    ("sneak", r"\b(sneak|tiptoe|quietly|hide|creep)\b"),
    ("investigate", r"\b(search|inspect|examine|investigate|look around|study)\b"),
    ("climb_or_move", r"\b(climb|jump|leap|swim|cross|push|run|swing across|vault)\b"),
    ("attack", r"\b(attack|stab|hit|slash|swing|strike|shoot|punch|kick|throw|smash|bash)\b"),
]
_UNOWNED = r"\b(sword|axe|bow|crossbow|wand|staff|grenade|bomb|lightning|fireball|teleport|dragon|pistol|gun|shield of|mace|spear|magic missile)\b"
_IMPOSSIBLE = r"\b(through the (stone )?wall|moon|teleport|fly away|time travel|call my mother)\b"


def _player_text(state: str) -> str:
    m = re.search(r'PLAYER ACTION: "(.*)"', state, re.S)
    return (m.group(1) if m else state).lower()


def _spread(options: list[str], pick: str, top: float = 0.82) -> dict[str, float]:
    rest = (1.0 - top) / max(1, len(options) - 1)
    return {o: round(top if o == pick else rest, 4) for o in options}


def _choice(options: list[str], pick: str, top: float = 0.82) -> dict[str, Any]:
    probs = _spread(options, pick, top)
    n = len(options)
    conf = max(0.0, min(1.0, (n * max(probs.values()) - 1.0) / (n - 1.0)))
    return {"type": "choice", "choice": pick, "probabilities": probs, "confidence": round(conf, 4)}


class FakeDecider:
    def __init__(self) -> None:
        self.calls = 0
        self.last_latency_ms: float | None = 1.0
        self.last_usage: dict[str, int] | None = {"input_tokens": 0, "output_tokens": 0}

    def ask(self, state: str, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        text = _player_text(state)
        out: dict[str, Any] = {}
        for name, q in questions.items():
            if q["type"] == "noul":
                bad = re.search(_IMPOSSIBLE, text)
                out[name] = {"type": "noul", "noul": 0.05 if bad else 0.93}
            elif name == "mechanic":
                opts = list(q["criteria"])
                pick, top = "free_roleplay", 0.45
                for mech, pat in _MECH_RULES:
                    if mech in opts and re.search(pat, text):
                        pick, top = mech, 0.85
                        break
                out[name] = _choice(opts, pick if pick in opts else opts[0], top)
            else:  # item_ref
                opts = list(q["criteria"])
                pick = "none"
                for o in opts:
                    words = [w for w in re.split(r"[_\s]+", o.replace("spell_", "")) if len(w) > 2]
                    if o not in ("none", "unlisted") and words and all(w in text for w in words):
                        pick = o
                        break
                if pick == "none" and re.search(_UNOWNED, text):
                    pick = "unlisted"
                out[name] = _choice(opts, pick if pick in opts else opts[0], 0.9)
        return out
