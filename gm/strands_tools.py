"""Strands tool wrappers for the 11 mechanics (used on the slow / escalated path only).

The fast path calls `gm.tools.TOOLS[...]` directly. The Bedrock arbiter sees the *same* mechanics
through these wrappers, so both paths mutate the same `Game` and return the same resolutions.
Every call is appended to `sink` so the GameMaster can narrate what actually happened.
"""

from __future__ import annotations

import json
from typing import Any

from strands import tool

from .tools import MECHANICS
from .world import Game


def normalise_ref(game: Game, item: str) -> str:
    """Map whatever the LLM wrote ('firebolt', 'iron dagger', 'none') to a game key or ''."""
    item = (item or "").strip().lower().replace(" ", "_")
    if item in ("", "none", "null", "n/a", "unlisted"):
        return ""
    if item in game.inventory or item.startswith("spell_"):
        return item
    if item in game.spells:
        return f"spell_{item}"
    for key, v in game.inventory.items():  # allow the display name
        if v["name"].replace(" ", "_") == item:
            return key
    return item  # unknown: the mechanic itself reports "no such thing"


def make_strands_tools(game: Game, sink: list[dict[str, Any]]) -> list[Any]:
    tools = []
    for name, (fn, description) in MECHANICS.items():
        tools.append(_wrap(game, sink, name, fn, description))
    return tools


def _wrap(game: Game, sink: list[dict[str, Any]], name: str, fn: Any, description: str) -> Any:
    @tool(name=name, description=description)
    def mechanic(target: str = "", item: str = "none") -> str:
        """Resolve one game mechanic.

        Args:
            target: who or what the action is aimed at (free text, may be empty).
            item: key of the inventory item or spell the action relies on, or "none".
        """
        resolution = fn(game, normalise_ref(game, item))
        sink.append(resolution)
        return json.dumps(resolution)

    return mechanic
