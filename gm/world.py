"""Game state: one character, one scene. Hard facts (inventory, slots, HP) live here and are
enforced by code. Decider only resolves the *fuzzy* part: which thing the player is referring to."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Game:
    name: str = "Mira"
    hp: int = 12
    stats: dict[str, int] = field(
        default_factory=lambda: {"str": 2, "dex": 3, "cha": 1, "int": 2, "wis": 1}
    )
    # key -> {"name", "desc", "qty"}
    inventory: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "iron_dagger": {"name": "iron dagger", "desc": "a short iron dagger", "qty": 1},
            "rope": {"name": "rope", "desc": "fifty feet of hemp rope with a grappling hook", "qty": 1},
            "healing_potion": {"name": "healing potion", "desc": "a red vial that restores health", "qty": 2},
            "torch": {"name": "torch", "desc": "a lit torch", "qty": 1},
        }
    )
    spells: dict[str, str] = field(
        default_factory=lambda: {
            "firebolt": "the firebolt spell: hurls a bolt of fire at a target",
            "shield": "the shield spell: a magical barrier that blocks one attack",
        }
    )
    slots: int = 2
    scene: str = (
        "A damp crypt antechamber lit by your torch. A goblin guard dozes against the far wall. "
        "A stone door in the north wall has a pressure plate in front of it. A 40-foot chasm "
        "splits the floor to the east. A locked chest sits in the corner. "
        "Splintered chairs and loose stones are scattered across the floor."
    )
    flags: dict[str, bool] = field(
        default_factory=lambda: {"trap_armed": True, "goblin_asleep": True, "chest_locked": True}
    )
    seed: int | None = None
    last_resolution: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # ---- text handed to Decider -------------------------------------------------
    def character_text(self) -> str:
        items = ", ".join(f"{v['name']} x{v['qty']}" for v in self.inventory.values() if v["qty"] > 0)
        spells = ", ".join(self.spells) or "none"
        return (
            f"Character: {self.name}. Inventory: {items or 'nothing'}. "
            f"Known spells: {spells} (slots left: {self.slots})."
        )

    def state_text(self, player_text: str) -> str:
        return (
            f"SCENE: {self.scene}\n"
            f"{self.character_text()}\n"
            f'PLAYER ACTION: "{player_text}"'
        )

    # ---- the option set for the grounding question -------------------------------
    def item_options(self) -> dict[str, str]:
        """option name -> description. Built from live state, so it changes as items are used."""
        opts: dict[str, str] = {}
        for key, v in self.inventory.items():
            if v["qty"] > 0:
                opts[key] = f"the character's {v['name']}: {v['desc']}"
        for key, desc in self.spells.items():
            opts[f"spell_{key}"] = desc
        opts["none"] = "the action relies on no particular item or spell (bare hands, words, movement)"
        opts["unlisted"] = (
            "the action relies on an item, weapon or spell that is NOT any of the options above"
        )
        return opts

    def roll(self, stat: str, dc: int) -> dict[str, Any]:
        d20 = self.rng.randint(1, 20)
        mod = self.stats.get(stat, 0)
        return {"d20": d20, "mod": mod, "total": d20 + mod, "dc": dc, "success": d20 + mod >= dc}
