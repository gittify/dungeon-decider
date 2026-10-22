"""The game mechanics. Plain deterministic functions: (game, item_ref) -> resolution dict.

`MECHANICS` also carries the one-line descriptions Decider reads as option criteria, so the
tool list and the model's view of it can never drift apart.
"""

from __future__ import annotations

from typing import Any, Callable

from .world import Game

Resolution = dict[str, Any]


def _res(tool: str, **kw: Any) -> Resolution:
    return {"tool": tool, **kw}


def attack(g: Game, ref: str) -> Resolution:
    if not g.flags["goblin_asleep"] and "goblin" not in g.scene:
        return _res("attack", success=False, detail="There is nothing left to attack.")
    held = g.inventory.get(ref)
    weapon = held["name"] if held and held["qty"] > 0 else "improvised strike"
    sneak_bonus = 2 if g.flags["goblin_asleep"] else 0
    r = g.roll("dex", 11 - sneak_bonus)
    if r["success"]:
        g.flags["goblin_asleep"] = False
    return _res("attack", success=r["success"], roll=r, detail=f"Attack with {weapon} on the goblin guard.")


def cast_spell(g: Game, ref: str) -> Resolution:
    key = ref.removeprefix("spell_")
    if key not in g.spells:
        return _res("cast_spell", success=False, detail="No such spell known.")
    if g.slots <= 0:
        return _res("cast_spell", success=False, detail="Out of spell slots.")
    g.slots -= 1
    r = g.roll("int", 10)
    return _res("cast_spell", success=r["success"], roll=r, detail=f"Cast {key}. Slots left: {g.slots}.")


def use_item(g: Game, ref: str) -> Resolution:
    item = g.inventory.get(ref)
    if not item or item["qty"] <= 0:
        return _res("use_item", success=False, detail="Nothing like that to use.")
    if ref == "healing_potion":
        item["qty"] -= 1
        g.hp = min(12, g.hp + 5)
        return _res("use_item", success=True, detail=f"Drank a healing potion. HP now {g.hp}.")
    return _res("use_item", success=True, detail=f"Used the {item['name']}.")


def check_inventory(g: Game, ref: str) -> Resolution:
    items = ", ".join(f"{v['name']} x{v['qty']}" for v in g.inventory.values() if v["qty"] > 0)
    return _res("check_inventory", success=True, detail=f"Carrying: {items}. Spells: {', '.join(g.spells)}. HP {g.hp}.")


def sneak(g: Game, ref: str) -> Resolution:
    r = g.roll("dex", 12 if g.flags["goblin_asleep"] else 15)
    return _res("sneak", success=r["success"], roll=r, detail="Stealth check past the goblin guard.")


def persuade(g: Game, ref: str) -> Resolution:
    r = g.roll("cha", 14 if g.flags["goblin_asleep"] else 12)
    return _res("persuade", success=r["success"], roll=r, detail="Charm or talk your way past.")


def intimidate(g: Game, ref: str) -> Resolution:
    r = g.roll("str", 13)
    return _res("intimidate", success=r["success"], roll=r, detail="Menace the goblin guard.")


def investigate(g: Game, ref: str) -> Resolution:
    r = g.roll("wis", 10)
    found = "a hairline seam under the chest lid and a trip-wire behind the door" if r["success"] else "nothing new"
    return _res("investigate", success=r["success"], roll=r, detail=f"Search the room: found {found}.")


def disarm_trap(g: Game, ref: str) -> Resolution:
    r = g.roll("dex", 13)
    if r["success"]:
        g.flags["trap_armed"] = False
    return _res("disarm_trap", success=r["success"], roll=r, detail="Work on the pressure plate before the stone door.")


def climb_or_move(g: Game, ref: str) -> Resolution:
    r = g.roll("str", 12)
    return _res("climb_or_move", success=r["success"], roll=r, detail="Athletics: climb, jump or cross terrain.")


def free_roleplay(g: Game, ref: str) -> Resolution:
    return _res("free_roleplay", success=True, detail="No mechanic needed; pure roleplay or a question.")


# name -> (function, description Decider reads as the option's meaning)
MECHANICS: dict[str, tuple[Callable[[Game, str], Resolution], str]] = {
    "attack": (attack, "hit, stab, shoot or otherwise physically harm a creature with a weapon or fists"),
    "cast_spell": (cast_spell, "cast a magic spell"),
    "use_item": (use_item, "use, drink, light, equip or apply a carried item (not a weapon attack)"),
    "check_inventory": (check_inventory, "look at what the character is carrying or their own status"),
    "sneak": (sneak, "move quietly, hide, or avoid being noticed"),
    "persuade": (persuade, "talk, charm, bargain, bribe or negotiate"),
    "intimidate": (intimidate, "threaten, frighten or bully someone into compliance"),
    "investigate": (investigate, "search, inspect or examine the surroundings for clues or hidden things"),
    "disarm_trap": (disarm_trap, "detect, trigger-test, jam or disarm a trap or mechanism"),
    "climb_or_move": (climb_or_move, "climb, jump, swim, run, push or force through a physical obstacle"),
    "free_roleplay": (free_roleplay, "no game mechanic applies: dialogue, a question to the GM, or flavor"),
}

TOOLS: dict[str, Callable[[Game, str], Resolution]] = {k: v[0] for k, v in MECHANICS.items()}
MECHANIC_CRITERIA: dict[str, str] = {k: v[1] for k, v in MECHANICS.items()}
