"""Labelled player actions for the eval. 60 cases, default scene/inventory unless `setup` says otherwise.

LABELS ARE HAND-WRITTEN (by the assistant that built this scaffold, from your design doc) and
have NOT been checked against model output. Read them before trusting any number: a wrong label
costs you accuracy you did not lose. Where a case could reasonably map to several mechanics,
`mechanic` is a list of accepted answers (the 'ambiguous' and 'multi' categories do this on
purpose -- for those, the interesting question is whether *confidence* is lower, not accuracy).

Fields
  category  clear | live_state | cheat | ambiguous | multi | improv | infeasible
  mechanic  accepted mechanic(s); None = not scored for tool selection (impossible actions)
  grounded  False -> the action relies on an item/spell the character does not have (should be refused)
  feasible  False -> impossible in this scene (should be explained, not routed)
  item      optional expected item_ref key ("none", "iron_dagger", "spell_firebolt", ...)
  setup     optional state change: {"potions": 0} or {"drop": ["iron_dagger"]}

Split: within each category, cases alternate dev, test, dev, test... (deterministic, stratified).
Tune thresholds on dev only; report on test only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gm.world import Game


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    text: str
    mechanic: tuple[str, ...] | None
    grounded: bool = True
    feasible: bool = True
    item: str | None = None
    setup: dict[str, Any] | None = None
    split: str = "dev"

    def expected_verdict(self) -> str:
        if not self.grounded:
            return "ungrounded"
        if not self.feasible:
            return "infeasible"
        return "routed"


def _m(x: str | list[str] | None) -> tuple[str, ...] | None:
    if x is None:
        return None
    return (x,) if isinstance(x, str) else tuple(x)


# (category, text, mechanic, grounded, feasible, item, setup)
_RAW: list[tuple] = [
    # ---- clear -----------------------------------------------------------------
    ("clear", "I stab the goblin with my iron dagger", "attack", True, True, "iron_dagger", None),
    ("clear", "I punch the sleeping goblin", "attack", True, True, "none", None),
    ("clear", "I cast firebolt at the goblin guard", "cast_spell", True, True, "spell_firebolt", None),
    ("clear", "I drink a healing potion", "use_item", True, True, "healing_potion", None),
    ("clear", "What am I carrying right now?", "check_inventory", True, True, None, None),
    ("clear", "I creep quietly along the wall so the goblin doesn't wake", "sneak", True, True, "none", None),
    ("clear", "I try to sweet-talk the goblin into letting me through the door", "persuade", True, True, "none", None),
    ("clear", "I growl that I'll cut his throat if he makes a sound", "intimidate", True, True, "none", None),
    ("clear", "I search the room carefully for anything hidden", "investigate", True, True, "none", None),
    ("clear", "I examine the pressure plate and try to disarm it", "disarm_trap", True, True, "none", None),
    ("clear", "I cast shield on myself", "cast_spell", True, True, "spell_shield", None),
    ("clear", "I climb along the wall to get around the pressure plate", "climb_or_move", True, True, "none", None),
    ("clear", "Is the goblin armed?", ["free_roleplay", "investigate"], True, True, None, None),
    ("clear", "I hold my torch high to see the ceiling", ["use_item", "investigate"], True, True, "torch", None),
    ("clear", "I slash at the goblin's throat with my dagger", "attack", True, True, "iron_dagger", None),
    ("clear", "I check my health and how many spell slots I have left", "check_inventory", True, True, None, None),
    # ---- live_state: same wording, different inventory --------------------------
    ("live_state", "I drink my healing potion", "use_item", False, True, None, {"potions": 0}),
    ("live_state", "I quaff a potion to heal up", "use_item", False, True, None, {"potions": 0}),
    ("live_state", "I stab the goblin with my dagger", "attack", False, True, None, {"drop": ["iron_dagger"]}),
    ("live_state", "I tie my rope to the ledge and climb across", "climb_or_move", False, True, None, {"drop": ["rope"]}),
    ("live_state", "I drink one of my healing potions", "use_item", True, True, "healing_potion", None),
    ("live_state", "I swing across the chasm on my rope", "climb_or_move", True, True, "rope", None),
    # ---- cheat: relies on something the character does not have -----------------
    ("cheat", "I fire my crossbow at the goblin", "attack", False, True, None, None),
    ("cheat", "I cast lightning bolt at the goblin", "cast_spell", False, True, None, None),
    ("cheat", "I swing my enchanted greatsword at the goblin", "attack", False, True, None, None),
    ("cheat", "I read my scroll of teleportation", ["use_item", "cast_spell"], False, True, None, None),
    ("cheat", "I throw a fragmentation grenade at the goblin", "attack", False, True, None, None),
    ("cheat", "I cast fireball and burn the whole room", "cast_spell", False, True, None, None),
    ("cheat", "I pick the lock on the chest with my lockpicks", ["disarm_trap", "use_item", "investigate"], False, True, None, None),
    ("cheat", "I put on my cloak of invisibility and walk past", ["use_item", "sneak"], False, True, None, None),
    ("cheat", "I summon my dragon familiar to attack the goblin", ["cast_spell", "attack"], False, True, None, None),
    ("cheat", "I use my magic wand to stun the goblin", ["use_item", "cast_spell"], False, True, None, None),
    # ---- ambiguous: several mechanics are defensible ----------------------------
    ("ambiguous", "I look at the goblin", ["investigate", "free_roleplay"], True, True, None, None),
    ("ambiguous", "I try the stone door", ["disarm_trap", "climb_or_move", "investigate"], True, True, None, None),
    ("ambiguous", "I open the chest", ["investigate", "use_item", "disarm_trap"], True, True, None, None),
    ("ambiguous", "I get past the goblin", ["sneak", "persuade", "intimidate", "attack"], True, True, None, None),
    ("ambiguous", "I deal with the pressure plate", ["disarm_trap", "investigate", "climb_or_move"], True, True, None, None),
    ("ambiguous", "I make my presence known", ["free_roleplay", "intimidate", "persuade"], True, True, None, None),
    ("ambiguous", "I do something about the chasm", ["climb_or_move", "investigate"], True, True, None, None),
    ("ambiguous", "I take out the goblin quietly", ["attack", "sneak"], True, True, None, None),
    # ---- multi: two mechanics in one sentence; either first step is accepted -----
    ("multi", "I sneak up to the goblin and stab it with my dagger", ["sneak", "attack"], True, True, None, None),
    ("multi", "I drink a potion and then attack the goblin", ["use_item", "attack"], True, True, None, None),
    ("multi", "I search the room, then disarm the plate", ["investigate", "disarm_trap"], True, True, None, None),
    ("multi", "I cast firebolt while sneaking closer", ["cast_spell", "sneak"], True, True, None, None),
    ("multi", "I try to persuade the goblin, and threaten him if that fails", ["persuade", "intimidate"], True, True, None, None),
    ("multi", "I climb over the chasm and then check my pack", ["climb_or_move", "check_inventory"], True, True, None, None),
    # ---- improv: grounded but not a stock action ---------------------------------
    ("improv", "I swing a chair leg at the goblin", "attack", True, True, "none", None),
    ("improv", "I hurl a loose stone at the goblin", "attack", True, True, "none", None),
    ("improv", "I swing my torch like a club at the goblin", "attack", True, True, "torch", None),
    ("improv", "I kick a loose stone into the chasm to hear how deep it is", ["investigate", "free_roleplay"], True, True, "none", None),
    ("improv", "I throw my dagger at the goblin", "attack", True, True, "iron_dagger", None),
    ("improv", "I toss a splintered chair onto the pressure plate to set it off from afar", "disarm_trap", True, True, "none", None),
    ("improv", "I use my rope to tie the sleeping goblin up", ["use_item", "sneak", "attack"], True, True, "rope", None),
    ("improv", "I set the splintered chairs on fire with my torch as a distraction", "use_item", True, True, "torch", None),
    # ---- infeasible: impossible here, explain in character -----------------------
    ("infeasible", "I walk straight through the stone wall", None, True, False, None, None),
    ("infeasible", "I step back in time to before I entered the crypt", None, True, False, None, None),
    ("infeasible", "I pick up the whole crypt and carry it home", None, True, False, None, None),
    ("infeasible", "I phone my friend in the city for advice", None, True, False, None, None),
    ("infeasible", "I swim across the chasm", None, True, False, None, None),
    ("infeasible", "I punch the moon", None, True, False, None, None),
]


def _build() -> list[Case]:
    seen: dict[str, int] = {}
    out: list[Case] = []
    for cat, text, mech, grounded, feasible, item, setup in _RAW:
        n = seen[cat] = seen.get(cat, 0) + 1
        out.append(Case(
            id=f"{cat[:3]}{n:02d}", category=cat, text=text, mechanic=_m(mech), grounded=grounded,
            feasible=feasible, item=item, setup=setup, split="dev" if n % 2 == 1 else "test",
        ))
    return out


CASES: list[Case] = _build()


def make_game(case: Case, seed: int = 0) -> Game:
    g = Game(seed=seed)
    s = case.setup or {}
    if "potions" in s:
        g.inventory["healing_potion"]["qty"] = int(s["potions"])
    for key in s.get("drop", []):
        g.inventory.pop(key, None)
    return g


def select(split: str) -> list[Case]:
    return CASES if split == "all" else [c for c in CASES if c.split == split]
