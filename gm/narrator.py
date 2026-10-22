"""Narration. Bedrock writes prose; it never decides outcomes.

`TemplateNarrator` is the --no-llm path: deterministic, instant, needs no AWS credentials.
`BedrockNarrator` is a Strands Agent with no tools -- it gets the authoritative facts and
only phrases them. "Confidence as personality": `style` is 'decisive' for fast turns and
'hesitant' for hedged ones.
"""

from __future__ import annotations

from typing import Any

from . import config

NARRATOR_SYSTEM = """You are the game master of a small fantasy dungeon crawl, narrating in the second person, present tense.
Rules:
- 2-3 sentences, vivid but brief.
- The OUTCOME FACTS are authoritative. Narrate exactly that outcome. Never change a success into a failure, never invent items, spells, damage or enemies that are not in the facts or scene.
- If STYLE is 'decisive', narrate with confident certainty.
- If STYLE is 'hesitant', let the game master sound unsure how to read the player's intent (e.g. 'you seem to be going for...'), then still narrate the stated outcome.
- If the outcome says the player lacks the item or spell, play it as the character reaching for something that is not there. Do not lecture.
- If the outcome says the action is impossible, explain in character why it cannot be done here.
- Do not mention dice, DCs or mechanics; the interface shows those separately."""


def _roll_text(res: dict[str, Any]) -> str:
    r = res.get("roll")
    if not r:
        return ""
    return f" [d20 {r['d20']}{r['mod']:+d} = {r['total']} vs DC {r['dc']}]"


def facts_block(game: Any, player_text: str, verdict: str, style: str, resolutions: list[dict[str, Any]], ruling: str = "") -> str:
    if verdict == "ungrounded":
        outcome = "The player tried to rely on an item or spell the character does not have. Nothing happens."
    elif verdict == "infeasible":
        outcome = "The action is impossible in this scene. Nothing happens."
    elif not resolutions:
        outcome = ruling or "No mechanic applied; treat it as roleplay."
    else:
        outcome = "\n".join(
            f"- {r['tool']}: {'SUCCESS' if r.get('success') else 'FAILURE'}. {r.get('detail', '')}"
            for r in resolutions
        )
    return (
        f"STYLE: {style}\nSCENE: {game.scene}\n{game.character_text()} HP {game.hp}.\n"
        f'PLAYER ACTION: "{player_text}"\nOUTCOME FACTS:\n{outcome}'
        + (f"\nARBITER NOTE: {ruling}" if ruling and resolutions else "")
    )


class TemplateNarrator:
    mode = "template"

    def narrate(self, game: Any, player_text: str, verdict: str, style: str,
                resolutions: list[dict[str, Any]], ruling: str = "") -> str:
        if verdict == "ungrounded":
            have = ", ".join(v["name"] for v in game.inventory.values() if v["qty"] > 0)
            spells = ", ".join(game.spells)
            return (f"Your hand closes on empty air -- you have nothing like that. "
                    f"You carry: {have}. Spells: {spells}.")
        if verdict == "infeasible":
            return "No one could manage that here; the crypt does not bend that way."
        if not resolutions:
            return ruling or "You say your piece; the crypt keeps its own counsel."
        lead = "You seem to be going for it, and " if style == "hesitant" else ""
        parts = []
        for r in resolutions:
            verb = "succeeds" if r.get("success") else "fails"
            parts.append(f"{r['tool'].replace('_', ' ')} {verb}: {r.get('detail', '')}{_roll_text(r)}")
        return lead + " ".join(parts)


class BedrockNarrator:
    mode = "bedrock"

    def __init__(self, model_id: str | None = None, region: str | None = None) -> None:
        from strands import Agent
        from strands.models import BedrockModel

        model = BedrockModel(model_id=model_id or config.BEDROCK_MODEL_ID,
                             region_name=region or config.AWS_REGION)
        self._agent = Agent(model=model, system_prompt=NARRATOR_SYSTEM, callback_handler=None)

    def narrate(self, game: Any, player_text: str, verdict: str, style: str,
                resolutions: list[dict[str, Any]], ruling: str = "") -> str:
        result = self._agent(facts_block(game, player_text, verdict, style, resolutions, ruling))
        return str(result).strip()
