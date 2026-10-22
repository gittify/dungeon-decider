"""ActionGate wired into a real Strands Agent (scripted model, fake Decider)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripted_model import ScriptedModel  # noqa: E402
from strands import Agent  # noqa: E402

from decider_patterns import ActionGate  # noqa: E402
from decider_patterns.fake import FakeDecider  # noqa: E402
from gm.strands_tools import make_strands_tools  # noqa: E402
from gm.world import Game  # noqa: E402


def _run(player_text: str, tool: str, args: dict):
    game = Game(seed=3)
    sink: list = []
    gate = ActionGate(FakeDecider(), game.state_text, game.item_options)
    model = ScriptedModel([("tool", tool, args), ("text", "Ruling given.")])
    agent = Agent(model=model, tools=make_strands_tools(game, sink), interventions=[gate],
                  callback_handler=None)
    result = agent(player_text)
    return game, sink, gate, str(result)


def test_gate_blocks_tool_for_item_the_character_lacks():
    game, sink, gate, out = _run("I swing my sword at the goblin", "attack", {"item": "sword"})
    assert sink == [], "tool must not run when the player relies on an unlisted item"
    assert gate.log and gate.log[0]["verdict"] == "guided"
    assert game.flags["goblin_asleep"] is True


def test_gate_lets_grounded_call_through():
    game, sink, gate, out = _run("I drink my healing potion", "use_item", {"item": "healing_potion"})
    assert len(sink) == 1 and sink[0]["tool"] == "use_item"
    assert gate.log[0]["verdict"] == "proceeded"
    assert game.inventory["healing_potion"]["qty"] == 1


def test_gate_fails_open_when_decider_is_down():
    class Down:
        def ask(self, *a, **k):
            from decider_patterns import ServerUnavailable
            raise ServerUnavailable("down")

    game = Game(seed=3)
    sink: list = []
    gate = ActionGate(Down(), game.state_text, game.item_options, fail_open=True)
    model = ScriptedModel([("tool", "check_inventory", {}), ("text", "ok")])
    agent = Agent(model=model, tools=make_strands_tools(game, sink), interventions=[gate], callback_handler=None)
    agent("what do I have")
    assert len(sink) == 1
