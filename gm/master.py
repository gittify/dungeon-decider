"""GameMaster: Decider routes, code enforces, Bedrock narrates (and arbitrates the unsure turns).

    player text
        |
        v   one Decider request: mechanic + item_ref + feasible          (~100 ms class, local)
    ActionRouter -> verdict (plain code over the numbers)
        |
        +-- ungrounded / infeasible -> refuse or explain in character      (no mechanic runs)
        +-- fast / fast_hedged      -> run the mechanic in code, narrate   (no LLM in the decision)
        +-- escalate                -> Bedrock arbiter Agent with all 11 tools, guarded by ActionGate
        |
        v
    narrator (Bedrock, or templates with llm=False)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from decider_patterns import ActionGate, ActionRouter, Decider, Decision, Thresholds

from . import config
from .narrator import BedrockNarrator, TemplateNarrator
from .strands_tools import make_strands_tools, normalise_ref
from .tools import MECHANIC_CRITERIA, TOOLS
from .world import Game

ARBITER_SYSTEM = """You are the rules arbiter of a dungeon crawl. A fast classifier was unsure which mechanic the player's action needs, so you decide.
Call the mechanic tool(s) that resolve the action (at most two). Pass `item` as one of the keys listed below, or "none".
If the action needs something the character does not have, do not call a tool at all.
Afterwards reply with ONE short sentence describing your ruling. Do not narrate; a narrator does that."""


@dataclass
class TurnResult:
    narration: str
    decision: Decision
    path: str                                   # fast | fast_hedged | escalated | ungrounded | infeasible
    resolutions: list[dict[str, Any]] = field(default_factory=list)
    decide_ms: float = 0.0                      # wall clock around the Decider call
    arbiter_ms: float = 0.0
    narrate_ms: float = 0.0
    total_ms: float = 0.0
    ruling: str = ""
    gate_log: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def ticker(self) -> str:
        parts = [self.decision.ticker()]
        if self.path == "escalated" or self.arbiter_ms:
            parts.append(f"arbiter {self.arbiter_ms:.0f}ms")
        parts.append(f"narrate {self.narrate_ms:.0f}ms")
        parts.append(f"turn {self.total_ms:.0f}ms")
        if self.gate_log:
            parts.append("gate: " + ", ".join(f"{g['tool']}->{g['verdict']}" for g in self.gate_log))
        if self.note:
            parts.append(self.note)
        return " | ".join(parts)


class GameMaster:
    def __init__(
        self,
        game: Game | None = None,
        decider: Decider | Any | None = None,
        *,
        llm: bool = True,
        thresholds: Thresholds | None = None,
        model_id: str | None = None,
    ) -> None:
        self.game = game or Game()
        self.decider = decider or Decider()
        self.llm = llm
        self.th = thresholds or Thresholds(
            config.FAST, config.HEDGE, config.UNGROUNDED, config.FEASIBLE_MIN
        )
        self.router = ActionRouter(self.decider, self.th)
        self.model_id = model_id
        self.narrator = BedrockNarrator(model_id) if llm else TemplateNarrator()
        self._bedrock_model = None  # built lazily, shared by arbiter agents

    # ---- helpers ---------------------------------------------------------------
    def _ref_for_mechanic(self, d: Decision) -> str:
        if d.item in ("none", "unlisted"):
            return ""
        return normalise_ref(self.game, d.item)

    def _run_arbiter(self, text: str, d: Decision) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
        """Bedrock Strands Agent with all tools; ActionGate guards every proposed call."""
        from strands import Agent
        from strands.models import BedrockModel

        if self._bedrock_model is None:
            self._bedrock_model = BedrockModel(
                model_id=self.model_id or config.BEDROCK_MODEL_ID, region_name=config.AWS_REGION
            )
        sink: list[dict[str, Any]] = []
        gate = ActionGate(
            self.decider,
            state_text=self.game.state_text,
            refs=self.game.item_options,
            ungrounded=self.th.ungrounded,
        )
        keys = ", ".join(self.game.item_options())
        agent = Agent(
            model=self._bedrock_model,
            tools=make_strands_tools(self.game, sink),
            interventions=[gate],
            system_prompt=f"{ARBITER_SYSTEM}\nItem keys: {keys}\n{self.game.state_text('(see user message)')}",
            callback_handler=None,
        )
        result = agent(text)
        return sink, str(result).strip(), gate.log

    # ---- the turn --------------------------------------------------------------
    def turn(self, text: str) -> TurnResult:
        g = self.game
        t0 = time.perf_counter()
        d = self.router.decide(text, g.state_text(text), MECHANIC_CRITERIA, g.item_options())
        decide_ms = (time.perf_counter() - t0) * 1000

        resolutions: list[dict[str, Any]] = []
        ruling = ""
        gate_log: list[dict[str, Any]] = []
        arbiter_ms = 0.0
        note = ""

        if d.verdict in ("ungrounded", "infeasible"):
            path = d.verdict
        elif d.verdict in ("fast", "fast_hedged"):
            path = d.verdict
            resolutions.append(TOOLS[d.tool](g, self._ref_for_mechanic(d)))
        else:  # escalate
            if self.llm:
                path = "escalated"
                t1 = time.perf_counter()
                resolutions, ruling, gate_log = self._run_arbiter(text, d)
                arbiter_ms = (time.perf_counter() - t1) * 1000
            else:
                path = "escalated"
                note = "--no-llm: escalation skipped, using Decider's top pick"
                resolutions.append(TOOLS[d.tool](g, self._ref_for_mechanic(d)))

        t2 = time.perf_counter()
        narration = self.narrator.narrate(g, text, d.verdict, d.style, resolutions, ruling)
        narrate_ms = (time.perf_counter() - t2) * 1000

        g.last_resolution = resolutions[-1] if resolutions else None
        return TurnResult(
            narration=narration, decision=d, path=path, resolutions=resolutions,
            decide_ms=decide_ms, arbiter_ms=arbiter_ms, narrate_ms=narrate_ms,
            total_ms=(time.perf_counter() - t0) * 1000, ruling=ruling, gate_log=gate_log, note=note,
        )
