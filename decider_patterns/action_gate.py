"""REUSABLE DECISION PATTERN: free-form input -> tool, plus a grounding check.

Two pieces you can drop into any Strands agent:

1. `ActionRouter.decide(text, state_text, tools, refs)`
   One Decider request, three typed questions about the same state:
     - mechanic : which of N tools does this input call for?        (choice)
     - item_ref : which concrete thing does it lean on?             (choice, options built
                  from LIVE state, with "none" and "unlisted" escape options)
     - feasible : is it possible at all in this situation?          (noul)
   Returns a `Decision` whose `verdict` is decided by plain code over the numbers
   (`classify`, a pure function you can unit-test and tune offline).

2. `ActionGate(InterventionHandler)`
   The same grounding question wired into Strands' `before_tool_call`, so it also guards
   tool calls an LLM proposes on the slow path. Returns `Guide` (cancel the call and send the
   model back with feedback) or `Proceed`.

Division of labour: Decider resolves *fuzzy references* ("the pointy thing", "my fire
spell"); code enforces *hard facts* (quantities, slots, HP). Decider never does a lookup
that a dict could do.

What the numbers mean (verified in the strands-decider source):
  * `tool_conf` / `item_conf` are Decider's *normalised* confidence, (N*p_max-1)/(N-1).
  * `item_probs["unlisted"]` is a real probability and is what the grounding check uses.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .client import Decider, parse_choice

try:  # Strands is only needed for ActionGate; the router works without it.
    from strands.interventions import Guide, InterventionHandler, Proceed
except ImportError:  # pragma: no cover
    Guide = Proceed = None  # type: ignore[assignment]
    InterventionHandler = object  # type: ignore[assignment,misc]

UNLISTED = "unlisted"


@dataclass
class Thresholds:
    fast: float = 0.70
    hedge: float = 0.45
    ungrounded: float = 0.60
    feasible_min: float = 0.35

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        data = json.loads(Path(path).read_text())
        data = data.get("thresholds", data)
        return cls(**{k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path, **extra: Any) -> None:
        Path(path).write_text(json.dumps({"thresholds": asdict(self), **extra}, indent=2) + "\n")


@dataclass
class Raw:
    """Everything Decider said about one turn, before any policy is applied."""

    tool: str
    tool_conf: float
    tool_probs: dict[str, float]
    item: str
    item_conf: float
    item_probs: dict[str, float]
    feasible_p: float
    latency_ms: float | None = None
    usage: dict[str, int] | None = None

    @property
    def p_unlisted(self) -> float:
        return self.item_probs.get(UNLISTED, 0.0)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Raw":
        return cls(**d)


def classify(raw: Raw, th: Thresholds) -> str:
    """Ordinary control flow over typed numbers: the model classifies, the code decides.

    Order matters: a made-up item is refused before anything else; an impossible action is
    explained before it is routed; only then does confidence pick the path.
    """
    if raw.p_unlisted >= th.ungrounded:
        return "ungrounded"
    if raw.feasible_p < th.feasible_min:
        return "infeasible"
    if raw.tool_conf >= th.fast:
        return "fast"
    if raw.tool_conf >= th.hedge:
        return "fast_hedged"
    return "escalate"


@dataclass
class Decision:
    verdict: str                 # fast | fast_hedged | escalate | ungrounded | infeasible
    raw: Raw

    # convenience passthroughs
    @property
    def tool(self) -> str:
        return self.raw.tool
    @property
    def tool_conf(self) -> float:
        return self.raw.tool_conf
    @property
    def item(self) -> str:
        return self.raw.item
    @property
    def item_conf(self) -> float:
        return self.raw.item_conf
    @property
    def feasible_p(self) -> float:
        return self.raw.feasible_p
    @property
    def latency_ms(self) -> float | None:
        return self.raw.latency_ms

    @property
    def style(self) -> str:
        """Narration hint: confidence shown as temperament."""
        return {"fast": "decisive", "fast_hedged": "hesitant"}.get(self.verdict, "neutral")

    def ticker(self) -> str:
        ms = f"{self.latency_ms:.0f}ms" if self.latency_ms is not None else "?ms"
        r = self.raw
        return (
            f"[decider {ms}] tool={r.tool} {r.tool_conf:.2f} | "
            f"item={r.item} {r.item_conf:.2f} (P(unlisted)={r.p_unlisted:.2f}) | "
            f"feasible={r.feasible_p:.2f} -> {self.verdict}"
        )


class ActionRouter:
    def __init__(self, decider: Decider, th: Thresholds | None = None) -> None:
        self.decider = decider
        self.th = th or Thresholds()

    @staticmethod
    def questions(tools: Mapping[str, str], refs: Mapping[str, str]) -> dict[str, dict[str, Any]]:
        return {
            "mechanic": Decider.choice(
                "Which game mechanic does the player's action call for?", tools
            ),
            "item_ref": Decider.choice(
                "Which of the character's items or spells does the action rely on?", refs
            ),
            "feasible": Decider.noul(
                "Is the player's action physically possible in the current scene?",
                {
                    "true": "the action could plausibly be attempted given the scene and the character",
                    "false": "the action is impossible here, e.g. reaching something far away or breaking the setting's rules",
                },
            ),
        }

    def ask(
        self, state_text: str, tools: Mapping[str, str], refs: Mapping[str, str]
    ) -> Raw:
        """One Decider request, no policy applied. Cache these to tune thresholds offline."""
        answers = self.decider.ask(state_text, self.questions(tools, refs))
        tool, tool_conf, tool_probs = parse_choice(answers["mechanic"], list(tools))
        item, item_conf, item_probs = parse_choice(answers["item_ref"], list(refs))
        return Raw(
            tool, tool_conf, tool_probs, item, item_conf, item_probs,
            feasible_p=float(answers["feasible"]["noul"]),
            latency_ms=self.decider.last_latency_ms,
            usage=self.decider.last_usage,
        )

    def decide(
        self, text: str, state_text: str, tools: Mapping[str, str], refs: Mapping[str, str]
    ) -> Decision:
        raw = self.ask(state_text, tools, refs)
        return Decision(classify(raw, self.th), raw)


class ActionGate(InterventionHandler):  # type: ignore[misc,valid-type]
    """Grounding guardrail for LLM-proposed tool calls (the slow / escalated path).

    Wire it with `Agent(tools=[...], interventions=[ActionGate(...)])`.

    `fail_open`: if the Decider server is unreachable, let the call through (True) or block it
    (False). Fail-open is right when hard facts are also enforced in the tools themselves, as
    they are here; use False when the gate is your only line of defence.
    """

    name = "action-gate"

    def __init__(
        self,
        decider: Decider,
        state_text: Callable[[str], str],
        refs: Callable[[], Mapping[str, str]],
        ungrounded: float = 0.60,
        fail_open: bool = True,
    ) -> None:
        self._decider = decider
        self._state_text = state_text
        self._refs = refs
        self._ungrounded = ungrounded
        self._fail_open = fail_open
        self.log: list[dict[str, Any]] = []

    @property
    def on_error(self) -> str:
        return "proceed" if self._fail_open else "deny"

    @staticmethod
    def _last_user_text(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                for block in message.get("content", []):
                    if isinstance(block, dict) and "text" in block:
                        return block["text"]
        return ""

    def before_tool_call(self, event, **kwargs):
        player_text = self._last_user_text(event.agent.messages)
        refs = self._refs()
        q = {"item_ref": Decider.choice(
            "Which of the character's items or spells does the action rely on?", refs)}
        use = event.tool_use
        state = (
            f"{self._state_text(player_text)}\nPROPOSED TOOL CALL: "
            f"{use['name']}({json.dumps(use.get('input', {}))})"
        )
        item, conf, probs = parse_choice(self._decider.ask(state, q)["item_ref"], list(refs))
        p_unlisted = probs.get(UNLISTED, 0.0)
        bad = p_unlisted >= self._ungrounded
        self.log.append({"tool": use["name"], "item": item, "conf": round(conf, 2),
                         "p_unlisted": round(p_unlisted, 2),
                         "verdict": "guided" if bad else "proceeded"})
        if bad:
            return Guide(
                feedback=(
                    "The player relies on an item or spell the character does not have. Do not run "
                    "this tool. Tell the player, in character, that they come up empty-handed."),
                reason=f"P(unlisted)={p_unlisted:.2f}",
            )
        return Proceed()
