# Dungeon Decider: a text-RPG game master where Strands Decider picks the mechanic

The player types free-form actions. **Strands Decider 2B** (local, one request per turn) decides
*which game mechanic* the action is, *which item or spell it leans on*, and *whether it is possible at all*.
Plain code turns those typed numbers into a policy. **Amazon Bedrock** only writes prose, and
arbitrates the turns Decider is unsure about.

tags: `#strands-decider-2b` `#tool-selection`

## Architecture

```
player text
   |  one POST /v1/systemone: three questions over the same encoded state
   v
Decider --> mechanic (11 options) + item_ref (built from LIVE state) + feasible (yes/no)
   |
   v  classify(): ordinary Python over the numbers (decider_patterns/action_gate.py)
   +-- ungrounded  P(item_ref == "unlisted") >= t   -> refuse in character, no mechanic runs
   +-- infeasible  P(feasible) < t                  -> explain in character
   +-- fast        confidence >= FAST               -> run the mechanic in code, narrator decisive
   +-- fast_hedged confidence >= HEDGE              -> run it, narrator hesitant ("confidence as personality")
   +-- escalate    otherwise                        -> Bedrock arbiter Agent with all 11 tools,
                                                       every proposed call checked by ActionGate
   v
narrator (Bedrock, or templates with --no-llm)
```

Division of labour (the design rule): **Decider does fuzzy work** ("the pointy thing" is the dagger;
"my fire spell" is `spell_firebolt`). **Code enforces hard facts** (quantities, spell slots, HP).
A dict lookup for "do I own X" is never Decider's job. The option set is rebuilt from live game
state each turn, so once the last potion is drunk the `healing_potion` option disappears and
"I drink a potion" resolves to `unlisted`.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                   # offline, no model, no AWS
python main.py --no-llm --fake-decider                # UI check with the keyword stand-in (NOT the model)
python main.py --no-llm                               # real Decider, template narration
python main.py                                        # real Decider + Bedrock narration
```

The ticker line under each turn shows Decider latency, tool + confidence, item + P(unlisted),
feasibility, the verdict, and (when used) arbiter/narrator time.

## Reusable pattern: `ActionRouter` + `ActionGate`

Copy `decider_patterns/` into any Strands agent. Two pieces:

**1. A typed question + option set, decided by plain code.**

```python
from decider_patterns import Decider, ActionRouter, Thresholds

router = ActionRouter(Decider("http://127.0.0.1:8099"), Thresholds(fast=0.7, hedge=0.45, ungrounded=0.6))
decision = router.decide(
    text, state_text,                       # state_text = everything the model should see
    tools={"attack": "hit or stab...", "cast_spell": "cast a spell", ...},    # option -> meaning
    refs={"iron_dagger": "the dagger", "none": "no item", "unlisted": "something NOT listed"},
)
decision.verdict   # fast | fast_hedged | escalate | ungrounded | infeasible
```

- `tools` is the one list that is both your tool registry and Decider's option descriptions, so they cannot drift.
- `refs` is **built from live state every call**, with two escape options: `none` and `unlisted`.
  Picking `unlisted` with high probability is the grounding signal.
- `classify(raw, thresholds)` is a pure function: unit-test it, tune it offline from cached answers.
- Use `probabilities["unlisted"]` for the guardrail and the normalised `confidence` for routing; they are different numbers.

**2. The same grounding question as a Strands `InterventionHandler`.**

```python
from strands import Agent
from decider_patterns import ActionGate

gate = ActionGate(decider, state_text=game.state_text, refs=game.item_options, ungrounded=0.6)
agent = Agent(model=model, tools=my_tools, interventions=[gate])
```

`before_tool_call` asks Decider whether the *proposed* call leans on something that is not in the
option set. If so it returns `Guide(...)` (the tool does not run, the model gets feedback); else `Proceed()`.
`fail_open=True` (default) lets calls through if the Decider server is down, which is right only
because the tools themselves also enforce the hard facts. Set `False` if the gate is your only defence.

Differs from the official Decider example (guess a city for `get_weather`) in what is being decided:
*which of N mechanics*, *which owned thing*, with the option set rebuilt from state, not a single
argument-grounding check.

## Evaluation

```bash
python -m eval.run_eval collect --out eval/raw.jsonl                    # real Decider, once per case
python -m eval.run_eval tune    --raw eval/raw.jsonl --out eval/thresholds.json      # DEV split only
python -m eval.run_eval report  --raw eval/raw.jsonl --thresholds eval/thresholds.json --split test --md eval/report_test.md
python -m eval.baseline_llm collect && python -m eval.baseline_llm report --split test    # LLM-only router, same labels
```

60 cases in 7 categories (clear, live_state, cheat, ambiguous, multi, improv, infeasible); within each
category cases alternate dev/test, so thresholds are tuned on 30 and reported on the other 30. That is
small: every figure carries a Wilson 95% interval and the intervals are wide. Say so in the article.
Reports include tool-selection accuracy, grounding recall / false-refusal, feasibility, calibration by
confidence bin, accuracy at confidence >= 0.9 with coverage, and latency median / p95.

## Layout

```
gm/world.py          game state; option set built from live state
gm/tools.py          the 11 deterministic mechanics + the descriptions Decider reads
gm/strands_tools.py  the same mechanics as Strands tools (escalated path)
gm/master.py         GameMaster: router -> fast path | arbiter Agent + ActionGate -> narrator
gm/narrator.py       Bedrock narrator / template narrator
gm/config.py         thresholds and tuning targets (untuned defaults)
decider_patterns/    the reusable part: client, ActionRouter, classify, ActionGate, fake (tests only)
eval/                cases, run_eval (collect/tune/report), baseline_llm
tests/               offline tests (scripted Strands model, keyword fake Decider)
```
