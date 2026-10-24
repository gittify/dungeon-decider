# Reusable recipe: route free-form input to a tool, with a grounding check

Tags: `#strands-decider-2b` `#tool-selection`

This is the pattern behind Dungeon Decider. It lives in [`decider_patterns/`](../decider_patterns/) and does not depend on the game.

## What the pattern does

A user types anything. One request to a local Strands Decider 2B server answers three typed questions over the same encoded state:

| Question | Type | Answer used for |
|---|---|---|
| Which tool does the input call for? | `choice` over your tool registry | the **route** (via normalised `confidence`) |
| Which known thing does it rely on? | `choice` over options built from live state, plus `none` and `unlisted` | the **grounding check** (via `probabilities["unlisted"]`) |
| Is it possible right now? | `noul` (yes/no probability) | the **feasibility check** |

Plain Python then turns those numbers into one of five verdicts. No prompt is involved in that step.

```
ungrounded   refuse: the input relies on something that is not there
infeasible   explain: it cannot be done in this state
fast         run the tool, respond decisively
fast_hedged  run the tool, respond with hesitation
escalate     hand the turn to an LLM agent with all tools, guarded by ActionGate
```

The division of labour is the design: Decider does the fuzzy reading, your code enforces hard facts (quantities, permissions, state), and an LLM writes prose or arbitrates only the turns Decider is unsure about.

## Two numbers that look alike

- `confidence` on a choice answer is **normalised**: `(N * p_max - 1) / (N - 1)`. Uniform is 0, one-hot is 1, for any number of options. Use it to pick the route.
- `probabilities["unlisted"]` is a **plain probability**. Use it for the grounding check.

Mixing them up is an easy bug. Both were checked against the `strands-decider` 0.2.0 source, and the client parser raises with the raw answer if the response shape changes.

## The one design rule that matters: build the options from live state

The `item_ref` options are rebuilt every call from the current state: one option per thing the user actually has (quantity above zero), plus two escape options.

- `none`: the input relies on no particular item.
- `unlisted`: the input relies on something that is not any other option.

The model never has to decide whether the user owns something. The list already says so. When a potion runs out, its option disappears and the same sentence moves probability toward `unlisted`, with no special case in the code.

## Drop-in wiring

Route one input:

```python
from decider_patterns import Decider, ActionRouter, Thresholds

router = ActionRouter(
    Decider("http://127.0.0.1:8099"),
    Thresholds(fast=0.70, hedge=0.45, ungrounded=0.30, feasible_min=0.35),
)
decision = router.decide(
    text,
    state_text,                                   # scene + user's facts + the input, as a string
    tools={"refund": "customer wants money back", "escalate": "needs a human"},
    refs={"order_1042": "the customer's headphones order",
          "none": "no particular order",
          "unlisted": "an order that is NOT listed above"},
)
decision.verdict      # fast | fast_hedged | escalate | ungrounded | infeasible
decision.tool, decision.tool_conf, decision.raw.p_unlisted
```

The `tools` mapping is both your registry and the option descriptions Decider reads, so the two cannot drift apart. The example tools and orders above are illustrative; supply your own.

Guard an LLM agent's proposed tool calls with the same question:

```python
from strands import Agent
from decider_patterns import ActionGate

gate = ActionGate(decider, state_text=get_state_text, refs=get_options,
                  ungrounded=0.30, fail_open=True)
agent = Agent(model=model, tools=my_tools, interventions=[gate])
```

`before_tool_call` asks Decider the grounding question about the proposed call. At or above the threshold it returns `Guide`, which cancels the call and tells the model why, so it answers in character instead. Otherwise it returns `Proceed`.

## Choosing thresholds

Do not tune by feel, and do not tune on the cases you report. The harness in [`eval/`](../eval/) has three steps:

1. `collect` queries the real model once per case and caches the raw answers.
2. `tune` picks thresholds from the **dev split only**.
3. `report` scores the **held-out test split** with the frozen values, with Wilson intervals.

The thresholds are plain data (`Thresholds.load` / `save`), so you can re-tune offline from cached answers without calling the model again.

## Limits and honest boundaries

- **The grounding check is the weak spot.** On the 30-case test split it caught 3 of 7 made-up items; an LLM-only baseline caught 7 of 7. An unknown spell name tends to snap to the nearest known one. Keep hard facts in code as a second line of defence.
- **Confidence is a routing signal, not a probability of being right.** On 27 labelled test turns, all 18 at 0.70 and above were right and both errors were below 0.70. That is a pattern from a small sample, not a guarantee.
- **The tuner needs enough hard cases.** With all 20 routable dev cases correct it chose `fast = 0` and `hedge = 0`, which switches off the hedged and escalated routes. The live demo used 0.70 and 0.45 instead; those two were not evaluated as a policy on the test split.
- **`fail_open`.** If the Decider server is unreachable the gate lets the call through by default. That is reasonable only when your tools also enforce the hard facts. Set `fail_open=False` if the gate is your only defence.
- **Narration is not controlled by the pattern.** An LLM narrator can contradict the outcome it was handed.
- **Small evaluation.** 60 cases, 30 in test, one labeller, one scene. Intervals are wide.

Full results, including latency and calibration, are in the README and the article. Every number there comes from a run you can repeat with the commands in the README.

## Where this fits beyond a game

A support bot that must not refund an order the customer does not have, a CLI assistant that must not run a command on a file that is not there, or a voice menu that must route a caller before a slow model answers.
