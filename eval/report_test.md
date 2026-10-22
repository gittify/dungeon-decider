# Eval report: split=test, n=30

Thresholds: fast=0.0 hedge=0.0 ungrounded=0.3 feasible_min=0.35  (tuned on dev)

## 1. Tool selection (mechanic)

Overall: 25/27 = 92.6%  (95% CI 77%-98%)

| category | accuracy |
|---|---|
| ambiguous | 4/4 = 100.0%  (95% CI 51%-100%) |
| cheat | 5/5 = 100.0%  (95% CI 57%-100%) |
| clear | 7/8 = 87.5%  (95% CI 53%-98%) |
| improv | 3/4 = 75.0%  (95% CI 30%-95%) |
| live_state | 3/3 = 100.0%  (95% CI 44%-100%) |
| multi | 3/3 = 100.0%  (95% CI 44%-100%) |

(ambiguous/multi accept several answers on purpose; see calibration for those.)

## 2. Grounding check (does the action rely on something the character lacks?)

- Cheats caught (recall): 3/7 = 42.9%  (95% CI 16%-75%)
- Legitimate actions wrongly refused (false-refusal rate): 0/23 = 0.0%  (95% CI 0%-14%)
- Precision of refusals: 3/3 = 100.0%  (95% CI 44%-100%)
- Overall grounding accuracy: 26/30 = 86.7%  (95% CI 70%-95%)

| case | category | expected | P(unlisted) | got |
|---|---|---|---|---|
| liv04 | live_state | refuse | 0.14 | allow |
| che02 | cheat | refuse | 0.06 | allow |
| che06 | cheat | refuse | 0.12 | allow |
| che08 | cheat | refuse | 0.17 | allow |

(rows above = grounding mistakes)

## 3. Feasibility check

- Impossible actions flagged: 1/3 = 33.3%  (95% CI 6%-79%)
- Feasible actions wrongly flagged: 0/20 = 0.0%  (95% CI 0%-16%)

## 4. Calibration: mechanic confidence vs. accuracy

| confidence bin | n | mean conf | accuracy |
|---|---|---|---|
| 0.00-0.45 | 3 | 0.26 | 2/3 = 66.7%  (95% CI 21%-94%) |
| 0.45-0.70 | 6 | 0.55 | 5/6 = 83.3%  (95% CI 44%-97%) |
| 0.70-0.90 | 9 | 0.82 | 9/9 = 100.0%  (95% CI 70%-100%) |
| 0.90-1.00 | 9 | 0.94 | 9/9 = 100.0%  (95% CI 70%-100%) |

Accuracy at confidence >= 0.90: 9/9 = 100.0%  (95% CI 70%-100%); coverage 9/27 = 33% of labelled turns

## 5. End-to-end verdicts

Refuse/explain/route decision matches the label: 24/30 = 80.0%  (95% CI 63%-90%)

| verdict | n | share | tool accuracy within band |
|---|---|---|---|
| fast | 26 | 87% | 22/24 = 91.7%  (95% CI 74%-98%) |
| fast_hedged | 0 | 0% | - |
| escalate | 0 | 0% | - |
| ungrounded | 3 | 10% | - |
| infeasible | 1 | 3% | - |

## 6. Decider latency (one request = mechanic + item_ref + feasible)

| metric | median ms | p95 ms | mean ms | n |
|---|---|---|---|---|
| server-reported latency_ms | 526 | 530 | 526 | 30 |
| client wall-clock (incl. HTTP) | 528 | 532 | 528 | 30 |

Mean tokens per request: input 662, output 3

Compare against an LLM-only router with `python -m eval.baseline_llm` (same cases, same labels).
