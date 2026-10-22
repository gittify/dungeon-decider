# LLM-only baseline: split=test, n=30  (model us.anthropic.claude-haiku-4-5-20251001-v1:0)

Unparseable replies: 0 (counted as wrong below)

## Tool selection

Overall: 24/27 = 88.9%  (95% CI 72%-96%)

## Grounding check (item == 'unlisted')

- Cheats caught (recall): 7/7 = 100.0%  (95% CI 65%-100%)
- Legitimate actions wrongly refused: 0/23 = 0.0%  (95% CI 0%-14%)
- Overall grounding accuracy: 30/30 = 100.0%  (95% CI 89%-100%)

## Feasibility

- Impossible actions flagged: 3/3 = 100.0%  (95% CI 44%-100%)
- Feasible actions wrongly flagged: 1/20 = 5.0%  (95% CI 1%-24%)

## Latency and tokens (one request per action)

| metric | median | p95 | mean | n |
|---|---|---|---|---|
| wall-clock ms (client -> Bedrock -> client) | 957 | 1150 | 997 | 30 |

Mean tokens per request: input 645, output 30 (price these from the current Bedrock price list; no prices are hard-coded here)

This baseline has no confidence number, so no calibration table or accuracy-at-confidence figure exists for it.
