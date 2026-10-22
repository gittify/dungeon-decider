"""Policy knobs. The model classifies; these numbers (owned by the code) decide.

The defaults below are UNTUNED guesses. Tune them on the dev split only:

    python -m eval.run_eval tune --raw eval/raw.jsonl     # writes eval/thresholds.json
    python main.py --thresholds eval/thresholds.json

then report held-out numbers with `python -m eval.run_eval report --split test`.
"""

import os

# Bedrock model that writes narration and arbitrates low-confidence turns.
# Any Strands model id works; this default is the one used in the strands-decider example.
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

# mechanic confidence bands (Decider's normalised confidence)
FAST = 0.70      # >= FAST: resolve instantly, narrator is decisive
HEDGE = 0.45     # HEDGE..FAST: resolve instantly, narrator hedges; < HEDGE: escalate to LLM arbiter

# guardrail: P(item_ref == "unlisted") at/above this -> refuse in character
UNGROUNDED = 0.60

# feasibility: P(true) below this -> the action is physically impossible in the scene
FEASIBLE_MIN = 0.35

# `eval.run_eval tune` targets (dev split only)
TARGET_FAST_ACC = 0.95     # accuracy required at or above the chosen FAST threshold
TARGET_HEDGE_ACC = 0.80    # accuracy required at or above the chosen HEDGE threshold
MAX_FALSE_REFUSAL = 0.05   # max share of legitimate actions the guardrails may refuse
HEDGE_FLOOR = 0.25         # tuning never lowers HEDGE below this
