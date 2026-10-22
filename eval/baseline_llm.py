"""LLM-only baseline: the SAME labelled cases, routed by a Bedrock model instead of Decider.

This is the comparison the hackathon rubric rewards ("measured latency/cost gains"). Run it on
your machine with AWS credentials and Bedrock model access:

    python -m eval.baseline_llm collect --out eval/raw_llm.jsonl      # calls Bedrock once per case
    python -m eval.baseline_llm report  --raw eval/raw_llm.jsonl --split test --md eval/report_llm_test.md

It asks for ONE JSON object per action -- {"mechanic", "item", "feasible"} -- i.e. the same three
questions Decider answers in one request, so the comparison is like for like. The model gets the
same scene text, character text, mechanic descriptions and item keys that Decider gets.

Honest-comparison notes (put these in the article too):
  * The LLM has no confidence number, so there is no calibration table; it either answers or not.
  * Latency here is wall-clock around the whole Agent call from your machine to Bedrock; Decider
    latency in eval.run_eval is the server-reported figure for a local model. State both.
  * Tokens are reported, not dollars: price them from the current Bedrock price list yourself.
  * No numbers from this file are checked in. Every figure comes from your own run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from gm import config
from gm.tools import MECHANIC_CRITERIA

from .cases import CASES, Case, make_game, select
from .run_eval import frac, pct, table

SYSTEM = """You are the rules adjudicator for a text RPG. Read the scene, the character sheet and the player's action, then answer with ONE JSON object and nothing else:
{{"mechanic": <one key from MECHANICS>, "item": <one key from ITEMS>, "feasible": true|false}}

MECHANICS (key: meaning):
{mechanics}

ITEMS (key: meaning). "none" = relies on no particular item or spell; "unlisted" = relies on an item, weapon or spell that is NOT any other option here:
{items}

"feasible" is false only if the action is physically impossible in this scene."""


def _system_for(game: Any) -> str:
    mech = "\n".join(f"- {k}: {v}" for k, v in MECHANIC_CRITERIA.items())
    items = "\n".join(f"- {k}: {v}" for k, v in game.item_options().items())
    return SYSTEM.format(mechanics=mech, items=items)


def parse_llm(text: str) -> dict[str, Any]:
    """Pull the first {...} object out of the model's reply. Raises ValueError if there is none."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    d = json.loads(m.group(0))
    return {"mechanic": str(d["mechanic"]), "item": str(d.get("item", "none")),
            "feasible": bool(d.get("feasible", True))}


def ask_llm(model: Any, case: Case) -> dict[str, Any]:
    from strands import Agent

    g = make_game(case)
    agent = Agent(model=model, system_prompt=_system_for(g), callback_handler=None)
    t0 = time.perf_counter()
    result = agent(g.state_text(case.text))
    wall = (time.perf_counter() - t0) * 1000
    usage = result.metrics.accumulated_usage
    row: dict[str, Any] = {"id": case.id, "wall_ms": round(wall, 2),
                           "input_tokens": usage.get("inputTokens"), "output_tokens": usage.get("outputTokens"),
                           "reply": str(result).strip()}
    try:
        row.update(parse_llm(row["reply"]))
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        row["parse_error"] = str(exc)
    return row


def cmd_collect(a: argparse.Namespace) -> int:
    from strands.models import BedrockModel

    model_id = a.model_id or config.BEDROCK_MODEL_ID
    model = BedrockModel(model_id=model_id, region_name=config.AWS_REGION, temperature=0.0)
    meta = {"model_id": model_id, "region": config.AWS_REGION, "collected": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_cases": len(CASES)}
    for c in CASES[: a.warmup]:  # warm the connection; not recorded
        ask_llm(model, c)
    lines = [json.dumps({"_meta": meta})]
    for c in CASES:
        lines.append(json.dumps(ask_llm(model, c)))
        print(f"{c.id} done", file=sys.stderr)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(CASES)} answers to {a.out}")
    return 0


def build_report(meta: dict[str, Any], rows: dict[str, dict[str, Any]], split: str) -> str:
    cases = select(split)
    missing = [c.id for c in cases if c.id not in rows]
    if missing:
        sys.exit(f"raw file has no answers for: {', '.join(missing[:8])}... rerun `collect`")
    data = [(c, rows[c.id]) for c in cases]
    ok_rows = [(c, r) for c, r in data if "parse_error" not in r]
    out = [f"# LLM-only baseline: split={split}, n={len(data)}  (model {meta.get('model_id')})\n"]
    out.append(f"Unparseable replies: {len(data) - len(ok_rows)} (counted as wrong below)\n")

    lab = [(c, r) for c, r in data if c.mechanic]
    k = sum(("parse_error" not in r) and r["mechanic"] in c.mechanic for c, r in lab)
    out.append("## Tool selection\n")
    out.append(f"Overall: {frac(k, len(lab))}\n")

    def refuses(r: dict[str, Any]) -> bool:
        return "parse_error" not in r and r["item"] == "unlisted"

    tp = sum(1 for c, r in data if not c.grounded and refuses(r))
    fn = sum(1 for c, r in data if not c.grounded and not refuses(r))
    fp = sum(1 for c, r in data if c.grounded and refuses(r))
    tn = sum(1 for c, r in data if c.grounded and not refuses(r))
    out.append("## Grounding check (item == 'unlisted')\n")
    out.append(f"- Cheats caught (recall): {frac(tp, tp + fn)}")
    out.append(f"- Legitimate actions wrongly refused: {frac(fp, fp + tn)}")
    out.append(f"- Overall grounding accuracy: {frac(tp + tn, tp + fn + fp + tn)}\n")

    g = [(c, r) for c, r in data if c.grounded]
    imp = [(c, r) for c, r in g if not c.feasible]
    pos = [(c, r) for c, r in g if c.feasible]
    out.append("## Feasibility\n")
    out.append(f"- Impossible actions flagged: {frac(sum(('parse_error' not in r) and not r['feasible'] for _, r in imp), len(imp))}")
    out.append(f"- Feasible actions wrongly flagged: {frac(sum(('parse_error' not in r) and not r['feasible'] for _, r in pos), len(pos))}\n")

    wall = [r["wall_ms"] for _, r in data]
    out.append("## Latency and tokens (one request per action)\n")
    out.append(table(["metric", "median", "p95", "mean", "n"],
                     [["wall-clock ms (client -> Bedrock -> client)", f"{statistics.median(wall):.0f}",
                       f"{pct(wall, .95):.0f}", f"{statistics.mean(wall):.0f}", len(wall)]]))
    tin = [r["input_tokens"] for _, r in data if r.get("input_tokens") is not None]
    tout = [r["output_tokens"] for _, r in data if r.get("output_tokens") is not None]
    if tin:
        out.append(f"\nMean tokens per request: input {statistics.mean(tin):.0f}, output {statistics.mean(tout):.0f} "
                   "(price these from the current Bedrock price list; no prices are hard-coded here)")
    out.append("\nThis baseline has no confidence number, so no calibration table or accuracy-at-confidence figure exists for it.")
    return "\n".join(out) + "\n"


def cmd_report(a: argparse.Namespace) -> int:
    meta: dict[str, Any] = {}
    rows: dict[str, dict[str, Any]] = {}
    for line in Path(a.raw).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            else:
                rows[d["id"]] = d
    text = build_report(meta, rows, a.split)
    print(text)
    if a.md:
        Path(a.md).write_text(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", default="eval/raw_llm.jsonl")
    c.add_argument("--model-id", default=None)
    c.add_argument("--warmup", type=int, default=2)
    c.set_defaults(fn=cmd_collect)
    r = sub.add_parser("report")
    r.add_argument("--raw", default="eval/raw_llm.jsonl")
    r.add_argument("--split", choices=["dev", "test", "all"], default="test")
    r.add_argument("--md", default=None)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
