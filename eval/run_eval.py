"""Eval harness: collect -> tune (dev only) -> report (held-out test).

    # 1. query the REAL Decider once per case; raw answers are cached so the next two steps are offline
    python -m eval.run_eval collect --out eval/raw.jsonl
    # 2. choose thresholds from the DEV split only
    python -m eval.run_eval tune --raw eval/raw.jsonl --out eval/thresholds.json
    # 3. report the HELD-OUT test split with those frozen thresholds
    python -m eval.run_eval report --raw eval/raw.jsonl --thresholds eval/thresholds.json --split test --md eval/report_test.md

`--fake` swaps in the keyword stand-in (plumbing check only). Every table it produces carries a
banner saying the numbers are meaningless. Do not publish them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from decider_patterns import ActionRouter, Decider, Raw, Thresholds, classify
from decider_patterns.fake import BANNER, FakeDecider
from gm import config
from gm.tools import MECHANIC_CRITERIA

from .cases import CASES, Case, make_game, select

BINS = [(0.0, 0.45), (0.45, 0.70), (0.70, 0.90), (0.90, 1.0001)]


# ---------------------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def frac(k: int, n: int) -> str:
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%}  (95% CI {lo:.0%}-{hi:.0%})"


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[i]


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_raw(path: str) -> tuple[dict[str, Any], dict[str, tuple[Raw, float]]]:
    meta: dict[str, Any] = {}
    rows: dict[str, tuple[Raw, float]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "_meta" in d:
            meta = d["_meta"]
        else:
            rows[d["id"]] = (Raw.from_json(d["raw"]), float(d["wall_ms"]))
    return meta, rows


def joined(rows: dict[str, tuple[Raw, float]], split: str) -> list[tuple[Case, Raw, float]]:
    cases = select(split)
    missing = [c.id for c in cases if c.id not in rows]
    if missing:
        sys.exit(f"raw file has no answers for: {', '.join(missing[:8])}... rerun `collect`")
    return [(c, rows[c.id][0], rows[c.id][1]) for c in cases]


def tool_ok(c: Case, r: Raw) -> bool | None:
    return None if c.mechanic is None else r.tool in c.mechanic


# ---------------------------------------------------------------------------- collect
def cmd_collect(a: argparse.Namespace) -> int:
    decider: Any = FakeDecider() if a.fake else Decider(a.url)
    meta: dict[str, Any] = {"fake": bool(a.fake), "collected": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "n_cases": len(CASES)}
    if not a.fake:
        meta["health"] = decider.health()
    router = ActionRouter(decider)
    warm = make_game(CASES[0])
    for _ in range(a.warmup):  # first calls pay for compilation / cache fill; do not time them
        router.ask(warm.state_text(CASES[0].text), MECHANIC_CRITERIA, warm.item_options())
    lines = [json.dumps({"_meta": meta})]
    for c in CASES:
        g = make_game(c)
        t0 = time.perf_counter()
        raw = router.ask(g.state_text(c.text), MECHANIC_CRITERIA, g.item_options())
        wall = (time.perf_counter() - t0) * 1000
        lines.append(json.dumps({"id": c.id, "raw": raw.to_json(), "wall_ms": round(wall, 2)}))
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(CASES)} answers to {a.out}" + ("   " + BANNER if a.fake else ""))
    return 0


# ---------------------------------------------------------------------------- tune (dev only)
def _grid(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def tune_thresholds(data: list[tuple[Case, Raw, float]]) -> tuple[Thresholds, list[str]]:
    notes: list[str] = []
    th = Thresholds()

    # grounding: maximise cheats caught subject to a false-refusal cap on legitimate actions
    cheats = [r for c, r, _ in data if not c.grounded]
    legit = [r for c, r, _ in data if c.grounded]
    best = None
    for t in _grid(0.05, 0.95, 0.05):
        rec = sum(r.p_unlisted >= t for r in cheats) / max(1, len(cheats))
        fpr = sum(r.p_unlisted >= t for r in legit) / max(1, len(legit))
        if fpr <= config.MAX_FALSE_REFUSAL and (best is None or (rec, t) > (best[0], best[1])):
            best = (rec, t, fpr)
    if best is None:
        th.ungrounded = 0.95
        notes.append("ungrounded: no threshold met the false-refusal cap; set to 0.95 -- inspect the data")
    else:
        th.ungrounded = best[1]
        notes.append(f"ungrounded={best[1]}: catches {best[0]:.0%} of {len(cheats)} dev cheats, "
                     f"false-refusal {best[2]:.0%} of {len(legit)} dev legit actions")

    # feasibility (judged on grounded cases only, so cheats do not pollute it)
    imp = [r for c, r, _ in data if c.grounded and not c.feasible]
    pos = [r for c, r, _ in data if c.grounded and c.feasible]
    best = None
    for t in _grid(0.05, 0.60, 0.05):
        rec = sum(r.feasible_p < t for r in imp) / max(1, len(imp))
        fpr = sum(r.feasible_p < t for r in pos) / max(1, len(pos))
        if fpr <= config.MAX_FALSE_REFUSAL and (best is None or (rec, -t) > (best[0], -best[1])):
            best = (rec, t, fpr)
    if best is None:
        th.feasible_min = 0.05
        notes.append("feasible_min: no threshold met the false-refusal cap; set to 0.05")
    else:
        th.feasible_min = best[1]
        notes.append(f"feasible_min={best[1]}: flags {best[0]:.0%} of {len(imp)} dev impossible actions, "
                     f"false-flag {best[2]:.0%} of {len(pos)} feasible")

    # mechanic confidence bands, on routable cases (grounded, feasible, labelled)
    routable = [(r.tool_conf, r.tool in c.mechanic) for c, r, _ in data
                if c.mechanic and c.grounded and c.feasible]

    def smallest(target: float) -> float | None:
        for t in _grid(0.0, 0.99, 0.01):
            sel = [ok for conf, ok in routable if conf >= t]
            if len(sel) >= 3 and sum(sel) / len(sel) >= target:
                return t
        return None

    f = smallest(config.TARGET_FAST_ACC)
    h = smallest(config.TARGET_HEDGE_ACC)
    if f is None:
        th.fast = 1.01
        notes.append(f"fast: no confidence level reaches {config.TARGET_FAST_ACC:.0%} accuracy on dev; "
                     "set to 1.01 (never decisive) -- every turn will be hedged or escalated")
    else:
        th.fast = f
        notes.append(f"fast={f}: accuracy at or above it >= {config.TARGET_FAST_ACC:.0%} on dev")
    th.hedge = min(h if h is not None else th.fast, th.fast)
    if th.hedge < config.HEDGE_FLOOR:
        notes.append(f"hedge raised from {th.hedge} to HEDGE_FLOOR={config.HEDGE_FLOOR} "
                     "(policy: keep the escalation path reachable)")
        th.hedge = min(config.HEDGE_FLOOR, th.fast)
    notes.append(f"hedge={th.hedge}: accuracy at or above it >= {config.TARGET_HEDGE_ACC:.0%} on dev"
                 if h is not None else f"hedge={th.hedge}: no level reached {config.TARGET_HEDGE_ACC:.0%}")
    n_dev = len(data)
    if n_dev < 40:
        notes.append(f"WARNING: only {n_dev} dev cases; thresholds will be noisy")
    return th, notes


def cmd_tune(a: argparse.Namespace) -> int:
    meta, rows = load_raw(a.raw)
    data = joined(rows, "dev")
    th, notes = tune_thresholds(data)
    th.save(a.out, tuned_on="dev", n_dev=len(data), fake=meta.get("fake", False), targets={
        "fast_acc": config.TARGET_FAST_ACC, "hedge_acc": config.TARGET_HEDGE_ACC,
        "max_false_refusal": config.MAX_FALSE_REFUSAL})
    if meta.get("fake"):
        print(BANNER)
    print("\n".join(notes))
    print(f"wrote {a.out}")
    return 0


# ---------------------------------------------------------------------------- report
def build_report(meta: dict[str, Any], data: list[tuple[Case, Raw, float]], th: Thresholds,
                 split: str, tuned_note: str) -> str:
    out: list[str] = []
    if meta.get("fake"):
        out.append(f"**{BANNER}**\n")
    out.append(f"# Eval report: split={split}, n={len(data)}\n")
    out.append(f"Thresholds: fast={th.fast} hedge={th.hedge} ungrounded={th.ungrounded} "
               f"feasible_min={th.feasible_min}  ({tuned_note})\n")

    # 1. tool selection
    lab = [(c, r) for c, r, _ in data if c.mechanic]
    k = sum(r.tool in c.mechanic for c, r in lab)
    out.append("## 1. Tool selection (mechanic)\n")
    out.append(f"Overall: {frac(k, len(lab))}\n")
    by: dict[str, list[bool]] = defaultdict(list)
    for c, r in lab:
        by[c.category].append(r.tool in c.mechanic)
    out.append(table(["category", "accuracy"], [[cat, frac(sum(v), len(v))] for cat, v in sorted(by.items())]))
    out.append("\n(ambiguous/multi accept several answers on purpose; see calibration for those.)\n")

    # 2. grounding
    out.append("## 2. Grounding check (does the action rely on something the character lacks?)\n")
    tp = sum(1 for c, r, _ in data if not c.grounded and classify_ground(r, th))
    fn = sum(1 for c, r, _ in data if not c.grounded and not classify_ground(r, th))
    fp = sum(1 for c, r, _ in data if c.grounded and classify_ground(r, th))
    tn = sum(1 for c, r, _ in data if c.grounded and not classify_ground(r, th))
    out.append(f"- Cheats caught (recall): {frac(tp, tp + fn)}")
    out.append(f"- Legitimate actions wrongly refused (false-refusal rate): {frac(fp, fp + tn)}")
    out.append(f"- Precision of refusals: {frac(tp, tp + fp)}")
    out.append(f"- Overall grounding accuracy: {frac(tp + tn, tp + fn + fp + tn)}\n")
    out.append(table(["case", "category", "expected", "P(unlisted)", "got"],
                     [[c.id, c.category, "refuse" if not c.grounded else "allow", f"{r.p_unlisted:.2f}",
                       "refuse" if classify_ground(r, th) else "allow"]
                      for c, r, _ in data if classify_ground(r, th) == c.grounded]))
    out.append("\n(rows above = grounding mistakes)\n")

    # 3. feasibility
    g = [(c, r) for c, r, _ in data if c.grounded]
    imp = [(c, r) for c, r in g if not c.feasible]
    ok = [(c, r) for c, r in g if c.feasible]
    out.append("## 3. Feasibility check\n")
    out.append(f"- Impossible actions flagged: {frac(sum(r.feasible_p < th.feasible_min for _, r in imp), len(imp))}")
    out.append(f"- Feasible actions wrongly flagged: {frac(sum(r.feasible_p < th.feasible_min for _, r in ok), len(ok))}\n")

    # 4. calibration
    out.append("## 4. Calibration: mechanic confidence vs. accuracy\n")
    rows = []
    for lo, hi in BINS:
        sel = [(c, r) for c, r in lab if lo <= r.tool_conf < hi]
        kk = sum(r.tool in c.mechanic for c, r in sel)
        mc = statistics.mean(r.tool_conf for _, r in sel) if sel else float("nan")
        rows.append([f"{lo:.2f}-{min(hi, 1.0):.2f}", len(sel), f"{mc:.2f}" if sel else "-",
                     frac(kk, len(sel)) if sel else "-"])
    out.append(table(["confidence bin", "n", "mean conf", "accuracy"], rows))
    hi_sel = [(c, r) for c, r in lab if r.tool_conf >= 0.9]
    out.append(f"\nAccuracy at confidence >= 0.90: {frac(sum(r.tool in c.mechanic for c, r in hi_sel), len(hi_sel))}; "
               f"coverage {len(hi_sel)}/{len(lab)} = {len(hi_sel) / max(1, len(lab)):.0%} of labelled turns\n")

    # 5. end-to-end verdicts
    out.append("## 5. End-to-end verdicts\n")
    verdicts = [(c, classify(r, th), r) for c, r, _ in data]

    def coarse(v: str) -> str:
        return v if v in ("ungrounded", "infeasible") else "routed"

    agree = sum(coarse(v) == c.expected_verdict() for c, v, _ in verdicts)
    out.append(f"Refuse/explain/route decision matches the label: {frac(agree, len(verdicts))}\n")
    rows = []
    for v in ("fast", "fast_hedged", "escalate", "ungrounded", "infeasible"):
        sel = [(c, r) for c, vv, r in verdicts if vv == v]
        routed = [(c, r) for c, r in sel if c.mechanic and v in ("fast", "fast_hedged", "escalate")]
        acc = frac(sum(r.tool in c.mechanic for c, r in routed), len(routed)) if routed else "-"
        rows.append([v, len(sel), f"{len(sel) / max(1, len(verdicts)):.0%}", acc])
    out.append(table(["verdict", "n", "share", "tool accuracy within band"], rows))
    out.append("")

    # 6. latency / tokens
    srv = [r.latency_ms for _, r, _ in data if r.latency_ms is not None]
    wall = [w for _, _, w in data]
    out.append("## 6. Decider latency (one request = mechanic + item_ref + feasible)\n")
    out.append(table(["metric", "median ms", "p95 ms", "mean ms", "n"], [
        ["server-reported latency_ms", f"{statistics.median(srv):.0f}", f"{pct(srv, .95):.0f}", f"{statistics.mean(srv):.0f}", len(srv)],
        ["client wall-clock (incl. HTTP)", f"{statistics.median(wall):.0f}", f"{pct(wall, .95):.0f}", f"{statistics.mean(wall):.0f}", len(wall)],
    ]))
    toks = [r.usage for _, r, _ in data if r.usage]
    if toks:
        out.append(f"\nMean tokens per request: input {statistics.mean(u['input_tokens'] for u in toks):.0f}, "
                   f"output {statistics.mean(u['output_tokens'] for u in toks):.0f}")
    out.append("\nCompare against an LLM-only router with `python -m eval.baseline_llm` (same cases, same labels).")
    return "\n".join(out) + "\n"


def classify_ground(r: Raw, th: Thresholds) -> bool:
    return r.p_unlisted >= th.ungrounded


def cmd_report(a: argparse.Namespace) -> int:
    meta, rows = load_raw(a.raw)
    data = joined(rows, a.split)
    if a.thresholds:
        th = Thresholds.load(a.thresholds)
        tuned = json.loads(Path(a.thresholds).read_text()).get("tuned_on")
        note = f"tuned on {tuned}" if tuned else "loaded from file, provenance unknown"
        if a.split in ("test", "all") and tuned != "dev":
            print("WARNING: these thresholds were not tuned on the dev split; test numbers are not clean", file=sys.stderr)
        if a.split == "dev":
            note += "; DEV numbers are optimistic because thresholds were fit here"
    else:
        th = Thresholds(config.FAST, config.HEDGE, config.UNGROUNDED, config.FEASIBLE_MIN)
        note = "UNTUNED DEFAULTS"
    if a.split == "all":
        print("WARNING: split=all mixes the tuning data in; do not quote it as held-out", file=sys.stderr)
    text = build_report(meta, data, th, a.split, note)
    print(text)
    if a.md:
        Path(a.md).write_text(text)
        print(f"wrote {a.md}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", default="eval/raw.jsonl")
    c.add_argument("--url", default=None)
    c.add_argument("--fake", action="store_true")
    c.add_argument("--warmup", type=int, default=3)
    c.set_defaults(fn=cmd_collect)
    t = sub.add_parser("tune")
    t.add_argument("--raw", default="eval/raw.jsonl")
    t.add_argument("--out", default="eval/thresholds.json")
    t.set_defaults(fn=cmd_tune)
    r = sub.add_parser("report")
    r.add_argument("--raw", default="eval/raw.jsonl")
    r.add_argument("--thresholds", default=None)
    r.add_argument("--split", choices=["dev", "test", "all"], default="test")
    r.add_argument("--md", default=None)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
