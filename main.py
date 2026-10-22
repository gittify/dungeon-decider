"""Text RPG game master: Strands Decider routes the action, Bedrock writes the story.

    python main.py                       # Decider server on :8099 + Bedrock narration
    python main.py --no-llm              # Decider only, template narration (no AWS needed)
    python main.py --no-llm --fake-decider   # offline UI/plumbing check (NOT the model)
    python main.py --thresholds eval/thresholds.json
"""

from __future__ import annotations

import argparse
import sys

from decider_patterns import Decider, ServerUnavailable, Thresholds
from decider_patterns.fake import BANNER, FakeDecider
from gm.master import GameMaster
from gm.world import Game

DIM, RESET = "\033[2m", "\033[0m"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="template narration, no Bedrock calls")
    ap.add_argument("--fake-decider", action="store_true", help="keyword stand-in for the model (plumbing only)")
    ap.add_argument("--decider-url", default=None)
    ap.add_argument("--thresholds", default=None, help="JSON from `eval.run_eval tune`")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model-id", default=None, help="Bedrock model id (default: BEDROCK_MODEL_ID)")
    args = ap.parse_args(argv)

    if args.fake_decider:
        decider = FakeDecider()
        print(f"{DIM}{BANNER}{RESET}")
    else:
        decider = Decider(args.decider_url) if args.decider_url else Decider()
        try:
            h = decider.health()
            print(f"{DIM}decider ok: {h.get('model')} on {h.get('device')}{RESET}")
        except ServerUnavailable as exc:
            print(exc, file=sys.stderr)
            return 2

    th = Thresholds.load(args.thresholds) if args.thresholds else None
    gm = GameMaster(Game(seed=args.seed), decider, llm=not args.no_llm, thresholds=th, model_id=args.model_id)
    g = gm.game
    print(f"\n{g.scene}\n{g.character_text()} HP {g.hp}\n")
    print(f"{DIM}Type what you do. /inv  /state  /quit{RESET}")

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/quit", "/q"):
            return 0
        if text == "/inv":
            print(g.character_text(), f"HP {g.hp}")
            continue
        if text == "/state":
            print(g.flags)
            continue
        try:
            res = gm.turn(text)
        except ServerUnavailable as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"\n{res.narration}")
        print(f"{DIM}{res.ticker()}{RESET}")


if __name__ == "__main__":
    raise SystemExit(main())
