"""Minimal synchronous client for a locally served Strands Decider 2B.

Start the server first (on an Apple-silicon Mac):

    strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099 --device mlx

One request carries one `state` and many typed questions; the state is encoded once and
shared, so asking several questions costs far less than several requests.

Wire format -- VERIFIED against the strands-decider 0.2.0 source (schema.py, server.py), not
guessed:

    POST /v1/systemone
    {"state": str, "model": str, "questions": {name: {"type": "noul"|"choice"|"score", ...}}}
    -> {"model": str, "answers": {name: Answer}, "usage": {input_tokens, output_tokens},
        "latency_ms": float}

    noul   answer: {"type": "noul",   "noul": P(true)}
    choice answer: {"type": "choice", "choice": <option name>,
                    "probabilities": {<option name>: p, ...},
                    "confidence": (N * p_max - 1) / (N - 1)}      # normalised, NOT p_max
    score  answer: {"type": "score", "score": float, "legend": {...},
                    "probabilities": {...}, "confidence": float}

Note `confidence` for `choice` is the *normalised* max-probability: uniform -> 0.0, one-hot
-> 1.0, independent of the number of options. It is not P(choice). Use `probabilities[x]`
when you want "how likely is option x" (the grounding check does).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

DEFAULT_URL = os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:8099")
SERVE_HINT = "strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099 --device mlx"


class ServerUnavailable(RuntimeError):
    pass


class DeciderRequestError(RuntimeError):
    """The server answered but rejected the request (HTTP 4xx/5xx)."""


class Decider:
    def __init__(self, url: str | None = None, timeout: float = 120.0) -> None:
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.calls = 0
        self.last_latency_ms: float | None = None
        self.last_usage: dict[str, int] | None = None

    # ---- question constructors -------------------------------------------------
    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        q: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            q["criteria"] = dict(criteria)
        return q

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    @staticmethod
    def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    # ---- asking ----------------------------------------------------------------
    def ask(self, state: str, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps(
            {"state": state, "model": "strands-decider", "questions": dict(questions)}
        ).encode()
        req = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise DeciderRequestError(f"Decider rejected the request ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise ServerUnavailable(
                f"no Decider server at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}"
            ) from exc
        self.calls += 1
        self.last_latency_ms = payload.get("latency_ms")
        self.last_usage = payload.get("usage")
        return payload["answers"]

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise ServerUnavailable(f"no Decider server at {self.url}: {exc}") from exc


def parse_choice(
    answer: Mapping[str, Any], options: list[str] | None = None
) -> tuple[str, float, dict[str, float]]:
    """Return (chosen option, normalised confidence, per-option probabilities).

    Strict on purpose: the shape is documented in the module docstring, so anything else is a
    version mismatch and should fail loudly with the raw answer in the message.
    """
    try:
        name = str(answer["choice"])
        conf = float(answer["confidence"])
        probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"unexpected choice answer shape: {dict(answer)!r}") from exc
    if options is not None and name not in options:
        raise ValueError(f"choice {name!r} is not one of the offered options {options!r}")
    return name, conf, probs
