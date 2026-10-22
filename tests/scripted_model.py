"""A scripted Strands Model for tests: replays canned assistant turns, no network, no AWS."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from strands.models import Model


class ScriptedModel(Model):
    """Each script entry is either ("tool", name, input_dict) or ("text", str)."""

    def __init__(self, script: list[tuple]) -> None:
        self.script = list(script)
        self.calls = 0

    def update_config(self, **model_config: Any) -> None: ...

    def get_config(self) -> Any:
        return {}

    def structured_output(self, output_model, prompt, system_prompt=None, **kwargs) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs: Any) -> AsyncIterable[dict[str, Any]]:
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        if step[0] == "tool":
            _, name, args = step
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self.calls}", "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": step[1]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}
