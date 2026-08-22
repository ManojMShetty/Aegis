"""In-memory provider for offline, free, deterministic tests and dry runs.

WHY A FIRST-CLASS FAKE
----------------------
Most of Aegis's LLM-adjacent logic - role routing, the trust flow around the
quarantine boundary, budget accounting, prompt assembly - has nothing to do with
the network and everything to do with correctness we want to pin down in fast
unit tests. A recorded, scriptable provider lets all of that run with no key, no
quota, and no flakiness, while still exercising the exact same
:class:`~aegis.llm.base.LLMProvider` surface the live client implements.

It does two jobs. First, it *records* every call so a test can assert what the
caller actually sent (system prompt, schema, temperature). Second, it *answers*:
from a queue of canned responses, or - in ``echo_json`` mode - by echoing the
request back as JSON, which is enough to drive a quarantine-style extraction
round-trip without a model.

It also accepts the shared :class:`Budget`, so an offline dry run enforces the
same frugality caps a live run would.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from aegis.llm.base import LLMResponse, Message
from aegis.llm.budget import Budget

__all__ = ["FakeProvider", "RecordedCall"]


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """A faithful snapshot of one :meth:`FakeProvider.complete` invocation."""

    messages: tuple[Message, ...]
    system: str | None
    max_output_tokens: int
    temperature: float | None
    json_schema: Mapping[str, object] | None


class FakeProvider:
    """A scriptable, offline :class:`~aegis.llm.base.LLMProvider`.

    Args:
        responses: Canned answers, consumed in order. Each item may be a ready
            :class:`LLMResponse` or a plain ``str`` (wrapped into one).
        model: Model id stamped onto synthesized responses.
        echo_json: When true and no canned response remains, return the request
            echoed as a JSON object instead of plain text.
        budget: Optional shared budget; when set it is checked and recorded
            exactly as the live provider would, so caps apply offline too.
    """

    def __init__(
        self,
        responses: Iterable[LLMResponse | str] = (),
        *,
        model: str = "fake",
        echo_json: bool = False,
        budget: Budget | None = None,
    ) -> None:
        self._responses: deque[LLMResponse | str] = deque(responses)
        self._model = model
        self._echo_json = echo_json
        self._budget = budget
        self.calls: list[RecordedCall] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        json_schema: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        if self._budget is not None:
            self._budget.check()

        self.calls.append(
            RecordedCall(
                messages=tuple(messages),
                system=system,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                json_schema=json_schema,
            )
        )

        result = self._next_response(messages, json_schema)

        if self._budget is not None:
            self._budget.record(result)
        return result

    # -- queries ---------------------------------------------------------

    @property
    def last_call(self) -> RecordedCall:
        """The most recent recorded call (raises if none happened)."""
        if not self.calls:
            raise IndexError("FakeProvider has recorded no calls yet")
        return self.calls[-1]

    @property
    def remaining(self) -> int:
        """How many canned responses are still queued."""
        return len(self._responses)

    # -- response synthesis ---------------------------------------------

    def _next_response(
        self,
        messages: Sequence[Message],
        json_schema: Mapping[str, object] | None,
    ) -> LLMResponse:
        if self._responses:
            item = self._responses.popleft()
            return item if isinstance(item, LLMResponse) else self._text_response(item)

        last_user = _last_user_text(messages)
        if self._echo_json or json_schema is not None:
            return self._text_response(json.dumps({"echo": last_user}))
        return self._text_response(last_user)

    def _text_response(self, text: str) -> LLMResponse:
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=0,
            output_tokens=_rough_tokens(text),
            finish_reason="STOP",
            raw={"fake": True, "text": text},
        )


def _last_user_text(messages: Sequence[Message]) -> str:
    for msg in reversed(messages):
        if msg.role != "system":
            return msg.content
    return ""


def _rough_tokens(text: str) -> int:
    """A whitespace word count - enough for budget/token assertions offline."""
    return len(text.split())
