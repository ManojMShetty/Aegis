"""Gemini3LLM against a fake genai client - no network, no key, no spend.

The load-bearing test here is :func:`test_second_turn_echoes_original_signature`:
it drives a two-turn tool loop through a stub client and proves that the SECOND
request hands the model back the ORIGINAL function-call ``Content`` - the object
carrying ``thought_signature`` - rather than the bare part AgentDojo's stock
converter would rebuild. That is the whole reason this element exists, so it is
proven directly, offline, against a signature no rebuild could reproduce.

The remaining tests pin the three bugs a prior review found: the fingerprint must
survive AgentDojo's in-place args mutation (fix 1), two identical call turns must
not collapse onto one signature and the stash must clear between conversations
(fix 2), and a transient 5xx must be retried with every attempt counted while a
deterministic 4xx is not retried (fix 3).
"""

from __future__ import annotations

from ast import literal_eval
from types import SimpleNamespace
from typing import Any

import pytest
from agentdojo.agent_pipeline.llms.google_llm import _message_to_google
from agentdojo.agent_pipeline.tool_execution import is_string_list
from agentdojo.functions_runtime import FunctionCall
from evals.agentdojo.gemini_llm import Gemini3LLM, Gemini3LLMError
from google.genai import types
from google.genai.errors import ClientError, ServerError
from tenacity import wait_none

# An opaque per-call signature the model would emit on a tool-call turn. No bare
# rebuild from (name, args) can reproduce it, so finding it on the wire is proof
# the original part was echoed verbatim.
SIGNATURE = b"THOUGHT-SIGNATURE-3x"


def _model_content_with_tool_call() -> types.Content:
    """The genai Content a Gemini 3.x tool-call turn returns, signature attached."""
    return types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="get_day", args={}, id="call_1"),
                thought_signature=SIGNATURE,
            )
        ],
    )


def _model_content_final_text() -> types.Content:
    return types.Content(role="model", parts=[types.Part(text="Today is Monday.")])


def _response_for(content: types.Content) -> Any:
    """Wrap a Content the way genai's response exposes it (candidates[0].content)."""
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


class FakeClient:
    """Stub genai client: records the contents of each request, returns canned ones."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._responses.pop(0)


def _text_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "content": text}]}


def _tool_result_message(call: FunctionCall, text: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call": call,
        "content": [{"type": "text", "content": text}],
        "tool_call_id": call.id,
        "error": None,
    }


def _empty_runtime() -> Any:
    """A runtime with no tools; the canned response drives the tool call anyway."""
    return SimpleNamespace(functions={})


def _model_role_contents(contents: list[Any]) -> list[Any]:
    return [c for c in contents if getattr(c, "role", None) == "model"]


def test_second_turn_echoes_original_signature() -> None:
    original_content = _model_content_with_tool_call()
    client = FakeClient(
        responses=[
            _response_for(original_content),
            _response_for(_model_content_final_text()),
        ]
    )
    llm = Gemini3LLM("gemini-3.5-flash-lite", client=client)
    runtime = _empty_runtime()

    # -- Turn 1: user asks; model returns a tool call (carrying the signature).
    messages: list[Any] = [
        _text_message("system", "You are a helpful agent."),
        _text_message("user", "What day is it?"),
    ]
    _, _, _, messages, _ = llm.query("What day is it?", runtime, messages=messages)

    assistant = messages[-1]
    assert assistant["role"] == "assistant"
    assert [tc.function for tc in assistant["tool_calls"]] == ["get_day"]

    # The bare rebuild AgentDojo would have produced drops the signature: this is
    # exactly the bug, asserted so the positive check below cannot pass vacuously.
    bare = _message_to_google(assistant)
    assert bare.parts[0].thought_signature is None

    # -- Between turns AgentDojo appends the tool result.
    messages.append(_tool_result_message(assistant["tool_calls"][0], "Monday"))

    # -- Turn 2: the fix must echo the ORIGINAL content back to the model.
    _, _, _, messages, _ = llm.query("What day is it?", runtime, messages=messages)

    second_request_contents = client.calls[1]["contents"]
    echoed = _model_role_contents(second_request_contents)
    assert len(echoed) == 1
    # Same object, byte-identical signature - not a rebuilt bare part.
    assert echoed[0] is original_content
    assert echoed[0].parts[0].thought_signature == SIGNATURE

    # The loop terminated with the model's final text answer.
    assert messages[-1]["content"][0]["content"] == "Today is Monday."


def test_counters_increment_per_turn_when_nothing_is_retried() -> None:
    """With no retry in play the two counters move together: one turn, one attempt.

    They are separate fields precisely because that coincidence does not hold in
    general (see the retry test below), so both are pinned here - a change that
    collapsed them back into one number would have to break this test first.
    """
    client = FakeClient(
        responses=[
            _response_for(_model_content_with_tool_call()),
            _response_for(_model_content_final_text()),
        ]
    )
    llm = Gemini3LLM("gemini-3.5-flash-lite", client=client)
    runtime = _empty_runtime()
    assert llm.call_count == 0
    assert llm.request_count == 0

    messages: list[Any] = [
        _text_message("system", "sys"),
        _text_message("user", "hi"),
    ]
    _, _, _, messages, _ = llm.query("hi", runtime, messages=messages)
    assert llm.call_count == 1
    assert llm.request_count == 1

    messages.append(_tool_result_message(messages[-1]["tool_calls"][0], "Monday"))
    llm.query("hi", runtime, messages=messages)
    assert llm.call_count == 2
    assert llm.request_count == 2


def test_unknown_tool_call_turn_falls_back_without_crashing() -> None:
    """An assistant tool-call turn we never stashed defers to the stock converter.

    This is the miss path: no signature to preserve, so the bare rebuild is the
    correct (and only possible) behaviour - it must not raise.
    """
    client = FakeClient(responses=[_response_for(_model_content_final_text())])
    llm = Gemini3LLM("gemini-3.5-flash-lite", client=client)

    foreign_assistant: dict[str, Any] = {
        "role": "assistant",
        "content": [],
        "tool_calls": [FunctionCall(function="send_email", args={"to": "x"}, id="zzz")],
    }
    messages: list[Any] = [
        _text_message("system", "sys"),
        _text_message("user", "hi"),
        foreign_assistant,
        _tool_result_message(foreign_assistant["tool_calls"][0], "ok"),
    ]
    # Should complete using the fallback conversion for the unknown turn.
    _, _, _, messages, _ = llm.query("hi", _empty_runtime(), messages=messages)
    assert messages[-1]["content"][0]["content"] == "Today is Monday."


# -- fix 1: the fingerprint survives AgentDojo's in-place args mutation ----------


def _model_content_with_list_arg() -> types.Content:
    """A tool-call turn whose args carry a STRING-encoded list, signature attached.

    ``id`` is ``None`` because Gemini usually omits it - the case that makes the
    args value, not the id, load-bearing for the fingerprint.
    """
    return types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    name="send_email",
                    args={"recipients": "['a@x.com', 'b@x.com']"},
                    id=None,
                ),
                thought_signature=SIGNATURE,
            )
        ],
    )


def test_stash_hits_despite_in_place_args_mutation() -> None:
    """AgentDojo's ToolsExecutor rewrites a string-encoded list arg into a real
    list IN PLACE between turns. The fingerprint applies the identical
    normalization, so the stash still hits on turn 2 and the ORIGINAL signed
    Content is echoed - instead of a bare rebuild that would 400 the whole run.
    """
    original = _model_content_with_list_arg()
    client = FakeClient(
        responses=[_response_for(original), _response_for(_model_content_final_text())]
    )
    llm = Gemini3LLM("gemini-3.5-flash-lite", client=client)
    runtime = _empty_runtime()

    messages: list[Any] = [_text_message("system", "sys"), _text_message("user", "email them")]
    _, _, _, messages, _ = llm.query("email them", runtime, messages=messages)

    call = messages[-1]["tool_calls"][0]
    assert isinstance(call.args["recipients"], str)  # still string-encoded on turn 1

    # Reproduce ToolsExecutor's exact in-place normalization of list-typed args.
    for arg_k, arg_v in list(call.args.items()):
        if isinstance(arg_v, str) and is_string_list(arg_v):
            call.args[arg_k] = literal_eval(arg_v)
    assert call.args["recipients"] == ["a@x.com", "b@x.com"]  # now a real list

    messages.append(_tool_result_message(call, "sent"))
    _, _, _, messages, _ = llm.query("email them", runtime, messages=messages)

    echoed = _model_role_contents(client.calls[1]["contents"])
    assert len(echoed) == 1
    # The stash hit despite the mutation: same object, signature intact.
    assert echoed[0] is original
    assert echoed[0].parts[0].thought_signature == SIGNATURE


# -- fix 2: identical call turns stay distinct; stash clears between tasks -------


def _tool_call_content(sig: bytes) -> types.Content:
    """A byte-identical (id=None, empty args) tool-call turn but for its signature."""
    return types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="ping", args={}, id=None),
                thought_signature=sig,
            )
        ],
    )


def test_identical_call_turns_do_not_collapse() -> None:
    """Two byte-identical tool-call turns (Gemini returns id=None) must each echo
    THEIR OWN signature. Keying the stash by conversation position keeps them
    distinct instead of the later one overwriting the earlier.
    """
    sig_a = b"SIGNATURE-A"
    sig_b = b"SIGNATURE-B"
    client = FakeClient(
        responses=[
            _response_for(_tool_call_content(sig_a)),
            _response_for(_tool_call_content(sig_b)),
            _response_for(_model_content_final_text()),
        ]
    )
    llm = Gemini3LLM("gemini-3.5-flash-lite", client=client)
    runtime = _empty_runtime()

    messages: list[Any] = [_text_message("system", "sys"), _text_message("user", "go")]
    _, _, _, messages, _ = llm.query("go", runtime, messages=messages)
    messages.append(_tool_result_message(messages[-1]["tool_calls"][0], "pong"))
    _, _, _, messages, _ = llm.query("go", runtime, messages=messages)
    messages.append(_tool_result_message(messages[-1]["tool_calls"][0], "pong"))
    _, _, _, messages, _ = llm.query("go", runtime, messages=messages)

    echoed = _model_role_contents(client.calls[2]["contents"])
    assert len(echoed) == 2
    # Each history turn kept its own signature - no collapse onto the later one.
    assert echoed[0].parts[0].thought_signature == sig_a
    assert echoed[1].parts[0].thought_signature == sig_b


def test_conversation_start_clears_stale_stash() -> None:
    """A fresh conversation (a history with no assistant turn yet) drops the
    previous task's stash, so a stale signature cannot leak across tasks and the
    dict cannot grow without bound across a whole run.
    """
    client = FakeClient(
        responses=[
            _response_for(_model_content_with_tool_call()),  # conv A, turn 1
            _response_for(_model_content_final_text()),  # conv A, turn 2
            _response_for(_model_content_with_tool_call()),  # conv B, turn 1
        ]
    )
    llm = Gemini3LLM("gemini-3.5-flash-lite", client=client)
    runtime = _empty_runtime()

    msgs_a: list[Any] = [_text_message("system", "sys"), _text_message("user", "a")]
    _, _, _, msgs_a, _ = llm.query("a", runtime, messages=msgs_a)
    msgs_a.append(_tool_result_message(msgs_a[-1]["tool_calls"][0], "Monday"))
    llm.query("a", runtime, messages=msgs_a)
    assert len(llm._original_contents) == 1  # A left one entry

    # Conversation B starts fresh: A's entry must be cleared before B stashes.
    msgs_b: list[Any] = [_text_message("system", "sys"), _text_message("user", "b")]
    llm.query("b", runtime, messages=msgs_b)
    assert len(llm._original_contents) == 1  # only B's entry - A did not accumulate


# -- fix 3: transient 5xx retried and every attempt counted; 4xx not retried ----


class _FlakyGenerate:
    """A generate_content stub that raises `exc` on the first `fail_times` calls."""

    def __init__(self, exc: Exception, fail_times: int, ok_response: Any) -> None:
        self._exc = exc
        self._fail_times = fail_times
        self._ok = ok_response
        self.attempts = 0

    def __call__(self, *, model: str, contents: Any, config: Any) -> Any:
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise self._exc
        return self._ok


def test_transient_server_error_is_retried_and_each_attempt_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient 5xx is retried under the stock policy, and the two counters
    separate: ``request_count`` counts EVERY attempt (each one spent quota), while
    ``call_count`` still records ONE turn, because the agent made one decision.

    This is the case that forces the split. A single field would have to mean one
    or the other, and the runner reports both under distinct names so a run's cost
    (attempts) is never confused with its length (turns). Retrying rather than
    failing also keeps a flaky 5xx from being mis-scored as a successful attack and
    inflating ASR.
    """
    # Instant retries: neutralise the exponential backoff so the test does not sleep.
    monkeypatch.setattr(Gemini3LLM._generate_content.retry, "wait", wait_none())

    flaky = _FlakyGenerate(
        exc=ServerError(503, {"error": {"message": "transient"}}),
        fail_times=2,
        ok_response=_response_for(_model_content_final_text()),
    )
    client = SimpleNamespace(models=SimpleNamespace(generate_content=flaky))
    llm = Gemini3LLM("gemini-3.5-flash-lite", client=client)

    messages: list[Any] = [_text_message("system", "sys"), _text_message("user", "hi")]
    _, _, _, messages, _ = llm.query("hi", _empty_runtime(), messages=messages)

    assert flaky.attempts == 3  # two failures then a success
    assert llm.request_count == 3  # every attempt counted, not just the success
    assert llm.call_count == 1  # ... but the agent only made one decision
    assert messages[-1]["content"][0]["content"] == "Today is Monday."


def test_client_error_is_not_retried_and_counted_once() -> None:
    """A deterministic 4xx is NOT retried (retrying a 400 only burns quota): one
    attempt, counted once under both units, and the error propagates.

    The turn is counted even though it produced no answer: a turn whose every
    attempt failed still cost the agent a decision, and the attempt still spent
    quota, so neither counter is allowed to quietly forget it.
    """
    boom = _FlakyGenerate(
        exc=ClientError(400, {"error": {"message": "bad"}}),
        fail_times=99,
        ok_response=_response_for(_model_content_final_text()),
    )
    client = SimpleNamespace(models=SimpleNamespace(generate_content=boom))
    llm = Gemini3LLM("gemini-3.5-flash-lite", client=client)

    messages: list[Any] = [_text_message("system", "sys"), _text_message("user", "hi")]
    with pytest.raises(ClientError):
        llm.query("hi", _empty_runtime(), messages=messages)

    assert boom.attempts == 1  # not retried
    assert llm.request_count == 1
    assert llm.call_count == 1


def test_missing_api_key_raises_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set a realistic-looking key value, then remove the variable the element
    # reads, so the missing-key branch fires. Asserting this concrete value is
    # absent from the message is a real leak check - unlike the old "key="
    # substring, which any message trivially satisfies.
    fake_key = "AIzaSy-FAKE-KEY-VALUE-must-never-leak-0123456789"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(Gemini3LLMError) as exc:
        Gemini3LLM("gemini-3.5-flash-lite")
    message = str(exc.value)
    assert "GEMINI_API_KEY" in message  # the variable NAME guides the user
    assert fake_key not in message  # ... and no key VALUE ever leaks


def test_explicit_client_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = FakeClient(responses=[])
    llm = Gemini3LLM("gemini-3.5-flash-lite", client=client)
    assert llm.client is client
    assert llm.name == "gemini3-gemini-3.5-flash-lite"
