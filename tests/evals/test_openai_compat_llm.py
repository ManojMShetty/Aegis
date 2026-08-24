"""CountingOpenAILLM and build_openai_compat_llm - offline, no network, no key.

The default Week-0 path runs AgentDojo's stock ``OpenAILLM`` over an
OpenAI-compatible endpoint, so there is no protocol behaviour to prove here (the
stock element is trusted). What we DO prove is everything this module adds on top
of it, because each addition was born as a live-fire hotfix and none of it is
exercised by the stock element's own tests:

* an accurate per-turn spend counter;
* disciplined key handling - the key is read only from the NAMED environment
  variable (never falling back to another provider's, even when one is sitting
  right there in the environment), the client is built bounded (timeout +
  max_retries=5, the SDK's Retry-After-aware retry for a free-tier 429), and a
  missing key fails with a message that names the variable and never a value;
* the rate throttle, driven by a FAKE CLOCK so the assertion is exact and the test
  suite never actually sleeps - including that its spacing is measured from request
  START, not from the previous response;
* the one contract this module has with the real SDK - that ``chat`` and
  ``completions`` are cached, so a guard installed on them stays installed. Every
  other test substitutes the SDK class, which cannot see that;
* single-tool-call truncation, which must fire for NVIDIA's NIM and must NOT fire
  for Groq (truncating a capable model's parallel tool calls would change the
  baseline's behaviour for no reason);
* reasoning-effort plumbing, including that a bad value is refused BEFORE a client
  exists - i.e. at a cost of zero requests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from evals.agentdojo.openai_llm import (
    DEFAULT_MIN_REQUEST_INTERVAL_S,
    CountingOpenAILLM,
    OpenAICompatLLMError,
    build_openai_compat_llm,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GROQ_MODEL = "openai/gpt-oss-120b"


class _FakeMessage:
    """A ChatCompletionMessage-shaped object: what the SDK exposes at choices[0]."""

    def __init__(self, content: str | None, tool_calls: Any = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeCompletion:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [_FakeChoice(message)]


class _FakeCompletions:
    """Records how many creates it served; returns canned ChatCompletion shapes.

    ``on_create`` runs INSIDE the create the guard wraps, which is the only place a
    simulated request duration is worth spending: it makes fake time elapse between
    the guard's timestamp and the response. A duration simulated in a wrapper placed
    AROUND the guard would elapse before the guard ever stamps, so the pre-request
    and post-response instants would coincide and a stamp-placement mutation would
    be invisible - which is exactly the tautology this hook exists to avoid.
    """

    def __init__(
        self,
        responses: list[_FakeCompletion],
        on_create: Callable[[], None] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._on_create = on_create
        self.creates = 0

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self.creates += 1
        if self._on_create is not None:
            self._on_create()
        return self._responses.pop(0)


class _FakeOpenAIClient:
    """Minimal stand-in for openai.OpenAI: only .chat.completions.create is used."""

    def __init__(
        self,
        responses: list[_FakeCompletion],
        on_create: Callable[[], None] | None = None,
    ) -> None:
        self.completions = _FakeCompletions(responses, on_create)
        self.chat = SimpleNamespace(completions=self.completions)


class _SDKConstructorSpy:
    """A stand-in for the ``openai.OpenAI`` CLASS, recording how it was built.

    ``build_openai_compat_llm`` patches ``chat.completions.create`` on whatever the
    constructor returns, so the instance must carry that shape; the recorded kwargs
    let a test assert the client was built bounded, and ``instances`` lets a test
    assert no client was built at all when configuration was refused up front.
    """

    def __init__(
        self,
        responses: list[_FakeCompletion] | None = None,
        on_create: Callable[[], None] | None = None,
    ) -> None:
        self.kwargs: dict[str, Any] = {}
        self.instances: list[_FakeOpenAIClient] = []
        self._responses = responses if responses is not None else []
        self._on_create = on_create

    def __call__(self, **kwargs: Any) -> _FakeOpenAIClient:
        self.kwargs = dict(kwargs)
        client = _FakeOpenAIClient(list(self._responses), self._on_create)
        self.instances.append(client)
        return client


class _FakeClock:
    """A monotonic clock that only moves when something explicitly moves it.

    Freezing time is what makes the throttle assertion exact: two creates issued
    back to back are, from the guard's point of view, issued at the same instant, so
    the second one must wait the FULL interval - and the recorded sleep duration is
    a number, not a tolerance band. It also means the suite never really sleeps.

    It starts at 0.0 deliberately. Starting at some large number made the "the first
    call never waits" assertion vacuous: any elapsed-time arithmetic against a
    missing previous call would already exceed the interval and skip the sleep for
    the wrong reason. From zero, a guard that treated "no previous request" as
    "the last request was at time 0" would sleep, and the assertion catches it.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self.monotonic_calls = 0

    def monotonic(self) -> float:
        self.monotonic_calls += 1
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        """Move time forward without sleeping - i.e. the way real work does."""
        self.now += seconds

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(time, "monotonic", self.monotonic)
        monkeypatch.setattr(time, "sleep", self.sleep)


def _tool_call(name: str) -> SimpleNamespace:
    """A tool_call-shaped object; only identity and count matter to the guard."""
    return SimpleNamespace(id=f"call_{name}", function=SimpleNamespace(name=name, arguments="{}"))


def _multi_tool_completion() -> _FakeCompletion:
    return _FakeCompletion(_FakeMessage(None, [_tool_call("a"), _tool_call("b")]))


def _user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "content": text}]}


def _empty_runtime() -> Any:
    return SimpleNamespace(functions={})


def _build_with_spy(
    monkeypatch: pytest.MonkeyPatch,
    spy: _SDKConstructorSpy,
    **kwargs: Any,
) -> CountingOpenAILLM:
    """Build through the real builder with the SDK constructor replaced by ``spy``."""
    monkeypatch.setattr("openai.OpenAI", spy)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-FAKE-TEST-KEY-not-real")
    params: dict[str, Any] = {
        "model": GROQ_MODEL,
        "base_url": GROQ_BASE_URL,
        "api_key_env": "GROQ_API_KEY",
        "timeout": 90.0,
    }
    params.update(kwargs)
    return build_openai_compat_llm(**params)


def test_call_count_increments_per_model_turn() -> None:
    """Each query - one model turn - bumps call_count and makes exactly one create."""
    client = _FakeOpenAIClient(
        responses=[
            _FakeCompletion(_FakeMessage("done", None)),
            _FakeCompletion(_FakeMessage("again", None)),
        ]
    )
    llm = CountingOpenAILLM(client=client, model=GROQ_MODEL)
    runtime = _empty_runtime()
    assert llm.call_count == 0

    messages: list[Any] = [_user_message("hi")]
    llm.query("hi", runtime, messages=messages)
    assert llm.call_count == 1
    assert client.completions.creates == 1

    llm.query("hi", runtime, messages=messages)
    assert llm.call_count == 2
    assert client.completions.creates == 2


def test_key_read_from_named_env_var_and_client_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key comes from the caller-named variable, and the client is built bounded:
    an explicit timeout (so a cold model cannot hang forever) and max_retries=5 (the
    SDK's Retry-After-aware retry, the primary defense against a free-tier 429)."""
    sentinel = "gsk-FAKE-TEST-KEY-not-real"
    spy = _SDKConstructorSpy()
    monkeypatch.setattr("openai.OpenAI", spy)
    monkeypatch.setenv("MY_CUSTOM_KEY_VAR", sentinel)

    llm = build_openai_compat_llm(
        model=GROQ_MODEL,
        base_url=GROQ_BASE_URL,
        api_key_env="MY_CUSTOM_KEY_VAR",
        timeout=90.0,
    )

    assert isinstance(llm, CountingOpenAILLM)
    assert spy.kwargs["api_key"] == sentinel  # read from the NAMED variable
    assert spy.kwargs["base_url"] == GROQ_BASE_URL
    assert spy.kwargs["timeout"] == 90.0
    # max_retries=5: the SDK's Retry-After-aware retry is the primary defense
    # against a free-tier 429 rate limit.
    assert spy.kwargs["max_retries"] == 5


def test_missing_key_raises_clean_error_naming_var_not_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing key fails naming the variable to set - never echoing a value.

    A realistic key value is set then removed, so the assertion that this concrete
    value is absent from the message is a genuine leak check.
    """
    fake_key = "gsk-SHOULD-NEVER-LEAK-0123456789"
    monkeypatch.setenv("GROQ_API_KEY", fake_key)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(OpenAICompatLLMError) as exc:
        build_openai_compat_llm(
            model=GROQ_MODEL,
            base_url=GROQ_BASE_URL,
            api_key_env="GROQ_API_KEY",
            timeout=90.0,
        )

    message = str(exc.value)
    assert "GROQ_API_KEY" in message  # the variable NAME guides the user
    assert fake_key not in message  # ... and no key VALUE ever leaks


def test_a_named_key_var_never_falls_back_to_another_providers_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom ``api_key_env`` must fail rather than quietly use GROQ_API_KEY.

    The other key tests set the variable they name, which a fallback would satisfy
    just as well, so none of them can see the cross-variable case: a REAL key sitting
    in the default variable while the caller asked for a different one. If the
    builder reached for it, the run would execute on a credential the operator did
    not ask for - a different account, a different quota, possibly a different model
    tier - and nothing in the result would say so. The gemini path has had this test
    all along; the primary path had not.
    """
    planted = "gsk-SHOULD-NEVER-BE-USED"
    monkeypatch.setenv("GROQ_API_KEY", planted)
    monkeypatch.delenv("MY_OTHER_VAR", raising=False)

    with pytest.raises(OpenAICompatLLMError) as exc:
        build_openai_compat_llm(
            model=GROQ_MODEL,
            base_url=GROQ_BASE_URL,
            api_key_env="MY_OTHER_VAR",
            timeout=90.0,
        )

    message = str(exc.value)
    assert "MY_OTHER_VAR" in message  # the variable actually asked for
    assert "GROQ_API_KEY" not in message  # ... and no hint of a fallback
    assert planted not in message  # ... and never a key value


def test_empty_key_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty value is as good as unset - it still fails cleanly, not silently."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(OpenAICompatLLMError) as exc:
        build_openai_compat_llm(
            model=GROQ_MODEL,
            base_url=GROQ_BASE_URL,
            api_key_env="GROQ_API_KEY",
            timeout=90.0,
        )
    assert "GROQ_API_KEY" in str(exc.value)


def test_name_is_set_for_pipeline_logging() -> None:
    """from_config reads llm.name; a non-None name keeps AgentDojo logging the run."""
    client = _FakeOpenAIClient(responses=[])
    llm = CountingOpenAILLM(client=client, model=GROQ_MODEL)
    assert llm.name == f"openai-compat-{GROQ_MODEL}"


# -- the rate throttle ---------------------------------------------------------


def test_throttle_spaces_a_second_immediate_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back-to-back creates are spaced by the interval; the first one never waits.

    With a frozen clock the second create is issued at the same instant as the
    first, so the guard must sleep the whole interval - the exact number asserted
    here. That spacing is what keeps a fast model from bursting into a 429.
    """
    clock = _FakeClock()
    clock.install(monkeypatch)
    spy = _SDKConstructorSpy(responses=[_FakeCompletion(_FakeMessage("a", None))] * 2)
    llm = _build_with_spy(monkeypatch, spy, min_request_interval_s=DEFAULT_MIN_REQUEST_INTERVAL_S)

    llm.client.chat.completions.create()
    assert clock.sleeps == []  # nothing to space the first call from

    llm.client.chat.completions.create()
    assert clock.sleeps == [pytest.approx(DEFAULT_MIN_REQUEST_INTERVAL_S)]


def test_throttle_does_not_sleep_when_enough_time_has_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that was already slow enough is not delayed further."""
    clock = _FakeClock()
    clock.install(monkeypatch)
    spy = _SDKConstructorSpy(responses=[_FakeCompletion(_FakeMessage("a", None))] * 2)
    llm = _build_with_spy(monkeypatch, spy, min_request_interval_s=1.2)

    llm.client.chat.completions.create()
    clock.advance(5.0)  # the model took five seconds to answer
    llm.client.chat.completions.create()
    assert clock.sleeps == []


def test_zero_interval_disables_the_throttle_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    """min_request_interval_s=0 never sleeps and never even reads the clock."""
    clock = _FakeClock()
    clock.install(monkeypatch)
    spy = _SDKConstructorSpy(responses=[_FakeCompletion(_FakeMessage("a", None))] * 3)
    llm = _build_with_spy(monkeypatch, spy, min_request_interval_s=0)

    for _ in range(3):
        llm.client.chat.completions.create()

    assert clock.sleeps == []
    assert clock.monotonic_calls == 0


def test_the_interval_is_measured_from_request_start_not_from_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spacing is request-START to request-START - a real requests-per-minute guard.

    The distinction only shows up when a request takes measurable time, so here the
    fake create consumes part of the interval. Stamping BEFORE the request (correct)
    leaves only the remainder to sleep off; stamping after the response would treat
    the whole interval as still owed and impose a fixed GAP between requests
    instead, throttling a slow model twice over - a mutation that is otherwise
    invisible against a frozen clock.

    The duration is spent INSIDE the create the guard wraps (via ``on_create``), not
    in a wrapper around the guard: only then does fake time elapse between the
    guard's timestamp and the response, which is what makes the two stamp placements
    distinguishable at all.
    """
    interval = 1.2
    request_duration = 0.5
    clock = _FakeClock()
    clock.install(monkeypatch)

    spy = _SDKConstructorSpy(
        responses=[_FakeCompletion(_FakeMessage("a", None))] * 2,
        on_create=lambda: clock.advance(request_duration),
    )
    llm = _build_with_spy(monkeypatch, spy, min_request_interval_s=interval)

    llm.client.chat.completions.create()
    assert clock.sleeps == []  # nothing to space the first request from

    llm.client.chat.completions.create()
    # Only the unused remainder of the interval is slept off: 1.2 - 0.5. Stamping
    # after the response would instead sleep the full 1.2.
    assert clock.sleeps == [pytest.approx(interval - request_duration)]


def test_a_request_slower_than_the_interval_is_not_delayed_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same property, stated where it is most visible: a model that already took
    longer than the interval to answer owes nothing, so the next request goes out
    immediately. Under a post-response stamp this would sleep the full interval.

    As above, the model's thinking time is spent inside the wrapped create, so it
    falls between the guard's timestamp and the response."""
    clock = _FakeClock()
    clock.install(monkeypatch)

    spy = _SDKConstructorSpy(
        responses=[_FakeCompletion(_FakeMessage("a", None))] * 2,
        on_create=lambda: clock.advance(10.0),
    )
    llm = _build_with_spy(monkeypatch, spy, min_request_interval_s=1.2)

    llm.client.chat.completions.create()
    llm.client.chat.completions.create()
    assert clock.sleeps == []


# -- the shim's contract with the real SDK -------------------------------------


def test_the_guards_really_attach_to_a_real_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build against the REAL ``openai.OpenAI`` and prove the patch actually stuck.

    Every other test here substitutes the SDK class, so all of them would still pass
    if ``chat`` and ``completions`` stopped being cached properties - the guard would
    then be installed on a throwaway object, rebuilt-and-discarded on the next
    attribute access, and both the request counter and the throttle would become
    silent no-ops with a green suite. This is the one test that pins that contract.

    It stays offline: constructing a client opens no connection, and nothing here
    issues a request. The key is a fake read from the environment, and the throttle
    is disabled so no clock is touched.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk-FAKE-TEST-KEY-not-real")
    llm = build_openai_compat_llm(
        model=GROQ_MODEL,
        base_url=GROQ_BASE_URL,
        api_key_env="GROQ_API_KEY",
        timeout=90.0,
        min_request_interval_s=0,
    )

    # The patched closure is what a caller reaches through the real accessor chain.
    assert llm.client.chat.completions.create.__name__ == "_create"
    # ... which holds only because each hop is cached rather than rebuilt per access.
    assert llm.client.chat is llm.client.chat
    assert llm.client.chat.completions is llm.client.chat.completions
    # With a live guard attached, the element reports what the guard counts.
    assert llm.request_count == 0


# -- single-tool-call truncation, and its conditionality -----------------------


def test_multi_tool_call_response_is_truncated_for_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVIDIA's NIM 500s on a >1 tool_call assistant turn, so we keep only the first.

    It also accepts-and-ignores parallel_tool_calls=False, which is why the fix has
    to happen on the response rather than in the request.
    """
    spy = _SDKConstructorSpy(responses=[_multi_tool_completion()])
    llm = _build_with_spy(monkeypatch, spy, base_url=NVIDIA_BASE_URL, min_request_interval_s=0)

    response = llm.client.chat.completions.create()
    tool_calls = response.choices[0].message.tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "a"  # the FIRST call survives


def test_multi_tool_call_response_is_left_alone_for_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Groq handles multi-tool-call history fine, so the baseline model keeps its
    parallel tool calls - truncating them would change agent behaviour for nothing."""
    spy = _SDKConstructorSpy(responses=[_multi_tool_completion()])
    llm = _build_with_spy(monkeypatch, spy, base_url=GROQ_BASE_URL, min_request_interval_s=0)

    response = llm.client.chat.completions.create()
    assert len(response.choices[0].message.tool_calls) == 2


# -- reasoning effort ----------------------------------------------------------


def test_reasoning_effort_reaches_the_stock_element(monkeypatch: pytest.MonkeyPatch) -> None:
    """The value lands on the attribute AgentDojo's OpenAILLM sends on every request."""
    spy = _SDKConstructorSpy()
    llm = _build_with_spy(monkeypatch, spy, reasoning_effort="low")
    assert llm.reasoning_effort == "low"


def test_reasoning_effort_defaults_to_none_and_omits_the_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted here means omitted on the wire - what a non-reasoning model wants."""
    spy = _SDKConstructorSpy()
    llm = _build_with_spy(monkeypatch, spy)
    assert llm.reasoning_effort is None


def test_invalid_reasoning_effort_is_refused_before_any_client_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad effort level fails naming the value and the allowed set, at zero cost.

    The value is CLI configuration, never a secret, so quoting it is the difference
    between a fixable typo and a mystery. No client is constructed, which is how we
    know the failure cost no requests.
    """
    spy = _SDKConstructorSpy()
    monkeypatch.setattr("openai.OpenAI", spy)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-FAKE-TEST-KEY-not-real")

    with pytest.raises(OpenAICompatLLMError) as exc:
        build_openai_compat_llm(
            model=GROQ_MODEL,
            base_url=GROQ_BASE_URL,
            api_key_env="GROQ_API_KEY",
            timeout=90.0,
            reasoning_effort="lowish",
        )

    message = str(exc.value)
    assert "lowish" in message  # the rejected value is named
    assert "low" in message and "xhigh" in message  # ... alongside the allowed set
    assert spy.instances == []  # ... and nothing was built
