"""The defense runtime: two methods, one shared taint state, no framework.

:class:`AegisMiddleware` is the object an integration holds. It knows nothing
about messages, environments or tool runtimes; it knows that untrusted text
arrived from a named tool, and that a named call with named arguments is pending.
Everything framework-shaped - finding the tool results in a conversation,
reinserting the guarded spans, turning a refusal into whatever the loop expects -
belongs to the adapter that calls it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from aegis.config.policy import SecurityPolicy
from aegis.domain.trust import Tainted, TrustTier, combine_all
from aegis.middleware.attribution import (
    CONTEXT_ARG,
    USER_TURN_SOURCE,
    appears_in,
    normalise,
)
from aegis.middleware.config import (
    QUARANTINE_INSTRUCTION_FLAG,
    QUARANTINE_INSTRUCTIONS,
    QUARANTINE_UNAVAILABLE_FLAG,
    DefenseConfig,
    QuarantineVerdict,
)
from aegis.middleware.ledger import TAINT_VIOLATIONS, GateAction, GateEntry
from aegis.middleware.taint import TaintRecord, TaintState
from aegis.middleware.types import Decision, GuardedOutput, ToolCall, ToolOutput
from aegis.security.capabilities import (
    AuthorizationContext,
    CapabilityGate,
    GateDecision,
    ToolPolicy,
)
from aegis.security.capabilities import Verdict as GateVerdict
from aegis.security.detector import DetectionResult, HeuristicDetector
from aegis.security.quarantine import QuarantineExtractor
from aegis.security.spotlight import Spotlighter

__all__ = ["AegisMiddleware"]


class AegisMiddleware:
    """Both sides of the tool call, for any agent loop that can name three things.

    An adopter supplies, per conversation, a stable id and a progress counter
    (:meth:`begin_turn`); per tool result, the text the model is about to read
    (:meth:`guard`); and per pending call, the name and arguments
    (:meth:`decide`). What comes back is rewritten text plus a taint record, and a
    decision plus the sentence to hand the model in place of a refused call.

    ONE INSTANCE, ONE CONVERSATION AT A TIME
    ----------------------------------------
    The output side records what the tools returned and the call side reads it
    back, so the two halves must be the SAME instance: a gate with its own state
    can never refuse anything, which looks exactly like a working integration. An
    instance is reused across conversations and reset by :meth:`begin_turn`; it is
    not safe to interleave two conversations through one instance, because the
    reset that keeps one conversation's evidence out of the next would fire in the
    middle of both.
    """

    def __init__(
        self,
        config: DefenseConfig,
        *,
        policy: SecurityPolicy | None = None,
        tool_names: Sequence[str] = (),
        authorization: AuthorizationContext | None = None,
        quarantine: QuarantineExtractor | None = None,
        state: TaintState | None = None,
    ) -> None:
        """Build the runtime for one arm of the ablation.

        Args:
            config: Which layers are on.
            policy: The loaded security policy, which is the only place tool sinks
                are declared. Loaded from ``config/trust_tiers.yaml`` on first use
                if omitted, so an output-only integration never has to have one.
            tool_names: Names fed to the L3 detector so its
                ``tool_invocation_attempt`` pattern has something to match.
                Without them that flag can never fire.
            authorization: What the user granted. Defaults to ``allow_all=True``,
                which is the ablation escape hatch documented on
                :class:`AuthorizationContext` and never a deployment setting.
            quarantine: The L4 extractor. Required when ``config.quarantine`` is on.
            state: An existing taint state to adopt. Only for a caller that has to
                hand the same state to something else; otherwise leave it.
        """
        # Asking for L4 without wiring an extractor is a wiring bug, and the quiet
        # version of it - running the "L4 on" arm with L4 silently off - publishes
        # a number for a layer that never executed. Construction is offline and
        # happens once, so failing here costs nothing and cannot end a run
        # mid-flight.
        if config.quarantine and quarantine is None:
            raise ValueError(
                "DefenseConfig.quarantine is on but no QuarantineExtractor was provided; "
                "an L4 arm with no extractor would report L4 numbers for a layer that "
                "never ran. Pass quarantine=QuarantineExtractor(...) or turn the layer off."
            )
        self._config = config
        # Both sub-elements are built with their layer's toggle as `enabled`, so an
        # off layer is off in two places: the branch here and the object itself. A
        # future call site that forgets the branch still gets the ablation arm the
        # config asked for.
        self._spotlighter = Spotlighter(config.spotlight_style, enabled=config.spotlight)
        self._detector = HeuristicDetector(tool_names=tool_names, enabled=config.detect)
        self._quarantine = quarantine if config.quarantine else None
        self._state = state if state is not None else TaintState()
        self._policy = policy
        self._gate: CapabilityGate | None = (
            policy.build_gate(enabled=config.gate) if policy is not None else None
        )
        self._authorization = (
            authorization if authorization is not None else AuthorizationContext(allow_all=True)
        )
        self._ledger: list[GateEntry] = []
        # Tool results L4 was asked about but could not judge. Counted rather than
        # swallowed: an arm where the extractor was down for half the run is not
        # the arm it claims to be.
        self.quarantine_failures = 0

    # -- accessors ------------------------------------------------------------

    @property
    def config(self) -> DefenseConfig:
        return self._config

    @property
    def state(self) -> TaintState:
        """What the tools returned this conversation. Written by :meth:`guard`,
        read by :meth:`decide`."""
        return self._state

    @property
    def policy(self) -> SecurityPolicy:
        """The loaded policy, read from disk on first use if none was supplied.

        Deferred rather than loaded in ``__init__`` so that constructing the
        middleware for its output side alone never depends on a policy file.
        """
        if self._policy is None:
            self._policy = SecurityPolicy.load()
        return self._policy

    @property
    def gate(self) -> CapabilityGate:
        return self._gate if self._gate is not None else self._built_gate()

    def _built_gate(self) -> CapabilityGate:
        gate = self.policy.build_gate(enabled=self._config.gate)
        self._gate = gate
        return gate

    @property
    def authorization(self) -> AuthorizationContext:
        return self._authorization

    @property
    def ledger(self) -> tuple[GateEntry, ...]:
        """Every decision made, oldest first.

        A copy, so a caller cannot append evidence of a refusal that never
        happened. It spans the whole life of the instance rather than one
        conversation - each entry carries its own ``conversation_key`` for
        splitting.
        """
        return tuple(self._ledger)

    @property
    def refusals(self) -> tuple[GateEntry, ...]:
        return tuple(e for e in self._ledger if e.refused)

    def entries_for(self, tool_name: str) -> tuple[GateEntry, ...]:
        return tuple(e for e in self._ledger if e.tool_name == tool_name)

    # -- lifecycle ------------------------------------------------------------

    def begin_turn(self, conversation_id: str, progress: int) -> bool:
        """Declare which conversation the next call belongs to. Returns True on reset.

        Call this at the top of every turn, from BOTH sides. ``progress`` is any
        integer that rises monotonically within one conversation - a message
        count, a turn index, a step counter - and it is the second reset signal:
        a value that did not grow cannot be a continuation of the conversation
        that produced the previous one, whatever the id says.

        This is deliberately not folded into :meth:`guard` or :meth:`decide`.
        Hiding it in one would leave the other side without a reset, and a missed
        reset does not merely add noise: it lets one conversation's evidence
        refuse the next conversation's call, which is a defense that never fired
        being reported as one that did.
        """
        return self._state.begin_turn(conversation_id, progress)

    # -- the OUTPUT side: L1-L4 ----------------------------------------------

    def guard(self, output: ToolOutput) -> GuardedOutput:
        """Record a tool result as untrusted, scan it, and mark it as data.

        L1 (provenance) always. L3 (detection) and L4 (quarantine) on the RAW
        evidence, before L2 rewrites the spacing out from under the patterns the
        detector relies on. L2 (spotlighting) last, per span.

        The order is load-bearing and the record is taken BEFORE L2, so an
        L2-off arm still records, and the text the gate matches arguments against
        is always the raw text rather than our rewrite of it.
        """
        # L1 - the output is attacker-controlled until proven otherwise, and the
        # provenance says which tool it came through.
        tainted: Tainted[str] = Tainted.untrusted(
            output.evidence, f"tool:{output.tool_name}", note=output.note
        )

        # L3 - advisory: the flags are recorded, the content is not touched.
        detection: DetectionResult | None = None
        if self._config.detect:
            detection = self._detector.scan(output.evidence)
            if detection.flags:
                tainted = tainted.flagged(*detection.flags)

        # L4 - also on the raw text: the isolated model must read what the tool
        # actually returned, not our rewrite of it.
        if self._quarantine is not None:
            tainted = self._assess_in_quarantine(tainted)

        record = TaintRecord(tool_name=output.tool_name, tainted=tainted, detection=detection)
        self._state.record(record)

        if not self._config.spotlight:
            # The same tuple object, so an all-layers-off arm can be compared
            # against its input by identity rather than by eye.
            return GuardedOutput(spans=output.spans, record=record)

        # L2 - wrap each span on its own. Wrapping per span rather than joining,
        # wrapping once and returning one string keeps whatever structure the
        # caller had and cannot drop content.
        spans = tuple(
            self._spotlighter.wrap(tainted.with_value(span)).text for span in output.spans
        )
        return GuardedOutput(spans=spans, record=record)

    def _assess_in_quarantine(self, tainted: Tainted[str]) -> Tainted[str]:
        """L4: ask the isolated extractor whether this text is addressing us.

        What comes back across the boundary is a :class:`QuarantineVerdict` - two
        booleans - and only the booleans are used. The T1 value itself is
        deliberately discarded rather than substituted into the output: L4's job
        here is to produce a high-confidence FLAG for the gate, not to launder
        attacker prose into a higher tier by round-tripping it through a model.

        Any failure flags ``quarantine_unavailable`` and returns the value
        otherwise unchanged. That is fail-OPEN, which is the opposite of what
        :mod:`aegis.security.quarantine` does internally, and it is the right
        choice at exactly this call site: the extractor's own fail-closed rule
        protects the trust lattice (no unvalidated value may reach T1, and none
        does here), whereas turning an outage into a blocking flag would refuse
        every side effect for the rest of the run and report the outage as a
        defense. The catch is broad on purpose - a transport error, a malformed
        body and a bug in this method all cost the same thing.
        """
        assert self._quarantine is not None  # guarded by the caller
        try:
            verdict = self._quarantine.extract(
                tainted, QuarantineVerdict, instructions=QUARANTINE_INSTRUCTIONS
            )
        except Exception:
            self.quarantine_failures += 1
            return tainted.flagged(QUARANTINE_UNAVAILABLE_FLAG)
        if verdict.value.addresses_the_agent or verdict.value.requests_a_tool_call:
            return tainted.flagged(QUARANTINE_INSTRUCTION_FLAG)
        return tainted

    # -- the CALL side: L5 ----------------------------------------------------

    def decide(self, calls: Sequence[ToolCall], *, known_tools: Collection[str]) -> list[Decision]:
        """Judge every pending call of ONE turn, in order.

        A whole turn at a time rather than one call at a time, because the text
        every call in the turn is matched against is snapshotted ONCE here. Judging
        calls one by one would either rebuild that snapshot per call or invite a
        caller to interleave :meth:`guard` between the calls of a single turn -
        which would let a call be judged against output the model had not seen when
        it chose its arguments.

        ``known_tools`` is the caller's registry. A call naming something absent
        from it has no side effect to prevent and is left ungated: the caller will
        answer it with its own "no such tool" error, and crediting the gate with
        refusing a hallucinated name would pad the ledger with attacks that never
        had a tool to reach.
        """
        haystack = tuple((r, normalise(r.tainted.value)) for r in self._state.records)
        return [self._decide_one(call, known_tools, haystack) for call in calls]

    def _decide_one(
        self,
        call: ToolCall,
        known_tools: Collection[str],
        haystack: tuple[tuple[TaintRecord, str], ...],
    ) -> Decision:
        name = call.name
        if not name or name not in known_tools:
            return self._ungated(call, "tool is not registered in the runtime")

        # From the POLICY, not from the gate's own copy: `CapabilityGate.check`
        # short-circuits on `enabled=False` while `policy_for` does not, so reading
        # the sink classification off the gate would make the read-only override
        # below depend on which arm of the ablation is running.
        policy = self.policy.policy_for(name)
        tainted = self._taint_args(call.args, policy, haystack)
        decision = self.gate.check(name, tainted, self._authorization)
        action, note = self._action_for(decision, policy)
        entry = GateEntry(
            conversation_key=self._state.conversation_key,
            tool_name=name,
            verdict=decision.verdict,
            action=action,
            effective_tier=decision.effective_tier,
            codes=decision.codes,
            reason=decision.explain(),
            tainted_args=tuple(k for k, v in tainted.items() if v.is_attacker_influenced),
            note=note,
        )
        self._ledger.append(entry)
        return Decision(call=call, entry=entry, gate=decision)

    def _ungated(self, call: ToolCall, note: str) -> Decision:
        entry = GateEntry(
            conversation_key=self._state.conversation_key,
            tool_name=call.name,
            verdict=GateVerdict.ALLOW,
            action=GateAction.EXECUTED,
            effective_tier=TrustTier.SYSTEM,
            reason=f"not gated: {note}",
            note=note,
        )
        self._ledger.append(entry)
        return Decision(call=call, entry=entry, gate=None)

    def _action_for(
        self, decision: GateDecision, policy: ToolPolicy | None
    ) -> tuple[GateAction, str]:
        """Turn a gate verdict into what actually happens to the call.

        Two places where the answer is not the verdict.

        READ-ONLY CALLS ARE NEVER REFUSED FOR BEING TAINTED. The gate already
        exempts read-only tools from detector flags; this extends that to the other
        two taint-derived codes, so no policy edit can turn a tier floor or a
        high-risk argument on a read into a refusal. The worst case for a poisoned
        read is a wrong answer; the cost of refusing reads is the whole task, every
        time, including the tasks with no attack in them. Whether a tool is a read
        is read from the loaded policy, never from a list in this file.

        CONFIRM IS TREATED AS DENY. ``requires_confirmation`` tools ask a human.
        Where there is no human the call cannot proceed, and treating CONFIRM as
        ALLOW would report a number for a configuration that does not exist - in
        deployment those calls stop and wait. So it is refused, and the ledger
        records verdict CONFIRM alongside action REFUSED so a write-up can separate
        "blocked" from "would have asked".
        """
        if decision.verdict is GateVerdict.ALLOW:
            return GateAction.EXECUTED, ""

        if (
            policy is not None
            and not policy.side_effecting
            and decision.verdict is GateVerdict.DENY
            and decision.codes
            and all(code in TAINT_VIOLATIONS for code in decision.codes)
        ):
            return (
                GateAction.EXECUTED,
                "read-only tool: taint alone never refuses a read (the worst case is a "
                "wrong answer; refusing costs the whole task, attack or not)",
            )

        if decision.verdict is GateVerdict.CONFIRM:
            return (
                GateAction.REFUSED,
                "CONFIRM treated as DENY: an unattended benchmark run has no human to ask",
            )
        return GateAction.REFUSED, ""

    def _taint_args(
        self,
        args: Mapping[str, Any],
        policy: ToolPolicy | None,
        haystack: tuple[tuple[TaintRecord, str], ...],
    ) -> dict[str, Tainted[Any]]:
        """Attach a tier and provenance to every argument of one call."""
        tainted = {key: self._taint_value(value, haystack) for key, value in args.items()}
        if policy is not None and policy.side_effecting and not tainted:
            context = self._context_taint()
            if context is not None:
                tainted[CONTEXT_ARG] = context
        return tainted

    def _taint_value(
        self, value: Any, haystack: tuple[tuple[TaintRecord, str], ...]
    ) -> Tainted[Any]:
        matches = [record for record, text in haystack if appears_in(value, text)]
        if not matches:
            return Tainted.trusted(
                value,
                TrustTier.USER,
                USER_TURN_SOURCE,
                note="not traceable to any tool output this task",
            )
        # combine_all takes the GLB of the matching outputs' tiers and merges their
        # provenance, so the detector flags raised on those outputs travel onto the
        # argument - which is what lets the gate's flag rule fire on the specific
        # value the attacker chose rather than on the conversation as a whole.
        return combine_all([record.tainted for record in matches], value)

    def _context_taint(self) -> Tainted[str] | None:
        records = self._state.records
        if not records:
            return None
        return combine_all([r.tainted for r in records], CONTEXT_ARG)
