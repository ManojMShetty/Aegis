# Security Policy & Research Scope

## What this project is

Aegis-RAG is a **defensive security research and portfolio artifact**. It implements and
measures defenses against *indirect prompt injection* — the attack where instructions
hidden in content an AI agent retrieves (a web page, an email, a document, a tool result)
hijack that agent into misusing its tools.

To measure a defense you must be able to run the attack. This repository therefore
contains attack payloads and an attack harness. Their sole purpose is to test the
defenses in this repository.

## Authorized scope — attacks run only against mock tools

**All offensive code in this repository targets simulated tools. Nothing in this
repository is intended for, or configured for, use against any system not owned by the
operator.**

- **Targets:** AgentDojo's mock suites (workspace / banking / travel / slack) and local
  fixtures only. Never a live third-party service.
- **Side effects:** mock tools such as `send_email` **record to a ledger and never act**.
  "Attack success" is defined as *the ledger shows an unauthorized side-effecting call* —
  a log entry, not a real-world effect.
- **No third-party testing:** this project performs no scanning, probing, or testing of
  any external system, and is not a tool for doing so.

## What actually keeps real credentials out of the attack loop today

Only one control below is built. Saying which is which is the point of this section: a
security document that describes intentions in the present tense is worse than one that
describes nothing, because a reader budgets trust against it.

**Enforced now.**

- **No real tools exist.** There are no real tool implementations in this repository to
  invoke, with or without a credential. The agent under test drives AgentDojo's mock
  suites, whose side-effecting tools mutate an in-memory environment.
- **The offline test suite cannot reach a provider.** `tests/conftest.py` strips
  `GROQ_API_KEY`, `NVIDIA_API_KEY` and `GEMINI_API_KEY` from the environment for every
  non-costly test, so a test that regressed into making a real call fails instead of
  spending quota.

**Not built — this is the intended lab design, not a control in force.**

1. **Network isolation.** The plan is an eval service on a Docker network declared
   `internal: true`, with no route to the WAN. There is no Dockerfile or compose file in
   this repository, and `tests/security/test_no_egress.py`, cited in earlier versions of
   this document as asserting that guarantee, does not exist.
2. **Configuration tripwire.** A fail-closed startup guard refusing to boot if
   `AEGIS_TOOLS=mock` while a real-credential-shaped variable is present. `AEGIS_TOOLS`
   appears in no source file today.
3. **Import guard.** Real tool implementations gated behind
   `AEGIS_ALLOW_REAL_CREDENTIALS=true`. Also not present, because there are no real tool
   implementations to gate.

**The honest statement of current isolation:** the agent under test is a hosted model
reached over the internet, so every recorded run makes outbound requests to that endpoint
(the 16-couple defended arm made 61). The runs are not network-isolated. What protects
against real-world side effects is that the *tools* are simulated, not that the process
is sandboxed.

## Honest limitations

Stating what a defense does *not* stop is part of the deliverable. Known residual holes:

1. **Data-as-argument.** Schema validation constrains a value's *shape*, not its
   *semantics*. A `str` field that survives quarantine is still attacker-influenced
   content. The capability gate — not the type system — must decide whether such a value
   may reach a side-effecting tool.
2. **Poisoning is not injection.** The quarantine model can faithfully extract a
   *correctly-typed but false* value from a poisoned document (PoisonedRAG). Provenance
   tells you where a claim came from; it does not tell you the claim is true.
3. **Adaptive attackers degrade string-fragile layers.** The detector (L3) is pattern-based
   and an adaptive adversary will erode it. The structural layers (L4 quarantine boundary,
   L5 capability gate) are designed not to depend on string matching. Phase 2 measures
   exactly where each layer is robust and reports where it breaks, with the attacker's
   budget stated.

Results published from this repository will report residual attack success, not only the
attacks that were stopped.

## Reporting a vulnerability

This is a research/portfolio project with no production deployment and no users. If you
find a flaw in the defenses — especially a bypass of the trust lattice or capability gate
— please open a GitHub issue. Bypasses are interesting findings here, not incidents.

## Credentials

No real credentials are required to run the evaluation suite. `.env.example` contains
placeholders only; `.env` is gitignored and is never mounted into the eval container.
