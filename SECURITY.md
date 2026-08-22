# Security Policy & Research Scope

## What this project is

Aegis-RAG is a **defensive security research and portfolio artifact**. It implements and
measures defenses against *indirect prompt injection* — the attack where instructions
hidden in content an AI agent retrieves (a web page, an email, a document, a tool result)
hijack that agent into misusing its tools.

To measure a defense you must be able to run the attack. This repository therefore
contains attack payloads and an attack harness. Their sole purpose is to test the
defenses in this repository.

## Authorized scope — attacks run only against mock tools in a sealed sandbox

**All offensive code in this repository targets simulated tools inside an isolated
container. Nothing in this repository is intended for, or configured for, use against any
system not owned by the operator.**

- **Targets:** AgentDojo's mock suites (workspace / banking / travel / slack) and local
  fixtures only. Never a live third-party service.
- **Side effects:** mock tools such as `send_email` **record to a ledger and never act**.
  "Attack success" is defined as *the ledger shows an unauthorized side-effecting call* —
  a log entry, not a real-world effect.
- **No third-party testing:** this project performs no scanning, probing, or testing of
  any external system, and is not a tool for doing so.

## Three independent guarantees that real credentials cannot enter the attack loop

Defense in depth applies to the lab, not just the product. Each guarantee is sufficient
alone; all three are enforced.

1. **Network** — the eval service runs on a Docker network declared `internal: true`. It
   has no route to the WAN. Even code that tried to reach Gmail or GitHub could not.
2. **Configuration** — the eval service is given no credential env file, and a
   fail-closed startup guard refuses to boot if `AEGIS_TOOLS=mock` while any
   real-credential-shaped variable is present in the environment.
3. **Code** — real tool implementations are import-guarded behind
   `AEGIS_ALLOW_REAL_CREDENTIALS=true`, which the eval image never sets.

`tests/security/test_no_egress.py` asserts guarantee (1) holds at runtime.

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
