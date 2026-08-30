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

Three controls are built and enforced by tests; one part of the intended design is
built-but-not-in-force, and one is still unbuilt. Saying which is which is the point of
this section: a security document that describes intentions in the present tense is worse
than one that describes nothing, because a reader budgets trust against it.

**Enforced now.** Each sentence names the test that fails if it stops being true.

- **No real tools exist, and none can be added unguarded.** There are no real tool
  implementations in this repository to invoke, with or without a credential; the agent
  under test drives AgentDojo's mock suites, whose side-effecting tools mutate an
  in-memory environment. The import guard below protects only modules that call it, so a
  second rule requires every module in `aegis.tools` to call it at import scope - adding
  an unguarded tool means deleting a test that says you may not.
  *Proved by* `test_every_module_in_aegis_tools_calls_the_guard` and its paired negative
  `test_the_guard_coverage_check_would_catch_an_unguarded_module`.
- **The offline test suite cannot reach a provider.** `tests/conftest.py` strips
  `GROQ_API_KEY`, `NVIDIA_API_KEY` and `GEMINI_API_KEY` from the environment for every
  non-costly test, so a test that regressed into making a real call fails instead of
  spending quota.
  *Proved by* `test_offline_suite_has_no_provider_key_in_the_environment` and
  `test_provider_client_refuses_to_build_without_a_key`.
- **Configuration tripwire.** `src/aegis/config/sandbox.py` is a fail-closed startup
  guard: under `AEGIS_TOOLS=mock` (the default), it refuses to start if the environment
  holds a variable shaped like a **real tool** credential — detected by name shape
  (`...TOKEN`, `...SECRET`, `...API_KEY`, ...) or by a well-known issuer prefix in the
  value (`ghp_`, `AKIA`, `xoxb-`, `glpat-`, `ya29.`, ...). The eval runner calls it in
  `main()` before it builds anything, and exits 2.
  *Proved by* `test_tripwire_fires_on_tool_credential_by_name_shape`,
  `test_tripwire_fires_on_tool_credential_by_value_prefix`,
  `test_tripwire_fires_even_inside_a_model_provider_variable`,
  `test_tripwire_is_the_default_with_no_tool_mode_set` and — that the runner really does
  call it, rather than merely owning a guard —
  `test_eval_runner_refuses_to_start_beside_a_tool_credential`.
- **The tripwire never echoes a value.** Findings carry the variable NAME and a reason;
  the value is never a field, never logged, and cannot appear in the refusal message. A
  guard that printed what it found would be a wider leak than the one it prevents.
  *Proved by* `test_error_names_the_variable_and_never_echoes_the_value`.
- **The model provider's key is deliberately allowed.** The agent under test *is* a
  remote model, so `GROQ_API_KEY` and its siblings are model-side by design and pass the
  guard; a run may declare its own key variable with `--api-key-env`. Everything else
  credential-shaped is treated as a tool credential the mock suites never need. A GitHub
  or Slack token pasted *into* a model-provider variable still fires.
  *Proved by* `test_model_provider_key_is_allowed_under_mock_tools`,
  `test_every_model_provider_variable_is_allowed`,
  `test_runs_own_key_variable_can_be_declared` and
  `test_declaring_a_model_variable_cannot_smuggle_a_tool_credential`.
- **Import guard.** `src/aegis/tools/guard.py` refuses to let a real tool module *import*
  unless `AEGIS_ALLOW_REAL_CREDENTIALS=true`, so the dangerous callable never enters the
  process and no registry or tool loop can reach it. There are no real tools yet: the
  guard is written ahead of the thing it guards on purpose, so the first one has to
  satisfy it. `src/aegis/tools/real_example.py` is an inert demonstration.
  *Proved by* `test_real_tool_module_refuses_to_import_by_default`,
  `test_refused_import_is_an_import_error` and
  `test_real_tool_module_imports_with_the_deliberate_opt_in`.
- **One auditable escape hatch.** `AEGIS_ALLOW_REAL_CREDENTIALS=true` — and nothing else —
  disables either guard, and `AEGIS_TOOLS=real` requires it. It is a separate, explicitly
  named variable so that enabling it is visible in a shell history.
  *Proved by* `test_escape_hatch_lets_a_deliberate_operator_through`,
  `test_escape_hatch_is_not_set_by_accident`,
  `test_real_tools_require_the_escape_hatch_even_with_a_clean_environment` and
  `test_unknown_tool_mode_fails_closed`.

All of the above live in `tests/security/test_no_egress.py` — the file earlier versions of
this document cited and did not have.

**Built, but NOT in force for any recorded run.**

- **The eval container and its internal network.** `docker/Dockerfile` and
  `docker/docker-compose.yml` exist, are real, and put the eval service on a network
  declared `internal: true`, with no `env_file` and no credential copied into the image.
  Docker attaches no gateway to an internal network, so a container on it has no route
  off the host.
  *Proved by* `test_eval_service_network_is_internal`,
  `test_every_service_is_attached_only_to_internal_networks`,
  `test_compose_mounts_no_credential_file`, `test_dockerfile_copies_no_credential_file`
  and — skipped cleanly when the Docker CLI is absent —
  `test_compose_file_validates_against_the_docker_cli`.

  **The honest caveat, which is why this is not listed as enforced:** an internal network
  cannot reach the hosted model endpoint either. Every recorded run in `results/` drove
  `gpt-oss-120b` via Groq and therefore did **not** run inside this compose file — it
  could not have. That compose service is for a future local-model or fully-mocked
  configuration. **No number published from this repository was produced inside that
  network.**

**Still not built — intended lab design, not a control in force.**

1. **Network isolation of an actual benchmark run.** That needs a local model (or fully
   recorded/mocked model responses) so the agent under test is reachable without a route
   to the WAN. Until then the compose file above is scaffolding, not a guarantee about any
   published result.

**The honest statement of current isolation:** the agent under test is a hosted model
reached over the internet, so every recorded run makes outbound requests to that endpoint
(the 16-couple defended arm made 61). The runs are not network-isolated. What protects
against real-world side effects is that the *tools* are simulated — now backed by two
enforced guards that keep a real tool credential, and a real tool module, out of the
process in the first place.

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
4. **Taint is attributed by value match, so reformatting defeats it.** An argument is
   treated as untrusted when its text appears verbatim in tool output this conversation
   returned (see [`attribution.py`](src/aegis/middleware/attribution.py)). Write
   `attacker @ evil.test` in the poisoned page and emit `attacker@evil.test` in the call —
   or the reverse — and the match misses, so the send is allowed. This is a deliberate
   trade: the sound alternative ("once any untrusted output is in the context, every later
   argument is untrusted") refuses every side effect in every task that reads something
   first, which is nearly all of them. It is the weaker rule that keeps the system usable,
   and this is what it costs.
5. **`high_risk_args` is a hand-written list of argument NAMES.** The rule that actually
   stops exfiltration only applies to arguments the policy names. A deployment whose email
   tool spells its recipient `to_addr` rather than `to` is unguarded for that argument until
   someone notices and edits the YAML — the taint is still tracked and still visible, and
   the gate simply does not act on it. Auditing `high_risk_args` against a deployment's real
   tool schemas is a manual step this project does not automate.

Results published from this repository will report residual attack success, not only the
attacks that were stopped. Holes 4 and 5 are not merely documented: they ship as clickable
scenarios in the console (`uv run python scripts/demo_ui.py`), which renders them in a
distinct colour from a successful defense, and a test asserts they still reproduce.

### The console binds a socket

`aegis.console` runs a local HTTP server, which is the only listening socket this
repository creates. It binds `127.0.0.1` as a module constant with no flag to change it, so
it is not reachable from the network; it serves exactly one static asset addressed by a
constant, so there is no path parameter that could be generalised into a file read from a
working directory that contains a gitignored `.env`; and it sends a `default-src 'none'`
CSP so the page cannot acquire an external origin by accident. It is a developer tool and
is not part of the eval path, which remains egress-free by network policy.

## Reporting a vulnerability

This is a research/portfolio project with no production deployment and no users. If you
find a flaw in the defenses — especially a bypass of the trust lattice or capability gate
— please open a GitHub issue. Bypasses are interesting findings here, not incidents.

## Credentials

One credential is required, and only one: a **model provider** key (`GROQ_API_KEY` by
default), because the agent under test is a hosted model. **No tool credential is ever
required** — the suites are mock — and the startup tripwire in
`src/aegis/config/sandbox.py` refuses to run beside one.

`.env.example` contains placeholders only; `.env` is gitignored, is stripped from the
environment for every non-costly test by `tests/conftest.py`, and is not copied into the
image or mounted by `docker/docker-compose.yml` (which declares no `env_file`).
