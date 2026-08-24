# Week-0 results: what is in this directory, and what may be quoted

A number in this directory is only useful if a reader can tell what produced it.
Two of the files here were, at one point, sitting at a path whose name claimed
more than the run behind it could support - so this file states, explicitly,
which result is the baseline of record and which ones must never be quoted as
one.

Everything here is written by `python -m evals.agentdojo.runner`. Raw per-run
task logs go to `results/raw/` and are gitignored; the curated JSON files beside
this README are the artifacts.

## The baseline of record

`results/week0_baseline.json` - **the default `--out` path, and currently
ABSENT on purpose.**

The Week-0 baseline of record is the Groq / `openai/gpt-oss-120b` run at
`--reasoning-effort low`: utility 100%, ASR 8.3% over 12 injected couples
(`--max-tasks 3 --max-injection-tasks 4`). That is the configuration the runner
defaults to, and it is the only run in this project that is both capable enough
to complete the user tasks and measurably hijackable - which is what makes it a
security baseline rather than a shrug.

It has not been re-recorded through the current runner yet, and this file will
not invent it. Re-record it live:

```
python -m evals.agentdojo.runner --max-tasks 3 --max-injection-tasks 4
```

That writes `results/week0_baseline.json` with the current schema. Until then,
quote the ASR from `week0_baseline_groq.json` (below) only with its caveat
attached.

## `week0_baseline_groq.json` - the real measurement, in the OLD schema

This is the run the 8.3% ASR comes from. It is genuine, and it is kept.

It predates the runner's current schema, so it does not round-trip with anything
written today:

- `"provider": "groq"` is not a valid `--provider` value any more (the providers
  are `openai-compat` and `gemini`; Groq is an endpoint, reached via
  `--base-url`);
- its keys are `users` / `injections` / `model_calls`, not `user_task_ids` /
  `injection_task_ids` / `total_model_calls` / `total_model_turns`;
- it carries no `timestamp`, `reasoning_effort`, `timeout`, `api_key_env`,
  `force_rerun`, `replayed`, or `max_injection_tasks`.

So it cannot be diffed field-by-field against a defended run recorded later.
Re-recording it through the current runner (above) is what fixes that; it is
worth the quota.

## Runs that are NOT valid baselines

`week0_nvidia_llama31_8b.json` (4 user x 2 injection tasks) and
`week0_nvidia_llama31_8b_wide.json` (5 x 5) are
`meta/llama-3.1-8b-instruct` via `https://integrate.api.nvidia.com/v1`. Both
score **ASR 0.0**, and that zero means nothing defensive.

An 8b model that cannot reliably drive the tool loop fails the attack because it
fails at the task - note the utility in the wide run: 0.4. "The attack did not
succeed because the agent did not succeed" is not resistance, and quoting a 0%
ASR from it would claim a defensive result that was never measured. The NVIDIA
path stays fully supported through flags (its endpoint quirks are handled in
`evals/agentdojo/openai_llm.py`), and these files are kept as evidence of the
path working - not as baselines.

They used to live at `week0_baseline.json` and `week0_baseline_wide.json`, i.e.
at the default output path, where the file name asserted exactly the thing the
runner's own docstring says never to claim. They are named after what they are
now.

## A note on the stale cache in `results/raw/`

AgentDojo caches each task result on disk under
`logdir/<pipeline name>/<suite>/<user task>/...`, and replays it instead of
calling the model whenever it is told not to force a rerun. The log directories
already here were written by earlier pipeline names.

Two things keep them from being served up as a fresh measurement: the runner
forces a rerun unless `--resume` is passed, and the pipeline name now carries a
fingerprint of every setting that can change a score (endpoint, reasoning
effort, timeout, benchmark version), so a differently-configured run cannot
collide with these entries. A run that somehow still reports metrics without
making a single request is flagged `"replayed": true` in its JSON and shouts
about it on the console. If you want to be certain, point `--logdir` somewhere
empty.
