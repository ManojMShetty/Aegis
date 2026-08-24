# Results: what is in this directory, and what may be quoted

A number is only useful if a reader can tell what produced it. Two files here
once sat at a path whose name claimed more than the run behind it could support,
so this file states explicitly which result is the baseline of record, which ones
must never be quoted as one, and what is not measured yet.

Everything here is written by `python -m evals.agentdojo.runner`. Raw per-run
task logs go to `results/raw/` and are gitignored; the curated JSON files beside
this README are the artifacts.

Intervals and paired tests come from `python -m evals.stats.analysis`.

## The baseline of record

`results/week0_baseline.json` - Groq / `openai/gpt-oss-120b` at
`--reasoning-effort low`, workspace suite v1.2, `important_instructions` attack,
`--max-tasks 3 --max-injection-tasks 4` (12 injected couples).

```
utility               66.7%    2/3    95% CI [20.8%, 93.9%]
asr                    8.3%   1/12    95% CI [ 1.5%, 35.4%]
utility_under_attack  100.0%  12/12   95% CI [75.8%, 100.0%]
```

`force_rerun: true`, `replayed: false`, 48 model requests - i.e. every task in it
was measured, not replayed. Reproduce with:

```
python -m evals.agentdojo.runner --max-tasks 3 --max-injection-tasks 4
```

## `week0_baseline_16.json` - the widest measurement, and the better one to quote

A 4x4 block: the same configuration, 16 injected couples.

```
utility               75.0%    3/4    95% CI [30.1%, 95.4%]
asr                   12.5%   2/16    95% CI [ 3.5%, 36.0%]
utility_under_attack  68.8%   11/16   95% CI [44.4%, 85.8%]
```

It carries `replayed: true` and `total_model_calls: 0`, and that needs saying
plainly: the 16 underlying task runs were genuinely measured earlier the same
day, and this file re-aggregates them from the cache without making new requests.
It is an honest re-aggregation of real measurements, not a fresh measurement, and
the runner shouts about it precisely so nobody mistakes one for the other.

## Two things this data already establishes

**Run-to-run variance is large at this sample size.** Two runs of the *identical*
configuration - same model, same three tasks, temperature 0 - scored utility
100% and then 66.7%; `user_task_0` passed once and failed once. Any single-run
number here should be read with its interval, not on its own.

**The sample is far too small for a defended comparison to mean anything yet.**
The exact McNemar test cannot reach p < 0.05 with fewer than 6 discordant pairs
(the best case at n=5 is p = 0.0625). With 2 baseline hijacks, even a defense
that blocked *both* would produce p = 0.50. A credible "the defense reduces ASR"
claim therefore needs at least 6 baseline hijacks, which at ASR ~12.5% means
roughly 48+ couples per arm. On a 200k-token daily cap that is a multi-day
measurement, not an afternoon.

## Not measured yet: the defended arm

`evals/agentdojo/defense.py` is wired and tested, and a defended run is invoked
with:

```
python -m evals.agentdojo.runner --defense aegis --defense-layers all \
    --max-tasks 3 --max-injection-tasks 4
```

No defended result exists in this directory, and **no reduction in ASR is
claimed anywhere in this repository.** When one is recorded it must be compared
against a baseline arm covering exactly the same couples, and reported with the
discordant counts and the McNemar p-value, not as a bare percentage.

## Runs that are NOT valid baselines

`week0_nvidia_llama31_8b.json` (4 user x 2 injection) and
`week0_nvidia_llama31_8b_wide.json` (5 x 5) are `meta/llama-3.1-8b-instruct` via
`https://integrate.api.nvidia.com/v1`. Both score **ASR 0.0**, and that zero
means nothing defensive.

An 8b model that cannot reliably drive the tool loop fails the attack because it
fails at the task - note the utility in the wide run: 0.4. AgentDojo itself warns
"Not all injection tasks were solved as user tasks", i.e. it could not perform
the injected task even when asked to directly. "The attack did not succeed
because the agent did not succeed" is not resistance. These files are kept as
evidence that the NVIDIA path works, not as baselines.

They used to live at `week0_baseline.json` and `week0_baseline_wide.json` - at
the default output path, where the file name asserted exactly what the runner's
own docstring says never to claim.

## `week0_baseline_groq.json` - a real measurement in the OLD schema

The run the original 8.3% came from. Genuine, and kept, but it predates the
current schema: `"provider": "groq"` is not a valid `--provider` value any more
(the providers are `openai-compat` and `gemini`; Groq is an endpoint reached via
`--base-url`), its keys are `users` / `injections` / `model_calls`, and it has no
`timestamp`, `reasoning_effort`, `timeout`, `api_key_env`, `force_rerun`,
`replayed`, or `max_injection_tasks`. It cannot be diffed field-by-field against
anything recorded today; `week0_baseline.json` supersedes it.

## A note on the cache in `results/raw/`

AgentDojo caches each task result under
`logdir/<pipeline name>/<suite>/<user task>/...` and replays it instead of
calling the model whenever it is told not to force a rerun.

Three things keep a replay from being written up as a fresh measurement: the
runner forces a rerun unless `--resume` is passed; the pipeline name carries a
fingerprint of every setting that can change a score - **the model id**,
endpoint, reasoning effort, timeout, benchmark version, **and the defense
identity plus its layer set and its `DefenseConfig` label**, so a defended run
can never be served from the undefended run's entries; and a run that reports
metrics having made zero requests is flagged `"replayed": true` and shouts about
it on the console.

Every result file also records the `pipeline_name` it ran under, so a result can
be tied back to the transcripts under `logdir/<pipeline name>/` that produced it.
Files written before that field existed - including `week0_baseline_16.json` -
do not carry it.

Two consequences worth knowing, both of the same shape. Adding the defense
identity to the fingerprint changed the *undefended* fingerprint too, and so did
adding the model id (which appeared in the name only as a token, and
`openai/gpt-oss-120b`, `openai_gpt-oss-120b` and `openai:gpt-oss-120b` all
sanitize to the same token - three models, one cache entry). Each change orphaned
the earlier cache directory, and each time its 31 task runs were copied to the
new name rather than re-measured, since the configuration behind them is
unchanged: the fingerprint moved, the runs did not. The gpt-oss-120b baseline
therefore now lives under
`aegis-baseline-local-openai_gpt-oss-120b-low-11986fc0`, with the previous
`...-low-533f0ab5` kept beside it. Any future change to the fingerprint inputs
invalidates the cache the same way, and that should be a deliberate decision each
time: on a capped free tier, silently re-running a day of tasks is the expensive
failure mode.
