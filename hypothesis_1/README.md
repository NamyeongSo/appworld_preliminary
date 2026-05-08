# Hypothesis 1 Experiment

This directory contains the isolated reflection-compression experiment.

## Build the manifest

```bash
.appworld/bin/python -m hypothesis_1.code.manifest
```

This writes `hypothesis_1/data/reflection_manifest.json` by scanning complete
counterfactual samples that have `analysis`, `analysis_random_pairs`, and
`analysis_root` reflection outputs. The current manifest contains 122 samples
across 25 tasks; six reflection artifacts have empty guidance and are recorded
with `guidance_chars: 0`.

## Dry-run prompt previews

```bash
.appworld/bin/python -m hypothesis_1.code.runner --condition reflection_analysis_pass1 --limit 1 --dry-run --quiet --no-lm-log
.appworld/bin/python -m hypothesis_1.code.runner --condition reflection_random_pairs_pass1 --limit 1 --dry-run --quiet --no-lm-log
.appworld/bin/python -m hypothesis_1.code.runner --condition reflection_root_pass1 --limit 1 --dry-run --quiet --no-lm-log
```

Previews are written under `hypothesis_1/outputs/prompt_previews/`; AppWorld
dry-run task scaffolding uses `hypothesis_1/appworld_root/experiments/outputs/hypothesis_1/dry_runs/`.

## Run conditions

Use `MODEL_SERVER_URL=http://localhost:8002` unless the server is exposed elsewhere.

To run the full schedule with concurrent independent AppWorld runs, use:

```bash
HYPOTHESIS_1_PARALLEL=13 MODEL_SERVER_URL=http://localhost:8002 \
  .appworld/bin/python -m hypothesis_1.code.runner --condition all --quiet
```

The runner starts one worker process per independent run spec, up to the
configured parallelism, so vLLM can batch simultaneous requests. It still skips
completed `run_record.json` files unless `--rerun-existing` is passed. Worker
stdout/stderr and per-worker summaries are written under
`hypothesis_1/outputs/parallel_logs/`.

You can also pass `--parallel 13` directly instead of using the environment
variable.

```bash
.appworld/bin/python -m hypothesis_1.code.runner --condition reflection_analysis_pass1 --quiet
.appworld/bin/python -m hypothesis_1.code.runner --condition reflection_random_pairs_pass1 --quiet
.appworld/bin/python -m hypothesis_1.code.runner --condition reflection_root_pass1 --quiet
.appworld/bin/python -m hypothesis_1.code.runner --condition vanilla_react_pass1 --quiet
.appworld/bin/python -m hypothesis_1.code.runner --condition vanilla_react_pass10 --quiet
```

AppWorld artifacts are written under
`hypothesis_1/appworld_root/experiments/outputs/hypothesis_1/`.

## Aggregate

```bash
.appworld/bin/python -m hypothesis_1.code.aggregate --verify-expected
```

This writes:

- `hypothesis_1/outputs/hypothesis_1_metrics.json`
- `hypothesis_1/reports/hypothesis_1_report.md`
