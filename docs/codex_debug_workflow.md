# Codex Debug Workflow (Auto Snapshot)

When a run fails, we automatically write a **Codex‑friendly prompt** and also
auto‑launch a non‑interactive Codex debug session. This makes it easy to hand
off debugging without losing context.

## Where prompts are saved

Prompts are saved under:

```
map-anything/experiments/long_runs/codex_threads/<project_tag>/<run_id>/prompt.md
```

An index is appended to:

```
map-anything/experiments/long_runs/codex_threads/<project_tag>/INDEX.md
```

## How it works

`run_with_retry.sh` now calls:

```
map-anything/bash_scripts/monitor/codex_failure_snapshot.sh
```

whenever a retry‑eligible failure happens (non‑zero exit, hang, or FAIL_REGEX hit).
The snapshot script will also launch:

```
codex exec --model gpt-5.2-codex ...
```

and save the output under the same run directory.

The snapshot includes:
- failure reason + exit status
- last command
- log tail
- system snapshot (GPU, disk, memory, ps)
- repo status (git diff/stat)

## Recommended usage

1. Open the newest prompt:
   ```
   map-anything/experiments/long_runs/codex_threads/<project_tag>/<run_id>/prompt.md
   ```
2. Start a new Codex session and paste the full prompt.
3. Keep each task in its own project tag (e.g. `map-anything`, `pinhole_e2e`, `cyl_det`)
   so threads are grouped by project.

## Configure tags

In your queue line or command, set:

```
CODEX_PROJECT_TAG=map-anything
CODEX_TASK_TAG=pinhole_pose_depth_scale
```

These tags control grouping in `codex_threads/`.

## Turn off auto‑launch (if needed)

Set:

```
CODEX_AUTO_DEBUG=0
```

and the prompt will still be generated, but Codex won't be auto‑run.

## Auto‑queue status & summaries

The watchdog now writes a lightweight status file and per‑run summaries:

- Status (Markdown): `map-anything/experiments/long_runs/STATUS.md`
- Summaries (JSON/MD): `map-anything/experiments/long_runs/auto_queue_logs/auto_queue_summaries/*`
- Progress table: `map-anything/experiments/long_runs/PROGRESS.md`

Each completed run generates `summary.json` and `summary.md` with the last
train/eval metrics parsed from the log, plus exit status and checkpoint path.

### Optional gating (result‑aware scheduling)

You can gate queue advancement using a custom script that reads `summary.json`.

Example gate (thresholds):
```
GATE_SCRIPT=map-anything/bash_scripts/monitor/gate_thresholds.py \
GATE_VAL_POSE=1.0 GATE_VAL_DEPTH=2.0 \
bash map-anything/bash_scripts/monitor/auto_queue_watchdog.sh ...
```

If the gate fails and `STOP_ON_GATE_FAIL=1` (default), the watchdog stops and
waits for manual intervention. Set `STOP_ON_GATE_FAIL=0` to keep looping.

### Optional auto‑decider (queue updates)

You can enable a decision agent that rewrites the queue based on the latest
`summary.json` using Codex.

Example:
```
DECIDE_SCRIPT=map-anything/bash_scripts/monitor/auto_queue_decider.sh \
CODEX_REASONING_EFFORT=xhigh \
bash map-anything/bash_scripts/monitor/auto_queue_watchdog.sh ...
```

The decider keeps the already‑completed prefix intact and only changes future
queue lines.
