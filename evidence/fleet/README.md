# evidence/fleet — what each self-hosted box actually has on it

Produced by [`.github/workflows/fleet-toolset.yml`](../../.github/workflows/fleet-toolset.yml)
(row 67-C2, PMAT-1098, issue #3083).

## Why this directory exists

Run `34448908554` — the 0.66.0 CUDA-asset backfill — resolved the tag, checked out,
built `apr --features cuda` on gx10, proved `libcuda.so` was in the bytes, packaged a
21 MB tarball, wrote its `sha256`, and then died on the last line:

```
gh: command not found
```

The answer to "does gx10 have `gh`?" existed only in somebody's memory, and it was
wrong. `scripts/check_runner_labels.sh` proves a self-hosted selector *discriminates*
between pools; it cannot know what the pool it selects has *installed*. So the toolset
of every box is now measured daily and left as a file.

## What is produced, and what deliberately is not

**Artifacts and a job summary. Not a pull request.** A PR per day against `evidence/`
would be ~365 machine-written PRs a year through a protected branch and a merge queue
that runs at roughly one PR an hour: the cost of the record would exceed the value of
the record. The workflow therefore uploads artifacts and renders one table into the run
summary. Landing a snapshot in this directory is a deliberate human act (below).

| | |
|---|---|
| Workflow | `.github/workflows/fleet-toolset.yml` |
| Schedule | daily at 04:47 UTC, plus `workflow_dispatch` |
| Per-box artifact | `toolset-<label>` — one of `toolset-clean-room`, `toolset-gx10`, `toolset-yoga`; each contains a single `preflight.json` |
| Merged artifact | `fleet-toolset` — every box's `preflight.json`, one directory per box |
| Retention | **90 days** (`retention-days: 90` on both uploads). After that the run is gone and only a file committed here survives |
| Producer | `scripts/ci_self_hosted_preflight.sh`, via `PREFLIGHT_OUT` |

A missing tool **fails** the probe job — that is the finding, and no step is
`continue-on-error`. The upload runs `if: always()`, because the record of a red box is
the record most worth keeping.

## Landing a snapshot here

Download the `fleet-toolset` artifact from a run and commit the file you care about as
`evidence/fleet/<runner-name>.json` — the runner name is the `runner` field inside the
file, not the artifact label, so two boxes in one pool never collide. Say in the commit
message which run it came from; a snapshot with no run id is a number without a
provenance, which this repo does not accept.

## Reading a `preflight.json`

```json
{
  "measured_at": "2026-09-10T11:41:07Z",
  "runner": "gx10-ephemeral-3",
  "labels": "self-hosted,Linux,ARM64,cuda,gx10,ephemeral,docker",
  "arch": "aarch64",
  "glibc": "ldd (Ubuntu GLIBC 2.39-0ubuntu8.3) 2.39",
  "cuda_requested": true,
  "cuda_devices": 1,
  "cuda_driver": "580.119.02",
  "tools": [
    {"name": "jq", "present": true, "path": "/usr/bin/jq", "version": "jq-1.7"},
    {"name": "gh", "present": false, "path": null, "version": null}
  ],
  "missing": ["gh"],
  "exit": 1
}
```

- `missing` empty and `exit: 0` — the box can do the job it was asked about.
- `missing` non-empty — `exit: 1`, and the named entries are what the box lacks.
  `cuda-device` in that list is not a tool: it means `nvidia-smi` is installed and
  `nvidia-smi -L` listed no device, which is the cgroup/permission failure that
  otherwise surfaces as a "CUDA correctness" defect three hours into a run.
- `measured_at` is the **only** clock reading the producer emits, and it is a field of
  a measurement record. Nothing compares it; nothing is gated on it.
- `glibc` is the floor of anything built on that box. The published CUDA assets inherit
  it, so a box whose glibc moves is a release-visible change.

`exit: 1` in a file here is not a stale record to be tidied away. It is the measurement.
