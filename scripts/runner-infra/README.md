# Self-hosted runner disk-guard infra

Automation preventing the `/` = 100% full class of failure that took 16 runners
offline on 2026-04-22. Two layers:

1. **Per-job pre-hook** (`runner-pre-job.sh`) — runs via
   `ACTIONS_RUNNER_HOOK_JOB_STARTED`. Checks disk; if usage ≥ 85%, aggressively
   prunes `_work/*/target/` before the job starts. Also chowns any root-owned
   leftovers from prior container builds.

2. **Nightly safety net** (`runner-disk-guard.service` + `.timer`) — at 04:00
   local daily, prunes any `_work/*/target/` that hasn't been modified in 7+
   days, regardless of disk usage.

## Installation

```bash
host=intel  # or whichever runner host
scp scripts/runner-infra/{runner-disk-guard.sh,runner-pre-job.sh,runner-disk-guard.service,runner-disk-guard.timer} "$host:/tmp/"
ssh "$host" '
  sudo install -m 0755 /tmp/runner-disk-guard.sh /usr/local/bin/runner-disk-guard.sh &&
  sudo install -m 0755 /tmp/runner-pre-job.sh    /usr/local/bin/runner-pre-job.sh &&
  sudo install -m 0644 /tmp/runner-disk-guard.service /etc/systemd/system/ &&
  sudo install -m 0644 /tmp/runner-disk-guard.timer   /etc/systemd/system/ &&
  sudo systemctl daemon-reload &&
  sudo systemctl enable --now runner-disk-guard.timer
'
```

Each runner's `.env` must point to the pre-job hook:

```
ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/bin/runner-pre-job.sh
```

**This is no longer true, and has not been since GH-202 (2026-08-16.)** MEASURED
on mac-server, 2026-08-22:

```
$ grep -rhs ACTIONS_RUNNER_HOOK_JOB_STARTED /etc/systemd/system/actions.runner.*.service.d/*.conf | sort | uniq -c
     17 Environment="ACTIONS_RUNNER_HOOK_JOB_STARTED=/home/noah/data/actions-runner/pre-job.sh"
$ ls -la /usr/local/bin/runner-pre-job.sh
-rwxr-xr-x 1 root root 2879 Aug 18 17:18 /usr/local/bin/runner-pre-job.sh     # orphaned
```

All 17 runners invoke `/home/noah/data/actions-runner/pre-job.sh` — a different
script, owned by paiml/infra, which calls `ci-reaper.sh` and never mentions
`runner-disk-guard.sh`. So `runner-pre-job.sh` here, and with it the
`--pre-job` mode of `runner-disk-guard.sh`, is DEAD on the fleet; the installed
copy at `/usr/local/bin/runner-pre-job.sh` is a shadowed artifact that nothing
executes. Only the `--nightly` timer path of `runner-disk-guard.sh` actually
runs. Editing the `--pre-job` path and expecting a fleet effect will do nothing
— check `type -aP` / the unit drop-ins first.

## Case table

```bash
bash scripts/runner-infra/runner-disk-guard.sh --self-test   # 7 rows, rc=0
```

Seven rows over a throwaway tree, covering both directions of the only judgement
this guard makes: which directories it refuses to touch. R3 and R6 are the ones
that matter — a live build under a parent whose mtime froze hours ago, in both
the `<PR>/run-<ID>` and the merge-queue `gh-readonly-queue/…/run-<ID>` shapes.
They are RED against the pre-2026-08-22 selection, which tested the mtime of the
depth-1 directory rather than of anything actually being written. R1, R5 and R7
fail if a "fix" simply stops reclaiming, which would trade a deleted build for a
full disk.

NOT WIRED INTO CI. This is a host script, deployed by hand (see Installation),
so nothing re-runs the table on a change to it. Run it before you `scp`.

## Tuning

Environment variables honored by `runner-disk-guard.sh`:

| Var | Default | Meaning |
|---|---|---|
| `HIGH_WATER_PCT` | 85 | Pre-job prune threshold |
| `STALE_DAYS` | 7 | Nightly: age cutoff — no write ANYWHERE below the dir |
| `RUNNERS_ROOT` | `/home/noah/data` | Parent of `actions-runner*` dirs |
| `BIND_MOUNT_ROOTS` | `/mnt/nvme-raid0/targets/aprender-ci` | CI bind-mount target roots |

## Manual recovery

If `/` goes 100% full before the guard can run:

```bash
ssh intel 'for svc in actions.runner.paiml.intel-clean-room.service \
                       actions.runner.paiml.intel-clean-room-{2..16}.service; do
  sudo systemctl stop "$svc"
done
sudo bash -c "for i in \"\" 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  rm -rf /home/noah/data/actions-runner\${i:+-}\${i}/_work
done"
for svc in actions.runner.paiml.intel-clean-room.service \
           actions.runner.paiml.intel-clean-room-{2..16}.service; do
  sudo systemctl start "$svc"
done'
```

## Why `target/` and not the whole `_work/`

`target/` is the Rust build directory — by far the biggest consumer (70–110 GB
per runner). It is fully reproducible from source. The rest of `_work/`
(checkouts, `_tool`, `_actions`) is small (~1 GB total). Leaving checkouts
intact lets GitHub's fetch-only diff pull work instead of a fresh clone per
job.
