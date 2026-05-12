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

(Already wired on intel's 16 clean-room runners as of 2026-04-22.)

## Tuning

Environment variables honored by `runner-disk-guard.sh`:

| Var | Default | Meaning |
|---|---|---|
| `HIGH_WATER_PCT` | 85 | Pre-job prune threshold |
| `STALE_DAYS` | 7 | Nightly: mtime age cutoff for target/ |
| `RUNNERS_ROOT` | `/home/noah/data` | Parent of `actions-runner*` dirs |

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
