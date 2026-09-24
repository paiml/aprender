# PMAT-4120 receipt: guard-tree killed the CI runner (#4120)

- ticket PMAT-4120 (from #4120), kind code; branch `fix/4120-teardown-never-pid-1` off `chore/0.69.1-merge-back` @375b34522 (the fold target, cop ruling); orchestrator opus-5-5.
- **Root cause (traced):** under `strace -f -e trace=execve,kill,tkill,tgkill` of `scripts/check_crux_serve_code.sh` alone, the sender was `bash scripts/lib/crux_cell_teardown.sh …/td.state …/init.pid`, issuing `kill(1, SIGTERM)` then `kill(1, SIGKILL)`. Row T2 wrote `1` into a pid file, using init as "a server that survives TERM and KILL". EPERM on dev boxes, so the row was green there (the guard printed `66 row(s), 0 failed`). In the CI runner container pid 1 is Runner.Listener, and the job may signal it (runs 35900354071, 35913559755, 35935643646). Only #4046's tree has this guard, which is why main's guard-tree passed.
- **Ruled out on the way:** GNU `timeout`'s `kill(0, TERM)`/`kill(0, CONT)` (check_story_json_streams `run_cmd 1 sleep 5`) calls `setpgid(0,0)` first, and a same-group victim survived. A gx10 Listener stand-in running main's guard_tree received no signal.
- **Commit 1 (dc92d818c, f1b09d4d6):**
  - the teardown refuses pid 1 by name (the cell FAILS, pid 1 is never signalled);
  - every TERM/KILL goes through one seam, `CRUX_TEARDOWN_KILL`;
  - T2 rebuilt around the guard's own `sleep` behind a no-op kill;
  - **T2b MUST-RED**: a pid file naming 1 is refused and never signalled;
  - mutant **M17** (refusal dropped) is killed; M10 re-anchored.
- **Verification (orchestrator re-runs):** whole guard under `strace -f -e trace=kill,tkill,tgkill` on the final tip: **68 rows, 0 failed, exit 0; kill(1,…) calls 0; group kills 0** (before the fix: 66 rows green with kill(1,TERM)+kill(1,KILL) in the trace). A hand spot-check through the seam: pid 1 → FAILED, no signal sent; our own survivor → `-TERM`, `-KILL` via the seam, FAILED. bashrs findings equal to the base.
- **Quorum (commit 1):** 3/3 PASS, sonnet-5 lanes, `degraded: same-family` (agy 429). Record: `docs/audits/quorum-PMAT-4120-commit1.json`.
- **Also found:** the M10 mutant row leaks one TERM-ignoring fixture loop per run; 37 stale loops (up to 15 h old) were killed on lambda. That fix is commit 2.
- **Commit 2 (follow-up, separate quorum):** EXIT-trap cleanup of stubborn fixtures; malformed pid lines fail the cell (lane A); T2b made non-vacuous (lane B); `setsid --wait` per guard in guard_tree.sh with a must-RED.
- Verdict for commit 1: DONE to the quorum receipt; not armed (6c folds into #4046).
