# PMAT-4055 receipt: serve teardown from a symlinked cwd (branch fix/4055-teardown-physical-cwd)

Base: 0dccce21b (the #4046 merge-back head). Fix commit: 849a96ffd.

## The defect, measured (yoga, 2026-09-23, apr 0.69.1 fc942f6be, rung qwen35-2b-q4km, cpu lane)
- The ladder ran from /mnt/nvme-raid0/tmp/apr36-4034ab/0dccce21b. On yoga and gx10,
  /mnt/nvme-raid0 -> /home/noah/eph-work/intel-mirror; lambda's is a real directory.
- `apr serve` pid 1442703 survived its probe, re-parented to init (ppid 1), with cwd
  /home/noah/eph-work/intel-mirror/tmp/apr36-4034ab/0dccce21b, which is PHYSICAL.
- It held fd 3 -> /tmp/apr-gpu.lock. /proc/locks: `FLOCK ADVISORY WRITE 1442700` (the dead flock
  wrapper), with waiters 1444583 (the next cell's `apr run --gpu`) and 1536287 (another session).
- The teardown's guard `readlink -f /proc/$p/cwd != $PWD` compared physical with logical, so it
  printed `failed` without killing.

## The fix
- `here=$(pwd -P)`; both cwd comparisons use it.
- ladder_td_verdict: `clean`/`escalated` become `failed` while /proc/locks records the GPU lock
  under a pid of the teardown's tree.

## Evidence
- check_ladder_serve_teardown.sh, new cases (real processes, real symlink, real flock):
  symlink-cwd -> `escalated`, server dead, lock free; lock-escapee -> `failed`.
- --self-test: the logical-$PWD plant turns symlink-cwd RED, and the lock-blind plant turns
  lock-escapee RED; the two existing plants are unchanged. rc 0.
- Against the pre-fix script the guard exits 2 (it refuses to judge a teardown with no
  ladder_td_verdict). It never passes it.
- check_model_ladder.sh --self-test: 302 ok / 0 FAIL. check_ladder_serve_verdict,
  check_serve_probe_evidence, check_serve_backend_record, check_ladder_output_judged,
  check_ladder_write_errors, check_ladder_only_selection, check_ladder_provenance: all rc 0.
  bashrs: 0 errors on both files.
- Real-host proof: running on yoga from the logical path (the A' leg of the #4034 A/B,
  c3054f4f1 = 0dccce21b + 849a96ffd); the result is appended to #4055 when it finishes.

## Not changed
- `flock -o` (closing the lock fd in the child) would stop a survivor holding the fleet lock at
  all. It changes what the lock covers during a serve and is left to its own decision.
