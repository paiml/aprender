# PMAT-4090 receipt: a GPU-lock wait is not a stalled server (branch fix/4090-lock-wait-not-a-stall)

Stacked on fix/4055-teardown-physical-cwd (c9930ef82), because both edit
scripts/check_ladder_serve_teardown.sh. Base of the stack: 0dccce21b (the #4046 head).

## The defect
A serve launched under apr_locked is, until the lock frees, a bare `flock` that writes no log
and burns no CPU. ladder_serve_wait_health's progress detector (#3943 b) called that `stalled`,
and the cell recorded `serve NOT PROBED: ... stalled after 90s ... the serve log was empty`: a
model RED for an ENV condition. This was reproduced by construction with the shipped function
(issue body). The same signature appeared once on yoga (#4034 A/B, B' leg) with the holder
unsampled.

## The fix (scripts/model_ladder.sh)
- ladder_tree_waits_on_lock <pid>: true when a pid of the wrapper's tree is a WAITER on the
  $GPU_LOCK inode in /proc/locks (`N: -> FLOCK ... <pid> <maj:min:inode>`).
- ladder_serve_wait_health: while that is true, neither the stall counter nor `waited` runs.
  flock's own `-w LOCK_WAIT` bounds the wait.
- ladder_serve_probe: a wrapper that `died` with LOCK_BUSY (75) never started a server, so it calls
  lock_timeout (rc 2, the holder named from /proc/locks).
- The cell: the probe runs inside $( ), so a probe rc 2 whose stderr carries `decline: ` now
  ends the ladder with rc 2 (the reason was already replayed to stderr).

## Evidence
- check_ladder_serve_teardown.sh:
  - lock-wait: a foreign holder keeps the lock 6 s, stall window 3 s; ends `ready 1` (the clock
    ran only after acquisition).
  - serve-lock-busy-declines: the SHIPPED ladder_serve_probe, lifted with its dependencies, runs
    with the lock held past LOCK_WAIT=2. It exits 2 with
    `decline: ENV the GPU lock <lock> was not free after 2s for apr serve run r1 (cuda) -- holder: pid <holder>`.
  - --self-test: 6 plants, each turning its case RED. The new ones are the lock-blind wait
    (lock-wait RED) and LOCK_BUSY ignored (serve-lock-busy-declines RED). The log-only plant
    still turns exactly slow-cpu RED.
- check_model_ladder.sh --self-test 302 ok / 0 FAIL. serve_verdict, serve_probe_evidence,
  serve_backend_record, output_judged, write_errors, only_selection, provenance and
  apr_bin_pinned: all rc 0.
- check_bashrs_gate.sh: 8 findings, the same set as base 0dccce21b (the two `eval "$body"`
  SEC001s in check_ladder_serve_teardown.sh are pre-existing and shifted). None added.

## Fold note
The #4034 branch moved the per-backend loop into ladder_backend_cell. The 3-line decline
carry-out below `serve_rc=$?` belongs in that function when both land.
