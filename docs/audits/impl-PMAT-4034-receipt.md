# PMAT-4034 receipt: the ladder's CPU lane (branch feat/4034-cpu-lane-unlocked)

Base: 0dccce21b (the #4046 merge-back head). Commits: d2ac64ec6 (lanes), ce6aabdee/009d2650f (entry),
c60ac43be (concurrency opt-in per the cop's ruling), 67f396654 (header).

## What ships
- `apr_lane <backend>`: the cpu lane calls `apr_cpu_unlocked`, which takes no flock, runs with
  CUDA_VISIBLE_DEVICES set and EMPTY (it cannot open a CUDA context), keeps choom 1000, and runs
  niced. Every other lane calls `apr_locked`, unchanged. Measured with apr 0.69.1 fc942f6be: with
  the devices hidden, `--no-gpu` run and chat exit 0, print no fallback string, and chat reports
  `{"requested":"cpu","ran":"cpu","fell_back":false}`.
- The per-backend loop body is `ladder_backend_cell` (it prints its fragment), and
  `ladder_run_lanes` assembles be_json. A lane with no parseable single-key fragment becomes an
  explicit RED fragment (ran:false, lane_error), never absent.
- Concurrency is OPT-IN (`MODEL_LADDER_CONCURRENT_LANES=1`; cop ruling (a)). The default is serial.
  Opted in, the cpu lane is started first in the background. A decline inside it still ends the
  ladder with rc 2, and the EXIT trap kills a still-running background lane.

## Evidence
- scripts/check_ladder_cpu_lane.sh (guard_tree-dispatched): 12 cases, and 11 --self-test mutants
  each killed by the named case. Its first run caught two real bugs in this change before commit:
  an empty fragment parsing as `{}` (the backend was dropped), and a `cuda,cpu` rung that started
  its cpu lane only after the GPU lane had finished.
- check_model_ladder.sh --self-test: 302 ok / 0 FAIL, the same as the base. The sibling guards
  (serve_teardown, serve_verdict, serve_probe_evidence, serve_backend_record, output_judged,
  write_errors, only_selection, provenance), check_guards_are_wired and check_apr_bin_pinned all
  pass. The wired bashrs gate (scripts/check_bashrs_gate.sh, `--no-ignore --level error`) reports
  the SAME 8 findings as the base 0dccce21b (all pre-existing, none in this diff's files). An
  earlier revision of this receipt said "bashrs: 0 errors" from a plain `bashrs lint`, and that
  was wrong: quorum lane 3 measured a new DET002 at check_ladder_cpu_lane.sh:140 (`date` into a
  file), now removed ($EPOCHREALTIME).
- Yoga A/B, qwen35-2b (#4034 comment 5800622966): no measurable saving from concurrency at 2B,
  hence opt-in. The measured win that ships on by default is that the CPU lane never holds the
  fleet lock (the base's CPU serve held it ~30 s per cell).

## Found along the way (filed, not in this diff)
- #4055: the serve teardown refused to kill its own survivor from a symlinked cwd (fix on its
  own branch, quorum AGREED).
- #4090: the health wait reads a GPU-lock wait as a stalled server.

## Round-2 quorum findings, fixed at the next head
- (lane 2, measured) On the DEFAULT serial path, a foreground lane's stderr was captured into
  $WORK/<rid>.<b>.lane.err. lock_timeout's `exit 2` ended the ladder before the replay line, and
  the EXIT trap deleted $WORK, so a decline lost its reason (including the lock holder's pid).
  Foreground lanes now write stderr straight through, as the old loop body did. New case
  `fg-decline-keeps-reason` and mutant `fg-stderr-captured`; `cuda-decline-kills-cpu-lane` now
  also asserts the reason text.
- (lane 2) measure()'s dead locals (flag run_out run_rc fb ran) removed.
- Guard flake, disclosed: 1 of 26 runs lost its whole case directory mid-run (`concurrent`) and
  never reproduced (0/25 after, 12 of them 4-way parallel). The case failed RED, which is the
  safe direction. The guard now uses a distinctive mktemp template.

