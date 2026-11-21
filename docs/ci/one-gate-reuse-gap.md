# ONE-GATE: does mg-reuse hide guard sections? (finding, af12cc515)

## When x86=1 (read from ci.yml + scripts/ci_mg_reuse.sh)
- `mg-reuse` has `if: event == push && ref == main` (S1, 20X r6). On merge_group it is SKIPPED, so
  `needs.mg-reuse.outputs.x86` is '' (never '1'): x86-main AND `guards` both RUN on every queue entry.
  `resolve` also ends in `force_no_reuse` (x86=0 det=0). The premise "x86=1 skips both on merge_group" does not hold here.
- x86=1 happens only on the push to main (`decide_push`): P1-P4 need a completed merge_group run on the same
  sha whose x86-main, determinism, gate, `ci / gate` succeeded and x86-main-advisories skipped. `guards` has
  `github.event_name == 'merge_group'`, so it never runs on push; on a reused push nothing re-runs guard sections.
- That is sound only because `gate` (ci.yml ~L1050) requires `needs.guards.result == success` on merge_group
  and reads GRD guard-tree/guard-cargo. The parity script's per-event section diff (S0-S3) already covers
  "push section the queue drops".

## What PR event runs
Only pr-light. guard-cargo-shard/guard-cargo/vendored-schemas first run in the queue (`guards`), by design.

## The real gap (not a skip): different COMPARAND per event
`include_fmt_ratchet.sh` -> lib_baseline_ratchet.sh compares against merge-base(HEAD, origin/main). The same
class (every ratchet "vs origin/main") sees a different base on PR-less push (HEAD == origin/main, so the
diff is empty or self) than in the queue. A queue-green / push-red is therefore a baseline or env difference
(rustfmt pin, checkout depth, origin/main ref), not a skipped section. Not yet measured: pull the failed push
run's include_fmt step log to confirm which.

## Proposal
1. Parity script: add a case that mutates `guards.if` to include `outputs.x86 != '1'` AND makes mg-reuse
   run on merge_group; must go RED (a reuse skip of guards without a PR-side run). Today unmodeled because
   mg-reuse outputs resolve to '' (benign while S1 holds). Add a standing check: mg-reuse `if` must stay push-only.
2. Run the guard-cargo ratchet sections in pr-light (or a PR `guards-lite`) so the first run is pre-queue.
3. Ratchet scripts: on push compare against `$GITHUB_EVENT_BEFORE`, not origin/main (self-compare).
