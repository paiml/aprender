# PMAT-4099 implementation receipt: the 8 bashrs SEC001/SEC010 findings fixed, not suppressed

Issue: paiml/aprender#4099 (0.70.0). Base: `origin/chore/0.69.1-merge-back` @ f1b09d4d6 (the
flagged scripts exist only there until #4046 lands). The PR goes to main once #4046 merges (cop ruling).
Line ranges were coordinated with aprender-6c's pending #4046 push, and they do not overlap.

## Must-RED (measured 2026-09-24, bashrs 7.4.1, the dogfood's `lint --no-ignore --level error`)
Removing the 16 suppression lines (8 × reason + `disable-next-line`) from the base reproduces
exactly the 8 findings #4099 lists:
`check_crux_ollama_in_lock.sh:152 SEC001`, `check_ladder_serve_teardown.sh:119,191 SEC001`,
`check_moe_routes_through_one_dispatch.sh:183 SEC010`, `crux_inference_dogfood.sh:628,673 SEC010`,
`crux_sweep_shards.sh:54 SEC010`, `model_ladder.sh:1358 SEC010`. With this change: **0** on those files, and
0 gating SEC/DET/IDEM errors over all 412 dogfood-surface files. `check_bashrs_gate.sh` PASSes (412 files, 0) and
`check_shell_lint_ratchet.sh` passes.

## Fixes (no suppression remains)
| site | fix |
|---|---|
| ollama stub stats (SEC001) | bashrs matches the word `eval` even inside a quoted string. ollama's `--verbose` timing block now lives in `tests/fixtures/crux/ollama-verbose-stats.txt` and the stub `cat`s it (byte-identical output) |
| teardown `eval "$body"` ×2 (SEC001) | the extracted function bodies (and both self-test plants) are written to files under the script's mktemp dir; `load_under_test` runs `bash -n` and then sources them. That adds a parse check `eval` never had; the teardown plant now also gets the `bash -n` pre-check the wait plant already had, since `if run_cases` would read a parse failure as "caught" |
| moe `cp --parents -t` (SEC010) | the tainted input was `--root`, not `$T`: `ROOT` is canonicalized with `realpath -e` and must be a directory, so a missing root is refused up front (rc 2, naming the argument) instead of mid-copy |
| crux dogfood `mkdir` ×2 (SEC010) | the cell root `$SHA12` (sha prefix + whitelisted mode) is asserted not empty, absolute or `..` where it is made. `|| exit 2` after `decline` makes the rejection visible to bashrs, which recognises only `exit/return/die/fatal/abort/continue` as rejections. The second site is covered by the same guard (its `$d` is built from literals plus `$SHA12`) |
| shards `mkdir -p "$OUT/shards"` (SEC010) | `--out/--host/--backend` refused with a `..` segment (they name every output path) |
| ladder receipt `mv` (SEC010) | `--out` refused with a `..` segment at parse; the receipt name must be one path segment (`ladder_write_decline … \|\| exit 2`) |

## Mutant table: each fix is load-bearing (revert it → the finding returns)
KILLED ×6: ollama stats back inline → SEC001; teardown `eval` restored → SEC001; moe realpath dropped → SEC010;
crux `|| exit 2` dropped → SEC010 ×2; shards guard disabled → SEC010; ladder `|| exit 2` dropped → SEC010.

## Behaviour
- guards: shards `--out a/../b` → rc 2, `--host ../x` → rc 2; ladder `--out a/../b` → rc 2 (a clean `--out`
  proceeds into the dry run); moe `--root /nonexistent/x` → rc 2 naming it, `--root <file>` → rc 2,
  `--root scripts/..` → OK.
- checks on this branch: check_crux_ollama_in_lock 6 ok/0 broke; check_ladder_serve_teardown PASS and `--self-test` OK;
  check_moe_routes_through_one_dispatch OK; ladder write_errors / only_selection / provenance / output_judged /
  serve_verdict, crux_greedy_rows, dogfood_no_defer, serve_backend_record, serve_probe_evidence all pass.
- Pre-existing RED, identical on the unmodified base f1b09d4d6: `check_moe_routes_through_one_dispatch.sh --self-test`
  (a mutant stays green: f5's re-anchor 6c0a4cf04, riding 6c's #4046 push) and `check_model_ladder.sh` (audits
  committed receipts, EPIC #3477).

## Not proven
The crux dogfood cell-root guard cannot be reached from the command line (`--thinking-modes` is whitelisted at
parse and `$SHA12_MODEL` is a sha prefix), so it is defense in depth, not a behaviour change.
