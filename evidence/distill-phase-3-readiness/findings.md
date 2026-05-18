# Phase 3 Dispatch Readiness Audit — Pre-cascade-land snapshot

**Date:** 2026-05-18
**Branch tip:** `feat/distill-cuda-backend-construction` @ ca6cc8223
**Cascade in flight:** PRs #1787 / #1788 / #1791 / #1792 / #1793 / #1795 / #1796 / #1797 — all OPEN/BLOCKED, auto-merge SQUASH armed, 0 failures
**Trigger:** dry-run of `scripts/dispatch-distill-phase-3-gx10.sh` validated, but a deeper read of the `apr distill` invocation reveals a flag-surface gap.

## Defect summary

The Phase 3 dispatch script (`scripts/dispatch-distill-phase-3-gx10.sh`, shipped in PR #1795) invokes `apr distill` with **six flags that do not exist** on the current CLI surface:

| Script flag           | Current CLI flag (Phase 3-prep) | Status                          |
|-----------------------|---------------------------------|---------------------------------|
| `--num-steps 500`     | (none)                          | ❌ no CLI surface                |
| `--batch-size 4`      | (none — uses default 32)        | ❌ no CLI surface                |
| `--learning-rate 1.5e-5` | (none — uses default 1e-4)   | ❌ no CLI surface                |
| `--student-init <ID>` | `--student <PATH>` (PathBuf)    | ⚠ name mismatch + type mismatch |
| `--output-dir <DIR>`  | `--output <PATH>` (file path)   | ⚠ name + semantics mismatch     |
| `--device cuda`       | `--backend cuda`                | ⚠ name mismatch                 |

When the cascade lands and the script is invoked, `apr distill` will reject the unknown flags and exit non-zero before any training begins. The smoke run cannot fire as-shipped.

## Root cause

PRs #1796 + #1797 (Phase 3-prep) added `--backend` and wired `run_cuda_backend()` but stopped at the minimum CLI surface needed for a fixture-vs-cuda backend switch. The dispatch script (#1795) was authored ASPIRATIONALLY against the full Phase 3 surface, on the assumption that the same PR cascade would add the missing flags. It didn't — the missing flags fell into the gap between #1797 and a future Phase 3 ticket.

## Local proofs

```
$ /mnt/nvme-raid0/targets/aprender/debug/apr distill --help
…
  --epochs <EPOCHS>            Training epochs [default: 3]
  --backend <BACKEND>          SPEC-DISTILL-001 Phase 3-prep …
```

No `--num-steps`, no `--batch-size`, no `--learning-rate`. Confirmed at branch tip `ca6cc8223` after a forced rebuild (cargo feature-cache staleness rule applied — `target/debug/apr` was hard-linked to a stale binary in `/mnt/nvme-raid0/targets/aprender/`; fresh binary at `/mnt/nvme-raid0/targets/aprender/debug/apr` has the Phase 3-prep `--backend` flag but nothing further).

## Recommended fix scope (Phase 3 CLI surface, new ticket — call it PMAT-698b)

Add three flags to `apr-cli` `Distill` enum variant:

1. `--num-steps <N>` (Option<u32>): if set, caps total training step count; pipeline ignores `--epochs` when this is set
2. `--batch-size <N>` (Option<u32>): override `DistillConfig::training.batch_size` (currently hardcoded to 32 via default)
3. `--learning-rate <F>` (Option<f64>): override `DistillConfig::training.learning_rate` (currently hardcoded to 1e-4)

Thread through `distill::run(...)` → `run_cuda_backend(...)` → `DistillConfig::training` overrides. Add `--device` as an alias for `--backend` (same enum). Accept `--student` with either a path OR an HF repo ID (auto-resolve via cache lookup).

Also: pipeline `train()` loop currently has `steps_this_epoch = (1000 / batch_size).max(1)` hardcoded — needs to honor `max_steps: Option<u32>` if set.

## Falsifier (F-DISTILL-CLI-PHASE-3-FLAGS, proposed)

A unit test in `crates/apr-cli/src/commands/distill.rs::tests`:

```rust
#[test]
fn falsify_phase_3_dispatch_script_flag_surface_aligned() {
    // Read the dispatch script, extract the `apr distill ...` invocation,
    // tokenize it, and assert that every `--flag` token appears in the clap
    // matches of `apr distill --help`.
    let script = include_str!("../../../../scripts/dispatch-distill-phase-3-gx10.sh");
    let flags_used: Vec<_> = extract_apr_distill_flags(script);
    let cli_flags: Vec<_> = extract_distill_clap_flags();
    for f in &flags_used {
        assert!(cli_flags.contains(f), "dispatch script uses {} which is not on apr distill CLI", f);
    }
}
```

This contract pins script↔CLI alignment forever.

## Resolution (post-cascade)

Cascade drained 2026-05-18 18:24 UTC:
- #1788 squash-merged Phase 2 (`11a0ba77f`)
- #1797 squash-merged the rest of the cascade (`aee8716d6`) per chain-PR squash leapfrog
- #1787 / #1791 / #1792 / #1793 / #1795 / #1796 closed as subsumed

This findings.md ships with PR PMAT-698b (`fix/distill-phase-3-dispatch-flags-pmat-698b`), which fixes the dispatch script to use the EXISTING CLI surface rather than the aspirational one. The CLI additions (`--max-steps`, `--batch-size`, `--learning-rate`) are deferred to a future PMAT-698c after the smoke run validates the pipeline end-to-end with default hyperparameters.

### Script fix (this PR)

| Aspirational flag         | Fixed to                              |
|---------------------------|---------------------------------------|
| `--teacher REPO`          | positional `<TEACHER_DIR>` (resolved) |
| `--student-init REPO`     | `--student <STUDENT_DIR>` (resolved)  |
| `--num-steps 500`         | `--epochs 17` (round-up of 500/31)    |
| `--batch-size 4`          | dropped — uses default 32             |
| `--learning-rate 1.5e-5`  | dropped — uses default 1e-4           |
| `--output-dir DIR`        | `--output DIR/student.apr`            |
| `--device cuda`           | `--backend cuda`                      |

HF repo IDs are resolved to local cache dirs via shell function `hf_repo_to_dir` inside the SSH heredoc — `~/.cache/huggingface/hub/models--<sanitized>/snapshots/<sha>/` is what `apr distill` expects (per `for_inference()` signature).

### Deferred to PMAT-698c (future PR)

Adding `--max-steps`, `--batch-size`, `--learning-rate` CLI flags lets the dispatch use the user's preferred hyperparameters (batch=4, lr=1.5e-5) instead of defaults. The pipeline.rs training loop also needs `max_steps: Option<u32>` to cap the loop. Scope: ~150 LOC + falsifier.

## Cascade state at audit time

```
#1787: OPEN/BLOCKED  S=3 P=3 F=0   (Phase 1b — CudaTrainerTeacher)
#1788: OPEN/BLOCKED  S=6 P=1 F=0   (Phase 2 — KD step) ← closest to merge
#1791: OPEN/BLOCKED  S=5 P=2 F=0   (Phase 2b — StudentLogitsProvider)
#1792: OPEN/BLOCKED  S=5 P=2 F=0   (Phase 2c — pipeline integration)
#1793: OPEN/BLOCKED  S=4 P=2 F=0   (Phase 2d — CudaStudentProvider)
#1795: OPEN/BLOCKED  S=2 P=4 F=0   (Phase 3 dispatch + watch scripts) ← contains the defect this audit identifies
#1796: OPEN/BLOCKED  S=5 P=2 F=0   (Phase 3-prep — --backend flag)
#1797: OPEN/BLOCKED  S=5 P=2 F=0   (Phase 3-prep — cuda backend construction)
```

All 8 PRs target `main` as siblings (not a daisy chain). Per the `feedback_chain_pr_squash_leapfrog.md` memory rule, when one squash-merges with the cascade content, GitHub will mark subsumed siblings as MERGED automatically.

## Lesson #N+1 candidate (NEW)

**Dispatch scripts authored mid-cascade must be flag-aligned with the CLI on the same branch as the script.** PR #1795 added a script that depended on a CLI surface PR #1796/#1797 didn't provide. The aspirational script + minimal-surface CLI shipped in different PRs, leaving the dispatch broken until a follow-up PMAT-698b lands.

Mitigation: every PR that ships an executable script under `scripts/` must include either:
- a smoke test that runs the script's invocation against the binary built from the same branch (DRY_RUN=1 + `--help` shape check), OR
- a `pv lint` contract assertion that the script's `apr <subcommand>` flags ⊆ the binary's clap matches

Without one of these, scripts drift silently.
