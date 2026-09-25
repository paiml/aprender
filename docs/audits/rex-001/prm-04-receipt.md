# PRM-04 — cell admission re-run under `parity.oracle` (rex-cell-admission-v1 1.1.0)

**Outcome: receipted refusal. 0 of 6 cells Admitted; S-7 stands** (`rex admission-check` exit 10).
C4 has a passing declared-oneshot parity receipt, but it measured 7 positions, and admission requires 64.

## What changed

- `admission::parity_from_receipt` derives an `Admitted` parity block from a receipt's bytes and never from typed numbers. It is covered by FALSIFY-RCA-004..006.
  - It reads two receipt shapes:
    - the `apr-review-serve parity` oneshot (comparator `apr-cpu…`, recorded as oracle `apr-parity-gpu-cpu`, the §4.1 rule-1 interim oracle);
    - `apr-parity-oracle/v1` from `apr parity-oracle` (#4444, 57's branch @03579caa9; oracle `llama.cpp@d1d3c3396`).
  - The cosine is the **minimum over the receipt's positions**. The receipt's summary `cosine` is never read.
  - The threshold and basis are the **declared** ones. A receipt's self-chosen threshold is never read.
  - It refuses on any of these:
    - fewer than `min_positions` positions;
    - exit ≠ 0, a verdict that is not passing, or failed > 0;
    - any position below the threshold, or a missing or non-finite cosine;
    - an undeclared oracle, an empty basis, or a non-finite threshold;
    - a binary or weights sha that differs from the row's.
- `rex admit --cell C --parity-receipt R --threshold T --threshold-basis B --min-positions N` appends the row. It admits one cell at a time, never `all`. When the receipt does not admit, it writes nothing, exits 1 and names the reason.

## The run

| Cell | State | Why |
|------|-------|-----|
| C1 intel/wgpu, C2 intel/cpu, C3 lambda-labs/cpu, C5a mini/cpu, C5b mini/metal | NotRun{NoDeclaredExecutor} | unchanged since REX-04; resumes on infra#1088 |
| C4 gx10/cuda | NotRun{Inadmissible} | `rex admit` refused: `7 positions < min_positions 64` |

The C4 evidence is `prm-04/gx10-parity-20260925T124817Z.json`, sha256 `54e2b72a12f2413e8b82ecb1a93b38a7b01a3ea671c5d4702365328b60844b67`. It was read from gx10's declared state dir `~/.local/state/apr-review-serve/parity/`.
- It is the output of the declared oneshot `apr-review-serve parity`.
- Binary: apr v0.69.3 `ad5d07a1`, binary sha `2f274f47…`.
- Weights: Qwen3.5-4B-Q4_K_M, sha `00fe7986…`.
- Result: exit 0, verdict pass, 7/7 positions passed, min cosine **0.99970** against the 0.98 threshold.
- **The only failing condition is the evidence floor.**
  - `evidence/parity/thresholds.yaml` sets `min_positions: 64` (basis I8 / CF-4, #1864: "any autoregressive gate validates over >= 64 positions").
  - The 0.98 default was itself measured over ≥ 64 positions.
  - A 7-position smoke check cannot use a threshold whose basis is 64 positions.

Positive control: the same receipt with `--min-positions 7` does admit (cosine 0.9997043609619141, oracle `apr-parity-gpu-cpu`). So the refusal is the floor and nothing else. The row it produced was a scratch file and is not part of this audit.

## Why not the llama.cpp oracle

`apr parity-oracle` (#4444) has only been measured on Qwen3.5-**0.8B**. The prereg-locked model is **4B**. Its threshold row `qwen3.5-0.8b@cpu-vs-llama.cpp` deliberately has no `min_cosine`. No 4B llama.cpp reference exists yet, so this oracle cannot admit a cell.

## To admit C4 (resume conditions)

1. The declared oneshot must measure at least 64 positions. The fix is a longer parity prompt, such as the 78-token corpus prompt the 0.98 basis used. Then re-run:
   ```
   rex admit --cell C4 --parity-receipt <new receipt> --threshold 0.98 --threshold-basis "evidence/parity/thresholds.yaml default.min_cosine" --min-positions 64 --apr-sha <binary_sha256> --weights-sha 00fe7986…
   ```
2. The serve declaration (infra e872544c) and the parity oneshot (infra 3f66d253/3d59dd3e) are live on gx10, but only through the open infra PRs #1142, #1154 and fold #1054. **They are not on infra main**, and infra#1088 is open. An Admitted C4 still rests on a declaration that has not landed.
3. H1 (verdict invariance) is evaluated in the analysis (PRM-08), not at admission.

## Note

The prereg lock the tool reads today is `f0b4e4ac…`, while the REX-04 rows carry `ef51087d…`. This file uses the current lock. REX-04's file is left untouched.

## Gates

- `cargo test -p aprender-review-experiment --lib`: 167 passed.
- `cargo clippy -p aprender-review-experiment --all-targets -D warnings`: clean.
- `pv validate contracts/rex-cell-admission-v1.yaml`: valid.
- `cargo mutants` on `admission.rs`: see the commit message.
