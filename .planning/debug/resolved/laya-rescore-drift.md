---
slug: laya-rescore-drift
status: resolved
trigger: "Rust Laya/ModernBERT re-score drifts from torch beyond the 1e-5 parity bar on real checkpoints (phase 08, plan 08-09 halted at stop rule)"
goal: find_and_fix
created: 2026-09-26
updated: 2026-09-26
---

# Debug Session: laya-rescore-drift

## Symptoms

**Expected behavior**
Plan 08-09's `pack` re-scores all 280 eval rows of a Laya run dir in Rust (from the packed
.apr bytes) and matches the torch-written probabilities within the D-17 bar
(`pack_rescore_probs_abs` = 1e-5, laya-parity-v1; `logits_abs` = 1e-4). For the
fail-closed vectors it should then refuse with `REFUSED GateFailed clauses=[ece_post]`
(exit 3).

**Actual behavior**
```
just laya-pack models/decide/tweet-stance-16 data/decide/tweet-stance-16 <base> models/decide/fail-closed-check.apr
REFUSED RescoreDrift which=zero_shot row=49 max_abs=0.000010281801223754883 (nothing written)
exit 2, wall 184s
```
Measured by the 08-09 executor over all 280 rows (throwaway example, not committed):

| Re-score | max abs dp | rows > 1e-5 | median | argmax |
|---|---|---|---|---|
| early_stopping fine-tuned (`models/decide/tweet-stance-16`, recipe 3d4b91da) | 6.26e-6 | 0 | 6e-7 | 280/280 |
| zero-shot base (same base in both run dirs) | 1.028e-5 | 1 (row 49; ~7.7e-5 in logits) | 7e-7 | 280/280 |
| fixed_epochs fine-tuned (`models/decide/tweet-stance-16-fixed-epochs`, recipe d0f4e40d) | 5.48e-5 | 17 | 6e-8 | 280/280 |

fixed_epochs row 147 is ~1.8e-3 off in logits, which breaks `logits_abs` 1e-4 by 18x.

**Error messages**
`RescoreDrift which=zero_shot row=49 max_abs=1.028e-5` (pack exit 2).

**Timeline**
- Spike 025 measured Rust vs torch at 3.8e-6, but only on 14 rows.
- The tiny-fixture parity (plans 08-03/08-04) passes at ~1e-7.
- 08-05 packed the real early_stopping run dir; its few probe rows agreed within 1.26e-6.
- A /simplify refactor, commit 71e2306e5, landed between 08-05 and 08-09. It hoisted
  the RoPE sin/cos tables, rotated the fused qkv in place with attention reading at stride 3d,
  replaced a hand-rolled transpose with `trueno::blis::transpose`, and cached Laya's row
  prefix. Its bit-identity proof covered the tiny fixture only.
- The first full 280-row real-checkpoint re-score is this 08-09 run.

**Reproduction**
`just laya-pack models/decide/tweet-stance-16 data/decide/tweet-stance-16 "$HOME/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851" models/decide/fail-closed-check.apr`
(~3 min on aarch64). Plan 08-09's code is committed in d7031c319 (`src/verify.rs`,
`examples/pack_laya.rs`).

**Facts already established**
- Both sides are CPU fp32 over the F16 reload; train.py scores every file on CPU.
- The Rust side is deterministic: two zero-shot runs in one process were bitwise identical.
- The gate recomputed in Rust from the stored files matches both reports within 1e-5. If
  the re-score bar held, both vectors would fail on `ece_post` alone.

## Constraints

- Do NOT widen any tolerance. D-17 and laya-parity-v1 stay as written (user decision 2026-09-26).
- The run dirs under `models/decide/` are gitignored evidence. Never modify, retrain or
  overwrite them. MPS training is not bitwise reproducible.
- Base snapshot: `~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
- The Python oracle env is `scripts/laya_train` (`uv run --frozen`). `common.py` holds the
  shared helpers, and `train.py`'s Scorer is how torch probabilities were produced.
- Any fix must keep green:
  - the tiny parity tests (`aprender-core` modernbert, `aprender-decide` laya)
  - the golden pack hash `37d65159…`
  - `just laya-fixtures` byte-identical
- Project rules: see CLAUDE.md (unwrap banned; `rtk proxy` for unfiltered cargo output;
  never read `$?` through a pipe; clippy per crate with `--no-deps`).
- The working tree is shared. Stage only the files you change. Leave `.planning/state.json`
  and the untracked files alone.

## Current Focus

bug_class: "Bohrbug (deterministic; both sides reproducible bit for bit)"
hypothesis: "ROOT CAUSE SET (AND-gate): (1) the 1e-5 prob / 1e-4 logit bar sits at or below the fp32 noise floor of real Laya checkpoints (torch's own fp32 answer is 3.7e-5 from exact on 13 fixed_epochs rows); AND (2) Rust's RoPE inv_freq was single-rounded from f64 while torch double-rounds in f32, a systematic RoPE-table deviation (fixed); AND (3) for the marginal zero-shot row, any 1-ULP perturbation (the 71e2306e5 sincos codegen change, or the inv_freq fix) moves it across the bar."
test: "DONE: inv_freq fix + regression test (RED on old code at exactly p=[6,14,23,25,26,27,28,31], GREEN on new); real laya-pack on ES and FE, reverted and reapplied."
expecting: "ES -> GateFailed[ece_post] exit 3 (observed); FE -> RescoreDrift (observed, unavoidable under this bar)."
next_action: "NONE — session resolved 2026-09-26. Fix committed 8e55e0bed (layer.rs only). User decisions applied: (1) fix accepted; (2) option A, 1e-5 bar kept, fixed_epochs recorded as a RescoreDrift refusal (exit 2, fail-closed). Plan 08-09's executor applies outcome A to its plan/SUMMARY; the bar-fragility question is queued on the calibration spike todo by the orchestrator."
scratch: "/private/tmp/claude-501/-Users-guy-Development-machine-learning-aprender/8686affb-84f9-4941-8b24-dd1ff10ba8b8/scratchpad/dump/"
scratch_tools: "torch_dump.py (Scorer + ladder), torch_f64.py (f64 forward), torch_local.py (per-layer local error), triad.py, cmp_local.py; the three Rust throwaway examples (zz_rescore_dump / zz_local_dump / zz_h0_dump) were REMOVED from crates/aprender-decide/examples and kept in scratchpad/dump/examples/ — copy back to reuse"

reasoning_checkpoint:
  hypothesis: "The pack refusal is an AND of a bar at the fp32 noise floor and a systematic RoPE inv_freq rounding mismatch; the refactor only moved the marginal row by a 1-ULP sincos codegen change."
  confirming_evidence:
    - "f64 forward of the same model: torch fp32 is 3.7e-5 (13 rows > 1e-5) from exact on fixed_epochs, so any non-bit-identical implementation fails there"
    - "per-sub-op local error of Rust == torch at all 28 layers; ids 280/280 equal"
    - "transformers' inv_freq bits reproduced 32/32 by f32(1/f32(theta^e)); Rust's old form off by 1 ULP at 8/32 and 10/32"
    - "revert/reapply of the fix on the real ES pack: RescoreDrift zero_shot row 49 1.0281801e-5 <-> GateFailed[ece_post] with zs 6.8e-6"
  falsification_test: "If the fixed build still refused ES on RescoreDrift, or if torch fp32 were within 1e-5 of the f64 forward on every fixed_epochs row, the root-cause set would be wrong."
  fix_rationale: "The inv_freq change removes a real deviation from the reference semantics (the code claimed 'torch order' and did not implement it). It does not, and cannot, make fixed_epochs pass; that part is a contract decision the user reserved."
  blind_spots: "ES/ZS now pass at 6.7e-6 / 6.8e-6, a 1.5x margin that a 1-ULP change elsewhere has been shown to consume; x86_64 (Lambda) not measured; sin/cos still differ from torch's vectorized kernels by 1 ULP on 3-9% of entries."
  candidate_causes:
    - "code: RoPE inv_freq single vs double rounding (CONFIRMED, small)"
    - "code: 71e2306e5 hoist changed LLVM sin+cos -> __sincosf_stret fusion (CONFIRMED, 1 ULP, flips row 49)"
    - "config/contract: 1e-5 / 1e-4 bars fitted from 14 spike rows on the base, below the fine-tuned model's fp32 noise floor (CONFIRMED)"
    - "data/model: fixed_epochs checkpoint is ill-conditioned (T=5, logits to 22, residual activations to 2.2e4) (CONFIRMED as amplifier)"
  and_gate: "yes: the ES refusal needed the marginal noise draw AND the systematic RoPE offset; the FE refusal needs only the noise floor."

## Evidence

- timestamp: 2026-09-26 (session 2)
  checked: "torch re-run of fixed_epochs checkpoint through train.py's Scorer (scratchpad/dump/torch_dump.py --all) vs models/decide/tweet-stance-16-fixed-epochs/eval-probs.json"
  found: "Bitwise identical on all 280 rows. zero-shot-probs.json is also byte-identical across the two run dirs. torch CPU fp32 scoring is reproducible, so the stored reference is exactly what torch computes today. (Caveat found: forward hooks on nn.TransformerEncoderLayer disable torch's fused fast path; the hooked run of row 147 differs from the stored value by ~1.2e-7 — harmless here, but ladder dumps with hooks are the slow path.)"
  implication: "The reference is stable; nothing about the torch side is flaky."

- timestamp: 2026-09-26 (session 2)
  checked: "Rust HEAD re-score of fixed_epochs through a throwaway example (crates/aprender-decide/examples/zz_rescore_dump.rs, uncommitted) that feeds torch's own ids/markers to Laya::forward_row"
  found: "Rust builder ids+markers == torch ids+markers on 280/280 rows. Rust reproduces the executor's numbers exactly: probs max 5.484e-5 at row 147, 17 rows > 1e-5, logits max 1.666e-3 at row 147, 45 rows > 1e-4, argmax 280/280. Row lengths 67..108 (all > 65, so the local window bites on every row; none > 129)."
  implication: "Tokenization / builder / row-prefix caching are eliminated as a cause. The drift is purely numeric, in the forward."

- timestamp: 2026-09-26 (session 2)
  checked: "float64 torch forward of the same model on the same ids (scratchpad/dump/torch_f64.py) as ground truth; both fp32 paths compared to it over all 280 rows"
  found: "torch fp32 vs f64: logits max 1.578e-3 (row 78), median 1.61e-5, mean 7.20e-5. Rust fp32 vs f64: max 1.990e-3 (row 72), median 1.85e-5, mean 8.98e-5. Rust error > torch error on 169/280 rows. Row 147: torch err 2.21e-4, Rust err 1.89e-3."
  implication: "This model's fp32 noise floor is ~1e-3 in logits even for torch itself (massive activations: residual stream reaches 2.2e4). But Rust is systematically ~25% worse on average and ~9x worse on the drift rows, so there is a real, localizable accuracy deficit in Rust — not just noise."

- timestamp: 2026-09-26 (session 2)
  checked: "Per-layer, per-token error vs f64 for row 147 (fixed_epochs), torch fp32 vs Rust"
  found: "Layers 0-17: Rust is as accurate as, or more accurate than, torch at every token. From about layer 18 on, Rust's error grows much faster. Token 32, a [MASK] marker: layer21 Rust 7.8e-4 vs torch 2.3e-4; layer23 3.0e-3 vs 3.0e-4; layer27 1.12e-2 vs 7.3e-4 (15x). Token 1: layer22 3.4e-4 vs 6.6e-5. After the final norm the marker token carries 9.5e-4 abs error (torch 4.7e-5), and that flows through head0/head1/scorer into the logits."
  implication: "Localized to the late encoder layers (18+). Next: measure each layer's LOCAL error (f64 reference input rounded to f32 -> one layer in torch fp32 and in Rust -> compare with the f64 layer output) to find which op inside the layer is less accurate."

- timestamp: 2026-09-26 (session 2)
  checked: "Per-layer LOCAL error, row 147 fixed_epochs: each layer fed the f64 reference's previous output rounded to f32; every sub-op (qkv, attn, attn_out, x_mid, mlp_in, wi, g, mlp_out, out) in torch fp32, torch f64 and Rust (scratchpad/dump/torch_local.py + crates/aprender-decide/examples/zz_local_dump.rs, uncommitted)"
  found: "Rust and torch fp32 have the SAME local error at every sub-op of every one of the 28 layers, both rel-rms over the row and max-abs at marker token 32 (e.g. layer 27 out: 6.5e-8 vs 6.4e-8 rel; attn 7.8e-7 vs 7.7e-7). No sub-op is less accurate in Rust. Also: a layer composed from core's public primitives in the PRE-refactor style (separate q/k copies + rope_rotate_half + attention()) is BITWISE identical to the post-refactor ModernBertLayer::forward on the real checkpoint for all 28 layers."
  implication: "There is no localized Rust precision defect. The late-layer growth on row 147 is the model amplifying an accumulated rounding difference (chaotic, heavy-tailed): Rust's walk hit an amplifying direction on rows 147/117/72, torch's own walk hit one on rows 78/132. The refactor's RoPE-table / strided-attention change is bit-identical on real weights."

- timestamp: 2026-09-26 (session 2)
  checked: "Three-way table, all 280 rows, for all three re-scores: torch fp32 (bitwise == stored file on 280/280 for each), Rust HEAD, and the float64 forward of the same model (ground truth)"
  found: |
    zero-shot (base, T=1.760): probs torch32-vs-f64 max 7.9e-6 (0 rows >1e-5); rust-vs-f64 1.18e-5 (1 row); rust-vs-torch32 1.028e-5 (1 row, row 49). Logit mean error vs f64: torch 8.22e-6, Rust 8.19e-6 (equal).
    early_stopping (T=3.144): torch32-vs-f64 9.5e-6 (0); rust-vs-f64 6.5e-6 (0); rust-vs-torch32 6.26e-6 (0). Logit mean error: torch 1.37e-5, Rust 1.36e-5 (equal).
    fixed_epochs (T=5.0): torch32-vs-f64 3.71e-5 (13 rows >1e-5); rust-vs-f64 6.27e-5 (16 rows); rust-vs-torch32 5.48e-5 (17 rows). Logits torch32-vs-f64 max 1.58e-3, 52 rows >1e-4.
  implication_note: "(see the H0 and RoPE entries below: the f64 reference inherits torch's f32 RoPE tables, so 'rust vs f64' includes Rust's RoPE-table mismatch)"
  implication: "DECISIVE. On fixed_epochs, torch's OWN fp32 answer is up to 3.7e-5 from the exact answer, on 13 rows. So even a perfect (exact) Rust implementation would fail pack_rescore_probs_abs = 1e-5 against the torch-written file on 13 rows, and logits_abs = 1e-4 on 52 rows. On the base, torch32 and Rust are two independent draws of the same ~8e-6 noise, and their difference crosses 1e-5 on one row. The bar sits at or below the fp32 noise floor of real Laya checkpoints; the tiny fixture (1e-7) and spike 025 (14 rows, 3.8e-6) under-sampled the tail. No Rust code change that keeps fp32 semantics can meet it on fixed_epochs."

- timestamp: 2026-09-26 (session 2)
  checked: "H0 test as planned: a version-neutral example (zz_h0_dump, Laya::from_parts on an in-memory .apr, Rust builder ids) built at c076fb0f5 in a scratch worktree (dep-info proves it compiled the worktree's pre-refactor layer.rs) and at HEAD; cross-process determinism checked for both (cmp identical)"
  found: "c076 and HEAD differ on 279/280 fe rows and 280/280 zs rows (logits up to 2.3e-3 fe, 2.4e-5 zs). zero-shot vs stored torch: c076 max 4.9e-6 (PASSES), HEAD 1.028e-5 (fails). fixed_epochs: c076 12 rows > 1e-5, HEAD 17. Builder ids identical in both (280/280 == torch). Feature sets identical (cargo tree -f '{p}|{f}'), no Cargo.lock version changes. c076 + HEAD's gemm.rs (trueno transpose) is bitwise == c076, so the transpose is exact. Ladder: emb identical, layer0 differs; per sub-op: qkv identical, q/k after RoPE differ (~0.6% of elements, 1 ULP). nm: the c076 binary imports ___sincosf_stret, the HEAD binary imports _sinf/_cosf."
  implication: "H0 is TRUE but only mechanically: hoisting the RoPE table into RopeTable::new made LLVM stop fusing sin+cos into Apple's __sincosf_stret, which differs from sinf/cosf by 1 ULP on some angles. A 1-ULP change on 0.6% of q/k elements flips the zero-shot vector from pass to fail. Neither build is more correct; this proves the bar sits at the noise floor."

- timestamp: 2026-09-26 (session 2)
  checked: "Rust's RoPE inv_freq vs transformers 5.17 compute_default_rope_parameters (`1.0 / (base ** (arange(0, dim, 2).float() / dim))`, all float32), and the resulting sin/cos tables for positions 0..127"
  found: "Rust computes inv_freq as (1 / theta^(2p/hd)) in f64 and rounds ONCE; torch rounds pow to f32 and then takes the f32 reciprocal (double rounding). Result: inv_freq differs by +/-1 ULP on 8/32 frequencies (theta 160000) and 10/32 (theta 10000). Bitwise reproduction of torch's inv_freq: f32(1 / f32(theta^e)) matches 32/32 for both thetas (pow via f64 powf->f32 and via libm powf agree). Downstream, with Rust's inv_freq only ~73-77% of sin and 80-88% of cos entries equal torch's, max |d| 1.9e-6; with torch's inv_freq, 96-97% sin / 91-93% cos equal, max |d| 6e-8 (1 ULP, libm vs torch's vectorized sin/cos)."
  implication: "A real, systematic, deterministic divergence from the reference semantics, separate from rounding noise: RoPE angles are off by pos x 1 ULP on a quarter of the frequencies, so sin/cos is off by up to ~32 ULP. The contract/doc comment ('inv_freq = 1 / theta^(2p / hd) narrowed to f32') describes the double-precision form, not what torch does. It is a candidate contributing cause of the drift; its SIZE must be measured by re-scoring with torch's inv_freq."

- timestamp: 2026-09-26 (session 2)
  checked: "H1 test: rope_inv_freq patched to torch order (f32(1 / f32(theta^e))), re-score of all three with zz_rescore_dump + triad.py"
  found: |
    zero-shot: rust-vs-torch32 max 6.83e-6 (0 rows > 1e-5; was 1.028e-5, 1 row). Mean logit disagreement 7.77e-6 -> 7.67e-6.
    early_stopping: 6.74e-6 (0 rows; was 6.26e-6). ES logits: one row at 1.01e-4 (logits_abs is not a pack check).
    fixed_epochs: 4.67e-5, 11 rows > 1e-5 (was 5.48e-5, 17 rows). Logits 44 rows > 1e-4 (was 45).
    Rust mean logit error vs f64 moved by noise only (zs 8.19e-6 -> 8.17e-6, es 1.36e-5 -> 1.42e-5, fe 8.98e-5 -> 7.71e-5).
  implication: "H1 CONFIRMED as a real but SMALL contributing cause: it removes a systematic deviation from the reference semantics and happens to bring the zero-shot vector under the bar, but the aggregate disagreement is still dominated by independent fp32 noise. fixed_epochs still cannot pass (torch's own error exceeds the bar on 13 rows)."

- timestamp: 2026-09-26 (session 2)
  checked: "The real reproduction, `just laya-pack` into the scratchpad (never into models/decide), with the fix, reverted, and reapplied"
  found: |
    early_stopping WITH fix: REFUSED GateFailed clauses=[ece_post] ... rescore_max_abs=6.735e-6 zs_rescore_max_abs=6.825e-6 argmax=280/280 (nothing written), exit 3, 142 s.  <- the plan 08-09 expected outcome
    early_stopping with the fix REVERTED: REFUSED RescoreDrift which=zero_shot row=49 max_abs=0.000010281801223754883, exit 2  <- the original symptom, bit for bit
    fixed_epochs WITH fix: REFUSED RescoreDrift which=fine_tuned row=59 max_abs=4.667e-5, exit 2
  implication: "The fix changes the ES vector's outcome to the planned GateFailed[ece_post]. The FE vector still refuses on the re-score, and by the f64 evidence it must under this bar: that needs a user decision (constraint: no tolerance widening)."

- timestamp: 2026-09-26 (session 2)
  checked: "Thread-count sensitivity of the fixed Rust re-score (RAYON_NUM_THREADS=3 vs the default 14), zero-shot, all 280 rows"
  found: "all.json byte-identical (cmp rc=0)"
  implication: "The Rust re-score does not depend on the host's core count; a pass/fail on this box is not a scheduling accident."

- timestamp: 2026-09-26 (session 3, after the checkpoint)
  checked: "User answers to the human-verify + decision checkpoint"
  found: "(1) YES, accept the rope_inv_freq fix and commit only layer.rs. (2) Option A: keep pack_rescore_probs_abs = 1e-5; fixed_epochs is recorded as a RescoreDrift refusal (exit 2, still fail-closed) and early_stopping stays the vector that demonstrates GateFailed[ece_post]. No contract or tolerance change. The orchestrator re-ran `just laya-pack` on early_stopping: exit 3 GateFailed[ece_post], rescore 6.7e-6, zs 6.8e-6."
  implication: "The code half of the root-cause set is closed. The contract half (cause 1) stays open by decision, and moves to the queued calibration spike as a fragility question."

- timestamp: 2026-09-26 (session 3)
  checked: "Pre-commit confirmation on the working tree (debug profile): `cargo test -p aprender-core --lib models::modernbert` and `cargo test -p aprender-decide --lib`; `rustfmt --check` on layer.rs; staged set"
  found: "aprender-core modernbert: 18 passed, 0 failed (incl. rope_inv_freq_is_torch_bitwise, tiny_parity). aprender-decide lib: 68 passed, 0 failed (incl. laya::tests::tiny_parity, artifact::determinism::golden_sha). rustfmt clean. Staged set = exactly crates/aprender-core/src/models/modernbert/layer.rs. Committed with normal hooks as 8e55e0bed on gsd/phase-2-contract-gate."
  implication: "The fix is committed; nothing else in the shared tree was staged."

- timestamp: 2026-09-26 (session 2)
  checked: ".planning/spikes/025-laya-rust-forward-parity/RUN-OUTPUT.md, the measurement the 1e-5 / 1e-4 bars were fitted from"
  found: "14 rows, base model only: worst |dz| 2.61e-5, worst |dp| 3.80e-6. The bars are 2.6x (probs) and 3.8x (logits) over the worst of 14 draws."
  implication: "Extreme-value growth alone takes the max of 280 heavy-tailed draws past a 2.6x headroom (the base measured 1.03e-5 at HEAD, 6.8e-6 with the fix, 4.9e-6 at c076 — all the same distribution). A fine-tuned checkpoint at T = 5 with logits to 22 has a noise floor 5-10x the base's. The bar was never sampled on a fine-tuned model."

## Eliminated

- hypothesis: "Tokenization / builder / Laya row-prefix caching produces different ids than torch"
  evidence: "Rust builder ids+markers == torch _encode_state ids+markers on 280/280 eval rows, at HEAD and at c076fb0f5"
  timestamp: 2026-09-26

- hypothesis: "A specific Rust op (GEMM, attention/softmax, LayerNorm, GELU, F16 widening, head, scorer) is less accurate than torch's"
  evidence: "Per-layer local error with f64-rounded inputs: Rust == torch fp32 at every sub-op of all 28 layers; mean logit error vs the f64 forward equal on zs (8.19e-6 vs 8.22e-6) and es (1.36e-5 vs 1.37e-5)"
  timestamp: 2026-09-26

- hypothesis: "The trueno::blis::transpose swap in 71e2306e5 changed values"
  evidence: "c076fb0f5 + HEAD's gemm.rs is bitwise identical to c076fb0f5 on all 280 zero-shot rows"
  timestamp: 2026-09-26

- hypothesis: "Cargo feature unification / dependency drift between c076 and HEAD changed kernels"
  evidence: "cargo tree -f '{p}|{f}' identical except the new unicode-normalization dep; no version changes in Cargo.lock"
  timestamp: 2026-09-26

- hypothesis: "Rust or torch scoring is nondeterministic"
  evidence: "torch re-run == stored files bitwise on 280/280 (all three); zz_h0_dump cross-process runs cmp-identical at both c076 and HEAD"
  timestamp: 2026-09-26

## Resolution

root_cause: |
  AND-gate, three contributing causes:
  (1) CONTRACT / NOISE FLOOR. pack_rescore_probs_abs = 1e-5 (and logits_abs = 1e-4) is at or below the fp32
      rounding-noise floor of real Laya checkpoints. The float64 forward of the same model shows torch's OWN fp32
      probabilities are up to 3.71e-5 from exact on fixed_epochs (13 rows > 1e-5; logits 52 rows > 1e-4), and up to
      7.9e-6 / 9.5e-6 on the base / early_stopping. Rust's per-op accuracy equals torch's at every sub-op of every
      layer, so Rust-vs-torch is two independent draws of that noise. The residual stream carries activations up to
      2.2e4 and fixed_epochs runs at T = 5 with logits to 22, which amplifies the tail. The bars were fitted from 14
      spike rows on the base model.
  (2) CODE (fixed). rope_inv_freq single-rounded 1/theta^(2p/hd) from f64; transformers rounds theta^e to f32 and then
      takes an f32 reciprocal. One-ULP differences at 8/32 (theta 160000) and 10/32 (theta 10000) frequencies put
      up to ~32 ULP into the sin/cos tables. The code's own doc called this "torch order".
  (3) CODE (mechanical, not a defect). 71e2306e5 hoisted the RoPE table into RopeTable::new; LLVM then stopped fusing
      sin+cos into Apple's __sincosf_stret (c076 imports it, HEAD imports sinf/cosf), a 1-ULP change on ~0.6% of q/k
      elements. That alone moved zero-shot row 49 from 4.9e-6 (pass) to 1.028e-5 (fail).
fix: |
  crates/aprender-core/src/models/modernbert/layer.rs: rope_inv_freq computes f32(1 / f32(theta^e)) exactly as
  transformers 5.17 does (reproduces its inv_freq bits 32/32 for both thetas); doc comments corrected; regression
  test rope_inv_freq_is_torch_bitwise pins the torch bit patterns for hd 64 and 16. Cause (1) is NOT fixed. It
  cannot be fixed in Rust without widening the bar or re-deriving the reference, both user decisions.
  Committed: 8e55e0bed "fix(08): compute ModernBERT RoPE inv_freq with torch's f32 rounding" on
  gsd/phase-2-contract-gate (layer.rs only, normal hooks).
user_decisions:
  fix_accepted: "YES (2026-09-26): accept the rope_inv_freq fix."
  fixed_epochs_outcome: |
    OPTION A (2026-09-26): keep the 1e-5 bar. No contract or tolerance changes (D-17 and laya-parity-v1 as
    written). The fixed_epochs vector is recorded as a RescoreDrift refusal: `REFUSED RescoreDrift which=fine_tuned
    row=59 max_abs=4.667e-5`, exit 2. It is still fail-closed (nothing is written). early_stopping remains the
    vector that demonstrates GateFailed[ece_post] (exit 3, rescore 6.7e-6, zs 6.8e-6; re-confirmed by the
    orchestrator). Plan 08-09's executor applies this outcome to its plan/SUMMARY; this session only records it.
fragility_caveat: |
  The 1e-5 bar sits at the fp32 noise floor of real Laya checkpoints. early_stopping and zero-shot now pass
  with about 1.5x headroom (6.7e-6 / 6.8e-6 against 1e-5), and this session showed that a single 1-ULP codegen
  change elsewhere (the sin+cos -> __sincosf_stret fusion lost in 71e2306e5) consumes that headroom. So any
  future toolchain, libm, LLVM or platform change (x86_64 Lambda was never measured) can flip early_stopping
  back to RescoreDrift without a real defect. The principled basis for a bar is a float64-reference
  measurement: bound |torch32 - f64| and |rust32 - f64| separately rather than |rust32 - torch32|. The
  orchestrator is adding this question to the queued calibration spike todo
  (.planning/todos/pending/spike-laya-calibration-slice-and-temperature-cap.md). It is not resolved here.
prevention:
  five_whys:
    code_branch: |
      Why did Rust's RoPE tables differ from torch's? rope_inv_freq single-rounded from f64. -> Why was that
      possible? The doc comment said "torch order" but described the mathematical formula, not torch's f32
      evaluation order, and nobody compared bit patterns. -> Why was it not visible? The tiny fixture (1e-7
      parity) uses short rows where the pos x 1-ULP angle error stays below the fixture's tolerance, and the
      /simplify refactor's bit-identity proof compared Rust against Rust, not Rust against torch.
    contract_branch: |
      Why did the pack refuse? Rust-vs-torch crossed 1e-5. -> Why could honest fp32 do that? The bar was fitted
      from the worst of 14 spike rows on the base model (2.6x headroom), which under-samples a heavy-tailed
      max over 280 rows and never sampled a fine-tuned, high-temperature checkpoint. -> Why was that not caught
      before 08-09? No earlier gate re-scored all rows of a real fine-tuned run dir, and no gate measured either
      side against a float64 reference.
  why_not_caught: |
    Two gates missed it. (a) The tiny-fixture parity tests (aprender-core modernbert tiny_parity, aprender-decide
    laya tiny_parity) ran at a tolerance the RoPE ULP error cannot reach on short rows, and no test pinned
    inv_freq bits. (b) For the contract half, no gate existed: the bar came from a 14-row spike on the base model,
    and 08-09 was the first full 280-row re-score of a real fine-tuned checkpoint.
  recurrence_guard: |
    crates/aprender-core/src/models/modernbert/layer.rs::tests::rope_inv_freq_is_torch_bitwise pins transformers
    5.17 / torch 2.14 inv_freq bit patterns for (theta 160000 | 10000) x (hd 64 | 16). It is RED on the old
    single-rounding form at p = [6, 14, 23, 25, 26, 27, 28, 31], and cargo-mutants caught 12/12. It passes in
    commit 8e55e0bed. For the bar, the guard is the fragility question on the calibration spike todo and this
    KB entry.
mempalace_index: "skipped: no MemPalace tool is available in this environment; .planning/debug/knowledge-base.md is the durable record"
oracle_type: "specified (bit patterns produced by the reference implementation, transformers 5.17 / torch 2.14)"
files_changed:
  - crates/aprender-core/src/models/modernbert/layer.rs
verification:
  target_test: { result: pass, test: "crates/aprender-core/src/models/modernbert/layer.rs::tests::rope_inv_freq_is_torch_bitwise", red_on_old_code: "theta 160000 hd 64: inv_freq differs from torch at p = [6, 14, 23, 25, 26, 27, 28, 31]" }
  real_repro: { result: pass, es_pack: "REFUSED GateFailed clauses=[ece_post] rescore 6.735e-6 zs 6.825e-6 argmax 280/280, exit 3", fe_pack: "REFUSED RescoreDrift which=fine_tuned row=59 max_abs 4.667e-5, exit 2 (expected: cause (1) is not fixable in Rust under this bar)" }
  mutation_check: { result: pass, mutant_killed: true, detail: "cargo-mutants 25.3.1, private worktree, -f layer.rs --re rope_inv_freq -- --lib -- modernbert:: : 12 mutants tested, 12 caught, 0 missed/unviable/timeout. Plus the manual mutant 'revert to the f64 single-rounding form': killed (RED at p=[6,14,23,25,26,27,28,31]). Known equivalent (not generated): dropping the `theta as f32` narrowing is invisible for thetas exactly representable in f32 (160000, 10000)." }
  no_op_deletion: { result: pass, deletion_justified_by_rca: n/a, note: "the diff replaces one expression with the torch-order computation and adds a test; nothing deleted or short-circuited" }
  adjacent_tests: { result: pass, suites_run: ["cargo test --release -p aprender-core --lib modernbert (18 passed, tiny_parity + window_mutation green)", "cargo test --release -p aprender-decide (68 lib incl. laya::tests::tiny_parity, artifact::determinism::golden_sha 37d65159..., probe ladder; ui 1)", "cargo test --release -p aprender-mcp-decide -p aprender-mcp-decide-lambda (25 + 2 + 27)", "debug profile (what CI runs): cargo test -p aprender-core --lib modernbert 18 passed; cargo test -p aprender-decide --lib 68 passed", "cargo test --release -p aprender-core --lib: 14285 passed, 2 failed — calibration::tests::{brier_multiclass,top_label_ece}_refuses_empty_input, PRE-EXISTING and release-only (they expect the #[contract] macro message 'precondition violated', which is a debug_assert!; calibration.rs is untouched)"], clippy: "cargo clippy -p aprender-core --lib --tests --no-deps: no finding in layer.rs (the -D warnings run fails on the PRE-EXISTING unreachable-expression warning in src/demo/reliable/performance.rs, untouched)", fmt: "rustfmt --check clean", laya_fixtures: "skipped: just laya-fixtures is Python-only (scripts/laya_train/fixtures.py imports no Rust); the change is outside its import graph, and running it rewrites committed fixture files in the shared tree" }
  revert_and_reconfirm: { result: pass, bug_returned_on_revert: true, fixed_on_reapply: true, detail: "real ES pack: reverted -> RescoreDrift zero_shot row 49 1.0281801223754883e-5 (the original symptom, bit for bit); reapplied -> GateFailed[ece_post]" }
  thread_invariance: "RAYON_NUM_THREADS=3 vs 14: re-score byte-identical"
  pre_commit_recheck: "2026-09-26 on the working tree (debug profile): cargo test -p aprender-core --lib models::modernbert 18 passed / 0 failed; cargo test -p aprender-decide --lib 68 passed / 0 failed (golden_sha 37d65159... and laya tiny_parity green); rustfmt --check clean"
  human_verify: "user confirmed 2026-09-26 (fix accepted, option A); orchestrator re-ran `just laya-pack` on early_stopping: exit 3 GateFailed[ece_post], rescore 6.7e-6, zs 6.8e-6"
  guardrail_verdict: accepted  # for cause (2) only; cause (1) is a contract decision, not a code fix
