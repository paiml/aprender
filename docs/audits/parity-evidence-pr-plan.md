# PMAT-3303 / PMAT-3091 — how the Qwen3.5 CPU parity evidence should land

**Status:** plan only. Nothing here has been pushed and no PR has been opened.
Written 2026-09-16 on `PMAT-3091-layerwise` after the whole stack was rebased onto the
current `origin/PMAT-3329-llama-pin-bump` (`601fd72c2`, PR #3331).

## The result being landed

apr's Qwen3.5 CPU forward (#3114), with ggml's own arithmetic emulated behind
`APR_EMULATE_GGML_VECDOT=scalar`, is **bit-identical** to a scalar-built llama.cpp
`d1d3c3396` in config C (`-ctk f32 -ctv f32 -fa off`): 1500/1500 (p4 pos 0–3) and
750/750 (orig pos 4, 28) dumped points bit-equal, `max_rel_l2 = 0.0`, the per-token
logit streams byte-identical (`cmp` rc 0), 0 argmax mismatches over 78 + 82 positions,
and **no algorithmic departure at any of the six iterations** — every departure closed
was float approximation or float evaluation order. The **shipped** apr path is untouched:
switch-OFF logits are byte-identical to the recorded subjects at every iteration
(`f6f79264…c0252` orig, `92f1b54d…d3ee9` p4).

The emulation and the observer live on `PMAT-3091-layer-observer` (worktree `obs-3091`,
`b59433f7d`), a **local measurement branch that is never PR'd**. The evidence cites it by
commit sha and by binary sha256. Nothing in the PR series below changes any shipped code —
all six branches are `evidence/` + `scripts/llama_pin.toml` only.

## The stack (rebased, linear, `origin/PMAT-3329-llama-pin-bump` = `601fd72c2`)

| # | commit | branch head |
|---|--------|-------------|
| 1 | `1d8c6ca5a` REFERENCE.md | `PMAT-3303-qwen35-cpu-reference` |
| 2 | `0f83ce476` REFERENCE-RAW.md + producer | `PMAT-3303-qwen35-raw-logits` |
| 3 | `dbe6b7d88`, `348972868` MEASUREMENT.md + comparator | `PMAT-3091-apr-vs-reference` |
| 4 | `c88c9de3e`, `7965af7df` PER-TOKEN.md + `--per-token` | `PMAT-3303-per-token-prefill` |
| 5 | `7c2baa200` VARIATION.md | `PMAT-3091-flip-variation` |
| 6 | 13 commits `c03d15614 … 011591966` + this pass | `PMAT-3091-layerwise` |

## Proposed series: 3 PRs, not 6

Six branches are six chapters of one measurement and their documents cross-cite; one PR
would be 11.8 MB and mix a *reference* with a *claim about apr*. The cut points below are
branch heads in the rebased stack, so each PR is a suffix of the one before it.

### PR 1 — the reference (branches 1–2, head `0f83ce476`)
- **Title:** `evidence(parity): a reproducible llama.cpp d1d3c3396 CPU raw-logit reference for Qwen3.5-0.8B (#3303)`
- **Base:** `PMAT-3329-llama-pin-bump` (#3331) — or `main` once #3331 lands.
- **Closes** #3303's reference deliverable. **Refs** #3091.
- **Size:** 9 files, 46,682 bytes. No binaries.
- **Rationale:** it makes *no* claim about apr. It is the artifact everything else is
  measured against, it is falsifiable on its own (n=5 byte-identical; the producer AGREES
  with `llama-perplexity` on 0/9.4M words), and it can land even if #3114 slips.
- **Body draft:**
  > A llama.cpp `d1d3c3396` CPU reference for Qwen3.5-0.8B Q4_K_M on intel: 5 runs,
  > 1 distinct sha256 (byte-identical), 38 saved positions per 78-token context
  > (`REFERENCE.md`), then a raw float32 logit producer covering **all 78** positions
  > with a falsifier against `llama-perplexity` (`REFERENCE-RAW.md`, 0 of 9.4M words
  > differ). Producer source and transcripts are committed; the 77.5 MB logit blob is
  > not, and is re-derivable from the recorded command + sha256.
  > **Sets no threshold.** The `qwen3.5-0.8b` row of `evidence/parity/thresholds.yaml`
  > stays `[U]` and fail-closed.

### PR 2 — apr against that reference (branches 3–5, head `7c2baa200`)
- **Title:** `evidence(parity): apr's Qwen3.5 CPU forward vs the d1d3c3396 raw-logit reference — 78 positions, per-token prefill, 5 prompts (#3091)`
- **Base:** PR 1.
- **Refs** #3091, #3114, #3303. Closes nothing.
- **Size:** +54 files, +578,478 bytes over PR 1. No binaries.
- **Rationale:** numbers, no verdict — min cosine 0.995905, 0 positions below 0.98, 4 argmax
  mismatches all at rank 2; then the two controls that stop a cause being named from one
  input: llama's own batched-vs-per-token prefill (flips 2 and 36 are llama near-ties, not
  apr) and 5 prompts × 468 positions with apr threads varied 1/8/32 (byte-identical).
- **Body draft:**
  > Measures #3114's CPU forward against PR 1's reference on the same host.
  > `MEASUREMENT.md`: 78 positions, min cosine 0.995905 @pos4, 0 below 0.98, 4 argmax
  > mismatches (all rank 2). `PER-TOKEN.md`: run llama one token at a time — flips 2 and
  > 36 vanish, so they were llama batch-vs-per-token near-ties; 28 and 73 persist.
  > `VARIATION.md`: 5 prompts, 468 positions, 12 persistent flips, all below llama's own
  > median gap; apr thread count 1/8/32 is byte-identical, so the flips are not a thread
  > confound. **No threshold is set and no verdict is drawn.**

### PR 3 — the landmark (branch 6)
- **Title:** `evidence(parity): with ggml's arithmetic emulated, apr's Qwen3.5 CPU forward is BIT-IDENTICAL to scalar llama.cpp d1d3c3396 (#3091)`
- **Base:** PR 2.
- **Refs** #3091, #3114, #3038. Closes nothing — it is a measurement, not a gate.
- **Size:** +392 files, +10.5 MB over PR 2 (see *Size note*).
- **Rationale:** this is the answer to "is there any algorithmic difference?" — no — and it
  deserves its own review thread, because it is the only PR whose reviewer has to judge a
  *bit-identity* claim and a 10 MB evidence tree.
- **Body draft:**
  > Walks the two engines layer by layer (`LAYERWISE.md`), emulates ggml's CPU
  > quantization + vec_dot bit-exactly (`EMULATION.md`, 10 C fixtures), removes the KV-cache
  > confound (`KVCONFIG.md`: f32 KV with FA **on** is byte-identical to the f16 default,
  > because the graph casts K/V to F16 — only FA off is exact attention), and then, against a
  > SCALAR-built llama, closes the remaining departures one at a time (`SCALAR.md`, 6
  > iterations, 84/84 SSE2 + 10/10 quant fixtures bit-exact).
  > **Result: 1500/1500 + 750/750 dumped points bit-equal, `max_rel_l2 = 0.0`, logit streams
  > byte-identical, 0 argmax mismatches, and not one algorithmic difference.** Every closed
  > departure was float approximation (ggml's SSE2 `expf`) or evaluation order (RMSNorm row
  > order, double-accumulated dots, ggml delta-rule order).
  > The emulation is behind `APR_EMULATE_GGML_VECDOT` and lives on a local measurement branch
  > that is **not** in this PR; switch-OFF output is byte-identical to the shipped path at
  > every iteration. **Still no threshold, and the parity row stays `[U]`.**

### Ordering and CI
`ci.yml` fires only on PRs targeting `main`, so PRs 2 and 3 are CI-dark while stacked.
Land #3331 first, then PR 1 → 2 → 3, retargeting each to `main` as its parent merges; or
merge the three in one train. Do not squash the stack into one commit — the per-iteration
commits are the audit trail of the SCALAR result.

### Size note (nothing has been deleted)
PR 3 carries 11.1 MB of committed evidence, all text, largest file 250,744 bytes:
- `scalar/iter/it{1..6}*/walk_{p4,orig}.tsv` — 8 iteration directories, ~435 KB each,
  ≈3.5 MB. They are near-duplicates by construction (the same 2250 rows re-measured), and
  they are what proves *each* iteration moved the first departure. A reviewer may ask for
  only `it1` + `it6-final`; that is a review decision, not one this pass took.
- `emulation/fixtures.bin` (93,250 bytes) is the **only** non-text file in the six branches:
  raw fixture rows and quantized blocks for the 10 ggml C fixtures, cited by `EMULATION.md`.
- No file exceeds 1 MB anywhere in the stack.

## Still [U] after this campaign (each with its command)

Run from the repo root; evidence paths are relative to
`evidence/parity/l0-1/intel/qwen35-apr-vs-reference/`.

1. **Sub-layer dumps for p1–p3.** The bit-identity walk covers p4 (pos 0–3) and orig
   (pos 4, 28) only; p1–p3 were compared at the logits only.
   `bash scalar/run_scalar_ref.sh` §2 with p1–p3 positions (llama side), then
   `qwen35_layer_obs dump` on the apr side and `python3 scalar/walk_points.py`.
2. **The native / SIMD llama build.** Bit-identity is proved against a llama built
   `GGML_NATIVE=OFF` with every ISA off and repack off. The native intel build takes the
   AVX2 `gemv_q4_K_8x8_q8_K` repack path for 98 Q4_K tensors, which is not ported. Either
   `cmake -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=OFF -DGGML_AVX2=OFF -DGGML_AVX=OFF` and rerun
   `bash kvconfig/run_kvconfig.sh` config C, or port `gemv_q4_K_8x8_q8_K` and the AVX2 float
   order behind `APR_EMULATE_GGML_VECDOT` and rerun `scalar/run_apr_scalar.sh`.
3. **`Qcur_normed` / `Kcur_normed` have no llama counterpart.** They are the per-head q/k
   RMSNorm outputs; llama computes them but leaves the nodes unnamed (`node_30` / `node_33`,
   `RMS_NORM f32 128x16`), so they are apr-side diagnostic points and entered no comparison —
   the walk iterates llama's manifest, and apr's p4 manifest has exactly 48 extra rows
   (2 points × 6 full-attention layers × 4 positions): 1548 vs 1500.
   `awk -F'\t' '$1 ~ /^[QK]cur_normed-/' layerwise/llama_tensor_names.tsv` shows what llama
   *does* name at that point (`MUL f32 256x8`) — a different tensor. To close it, name the
   nodes in llama's graph (`llm_graph_context::build_attn` q/k norm) and re-dump.
4. **The Qwen3.5 parity row — still `[U]`, and this campaign sets no threshold.**
   `min_cosine` is deliberately absent from `models.qwen3.5-0.8b` in
   `evidence/parity/thresholds.yaml`, so `scripts/check_model_parity.sh --judge` exits 2
   rather than inheriting the GPU 0.98 (whose basis is a GPU-vs-CPU population on two dense
   qwen2.5 models — one implementation, not two).
   `python3 -c "import yaml;print('min_cosine' in yaml.safe_load(open('evidence/parity/thresholds.yaml'))['models']['qwen3.5-0.8b'])"` → `False`.
   Four things are owed before a number, per that row's own `basis:` — and none of them is a
   number. **Note for a follow-up ticket:** that `basis:` still says the comparator pin
   `39173bcac` cannot load qwen35. #3331 re-pins to `d1d3c3396`, which can. The row's text is
   stale in that one respect; correcting it belongs with whoever owns the row, not with this
   evidence series.
5. **Everything each document already lists as `[U]`** — in particular `LAYERWISE.md`'s
   Q8_K-mechanism proof and the F32-`ssm_out` GGUF re-encode, and `KVCONFIG.md`'s p1–p4 `-v`
   engagement lines.
