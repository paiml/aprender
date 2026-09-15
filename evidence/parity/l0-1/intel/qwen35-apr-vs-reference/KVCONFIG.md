# Qwen3.5-0.8B Q4_K_M: apr against a llama.cpp reference with exact attention arithmetic (PMAT-3091)

**Question.** How much of the apr-vs-llama.cpp logits gap comes from the reference's attention/KV configuration (f16 KV cache, Flash Attention on)? The test: rebuild the reference with f32 K/V and Flash Attention off, then re-measure apr against it with the ggml vec_dot emulation OFF and ON.
**No threshold is set anywhere. The Qwen3.5 parity row stays [U], fail-closed.**

**Result in brief:**
- **f32 KV with FA on (config B) is not an exact-arithmetic reference at d1d3c3396.** It is byte-identical to the default (A) on all 5 prompts, even though llama logs `K (f32), V (f32)` and `flash_attn = enabled`. The source explains it: `src/llama-graph.cpp:2624-2630` casts an F32 `k`/`v` to F16 right before `ggml_flash_attn_ext`. The CPU FA op then quantizes Q to F16 (`ggml/src/ggml-cpu/ops.cpp:8689-8690`) and dots in F16.
- **FA off with f32 KV (config C) changes the reference on all 5 prompts** (`cmp` rc 1 against A and B). It is deterministic: C n=2 on orig is byte-identical.
- **The attention/KV configuration accounts for only a few percent of the Frobenius logits gap.** Share of the OFF-vs-A gap that is removed:

  | prompt | OFF vs C | ON vs A | ON vs C |
  |---|---|---|---|
  | orig | 2.2% | 15.7% | 15.1% |
  | p1 | 2.3% | 0.8% | 3.6% |
  | p2 | −1.0% | 4.3% | 6.2% |
  | p3 | −0.9% | 10.2% | 8.6% |
  | p4 | 5.1% | 32.3% | 34.8% |

  Most of the gap remains under C on every prompt, with emulation ON or OFF.
- **At the positions where EMULATION.md found the layer-3 residual, C removes it.** At p4 pos 0 and pos 1, apr ON against C stays at rel L2 ≤ 0.000002 on every `l_out` from layer 0 through 11, including full-attention layers 3, 7 and 11. Against A, `l_out-3` was 0.003007 (pos 0) and 0.010622 (pos 1).
- **Under C the residual enters in DeltaNet layers.** Where it enters depends on the position: layers 0–2 at p4 pos 2–3 and orig pos 4/28, which precede any attention layer and match the A curve to 6 digits; layer 12 at p4 pos 1 and layer 13 at p4 pos 0. The largest step at p4 pos 1 is still `l_out-21` (layer 21 DeltaNet FFN), 0.078382 → 0.088171, step **+0.009789** (under A: +0.012802).

## Tree

| side | tree |
|---|---|
| evidence | worktree `layer-3091`, branch `PMAT-3091-layerwise`, on `4017a5187` (EMULATION.md) plus this commit |
| apr code | `obs-3091` @ **`2ae1971606724c1988722d6cc0146b02dff0e352`** (read-only in this phase), stacked on #3114 **`31448f6c3`**. Binary `qwen35_layer_obs` sha256 `e56bacdb0ba9fc9a5c32213671dccf130a64fb3bf2fae6f45b68bb6577c167e3`, the same file EMULATION.md ran; it was not rebuilt |
| reference | llama.cpp **`d1d3c3396`**, intel `~/src/llama.cpp-d1d3c3396`. Producer source `qwen35-cpu-reference/producer/apr_raw_logits.cpp` sha256 `5038776ebd1f1f75870a3290c3405c1f80ef464ebc2026dd0413ed2652e9c741`. The binary went from `5c5df52d…dd06` to **`a71dde33b859f3a21c7462e3a3f56f637706d79831edc1426a78f0b1f4642611`**, and the rebuild reproduced the same sha (`intel/rebuild.transcript`) |
| model | `Qwen3.5-0.8B-Q4_K_M.gguf` sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` (intel and lambda) |

## Hosts and load

- **intel** (`mac-server`, 32 threads, glibc 2.35), `-t 8` and `RAYON_NUM_THREADS=8`, one job at a time, `timeout` and `</dev/null` on every binary. Writes went only to `~/parity-ref/kvconfig-3091/`, `/tmp/kv3091/` and the producer build dir.
  - All runs 22:08:29Z–22:12:22Z. Load 57–86, from other tenants (`intel/run_kvconfig.transcript` logs load per run). Every output is compared by bytes or by hashed inputs, so load does not change a result.
  - Rebuild 22:12:26Z. Free disk 386G before the dumps and 382G after.
- **lambda** (48 threads): comparators 22:12:43Z–22:13:53Z, load 0.55 → 9.74 (`compare_kvconfig.transcript`).

## 1. Producer flags

These are parsed and removed before `common_params_parse`, so common's own `-fa/--flash-attn` never sees them. They set common params, and `common_context_params_to_llama` copies those onto `llama_context_params`. The field names and enums were read from the d1d3c3396 headers on intel:

| flag | common_params field | llama_context_params field (`common.cpp`) | enum / type (`llama.h`) |
|---|---|---|---|
| `--kv-type f16\|f32` | `cache_type_k`, `cache_type_v` | `type_k`, `type_v` (`:1751-1752`) | `ggml_type` (`llama.h:389-390`) |
| `--flash-attn on\|off\|auto` | `flash_attn_type` | `flash_attn_type` (`:1742`) | `llama_flash_attn_type` AUTO=−1, DISABLED=0, ENABLED=1 (`llama.h:190-193`) |

When neither flag is given, the params are left untouched.

**No-flag regression (rebuilt producer).** Per-token logits are `cmp`-equal to the committed references on all 5 prompts, rc 0:

| prompt | sha256 |
|---|---|
| orig | `2401c110…c4bc` |
| p1 | `67dcc9e7…f10a` |
| p2 | `a0f87b14…b760` |
| p3 | `df47beff…6ea5` |
| p4 | `0eca077c…6e1e` |

The `-v` probe `A-v-orig` is also equal. The producer's logged token ids equal the committed `.ids` in every run.

**Engaged: llama's own context-creation log**, from `-v` runs (`engaged/*.excerpt.log`, each with its full-log sha256):

| config | `llama_context: flash_attn` | `llama_kv_cache: size = …` |
|---|---|---|
| A (no flags) | `= auto`, resolved `resolve_fused_ops: Flash Attention enabled` (orig) | `K (f16): 1.50 MiB, V (f16): 1.50 MiB` |
| B | `= enabled` | `6.00 MiB (256 cells, 6 layers, 1/1 seqs), K (f32): 3.00 MiB, V (f32): 3.00 MiB` |
| C | `= disabled` | `6.00 MiB (256 cells, 6 layers, 1/1 seqs), K (f32): 3.00 MiB, V (f32): 3.00 MiB` |

- Each line appears twice per log, and all 5 prompts and the C n=2 run show the same lines.
- The producer also prints `apr_raw_logits: kvconfig cache_type_k=f32 cache_type_v=f32 flash_attn_type=enabled|disabled`.
- Another engagement proof: the C dump runs record 1350 distinct graph node names (`intel/C-tensor_names.tsv`); the default graph had 1314 (LAYERWISE.md).

## 2. Reference configs (per-token, n=1; C also n=2 on orig)

sha256 of the per-token logits (`intel/logits.sha256`):

| prompt | A (default) | B (f32, FA on) | C (f32, FA off) |
|---|---|---|---|
| orig | `2401c110…c4bc` | `2401c110…c4bc` (= A) | `77d587bc…ad30d`; n=2 `77d587bc…ad30d` (`cmp` rc 0) |
| p1 | `67dcc9e7…f10a` | = A | `2ed4d864…4d7e` |
| p2 | `a0f87b14…b760` | = A | `e0f6a584…190b` |
| p3 | `df47beff…6ea5` | = A | `d2c8be74…b506` |
| p4 | `0eca077c…6e1e` | = A | `0ef9cdd2…1ef2` |

**Config C sub-layer dumps.** Same regex and positions as LAYERWISE.md §5: p4 pos 0–3 gives 1500 rows, orig pos 4/28 gives 750 rows.
- Both dump runs' logits are `cmp`-equal to C, so the callback did not alter the computation.
- Manifest sha256: p4 `4c20cf34…ed46`, orig `c973e2d4…9909` (`intel/C-sub-*.manifest.sha256.tsv`).
- The comparator self-check passed at every position: callback `result_output` == row P of the C per-token file.
- All 2250 rows matched a name apr wrote; status U = 0.

## 3. apr subjects

- **OFF:** `/tmp/emul3091/runs/off-*.bin`, the byte-identical subjects (EMULATION.md §4).
- **ON:** `/tmp/emul3091/runs/on-*.bin`, the lambda runs (EMULATION.md §5).
- **ON on intel (closes an EMULATION.md unknown).** The same binary `e56bacdb…` ran on intel with `APR_EMULATE_GGML_VECDOT=1 RAYON_NUM_THREADS=8`:
  - orig `485341ae…9a53`, counters `[7644, 2808, 1326, 2808]`;
  - p4 `6ccada2b…1f54`, counters `[8036, 2952, 1394, 2952]`.

  Both are **`cmp`-equal to the lambda ON runs** (rc 0). ON output is the same on the two hosts (lambda 48 threads, intel 8 rayon threads).

## 4. Logits: apr against A, B and C (committed `compare_raw_logits.py`, `emulation/logits_gap.py`)

Frobenius = ‖sub − ref‖_F / ‖ref‖_F over all positions. Gap removed = 1 − Frobenius / Frobenius(OFF vs A). Every B row equals its A row, because the files are identical. `cmp/tables.md`, `cmp/tables.json`:

| prompt | apr | ref | min cos | mean cos | median cos | n<0.98 | argmax mismatches | Frobenius rel L2 | gap removed |
|---|---|---|---|---|---|---|---|---|---|
| orig | OFF | A/B | 0.994599 | 0.998995 | 0.999282 | 0 | 4 | 0.044677 | 0.0% |
| orig | OFF | C | 0.996167 | 0.999032 | 0.999256 | 0 | 4 | 0.043681 | 2.2% |
| orig | ON | A/B | 0.998157 | 0.999266 | 0.999319 | 0 | 3 | 0.037667 | 15.7% |
| orig | ON | C | 0.998536 | 0.999270 | 0.999319 | 0 | 3 | 0.037928 | 15.1% |
| p1 | OFF | A/B | 0.997448 | 0.999509 | 0.999580 | 0 | 4 | 0.031619 | 0.0% |
| p1 | OFF | C | 0.997383 | 0.999512 | 0.999611 | 0 | 6 | 0.030891 | 2.3% |
| p1 | ON | A/B | 0.998868 | 0.999528 | 0.999565 | 0 | 8 | 0.031376 | 0.8% |
| p1 | ON | C | 0.998796 | 0.999544 | 0.999575 | 0 | 6 | 0.030473 | 3.6% |
| p2 | OFF | A/B | 0.995370 | 0.998371 | 0.998489 | 0 | 2 | 0.056340 | 0.0% |
| p2 | OFF | C | 0.995651 | 0.998352 | 0.998444 | 0 | 3 | 0.056881 | −1.0% |
| p2 | ON | A/B | 0.993047 | 0.998519 | 0.998728 | 0 | 2 | 0.053926 | 4.3% |
| p2 | ON | C | 0.995296 | 0.998572 | 0.998670 | 0 | 3 | 0.052848 | 6.2% |
| p3 | OFF | A/B | 0.995687 | 0.999133 | 0.999353 | 0 | 5 | 0.040229 | 0.0% |
| p3 | OFF | C | 0.995828 | 0.999105 | 0.999342 | 0 | 4 | 0.040604 | −0.9% |
| p3 | ON | A/B | 0.997929 | 0.999302 | 0.999432 | 0 | 5 | 0.036117 | 10.2% |
| p3 | ON | C | 0.994753 | 0.999252 | 0.999441 | 0 | 4 | 0.036772 | 8.6% |
| p4 | OFF | A/B | 0.961309 | 0.994771 | 0.998415 | 4 | 6 | 0.092010 | 0.0% |
| p4 | OFF | C | 0.957286 | 0.995217 | 0.998602 | 2 | 8 | 0.087292 | 5.1% |
| p4 | ON | A/B | 0.984332 | 0.997562 | 0.998841 | 0 | 3 | 0.062323 | 32.3% |
| p4 | ON | C | 0.984187 | 0.997733 | 0.998850 | 0 | 4 | 0.060027 | 34.8% |

Under C, argmax mismatches move by −2 to +2 per prompt, in both directions.

## 5. Per-layer: apr against config C dumps (committed `compare_layerwise.py`, `layer_steps.py`)

Residual stream `l_out-0 … l_out-23` rel L2, then `result_norm`. Full-attention layers are 3, 7, 11, 15, 19 and 23.

**apr ON vs C:**

| (label, pos) | curve |
|---|---|
| p4 pos 0 | 0.000001 ×6, 0.000000, 0.000001 ×5, **l_out-12 0.000002, l_out-13 0.003695**, 0.001664, 0.029170, … l_out-23 0.070949, result_norm 0.093603 |
| p4 pos 1 | ≤0.000002 through l_out-11, **l_out-12 0.000553** (attn_residual-12 0.000006, linear_attn_out-12 0.000032, ffn_out-12 0.001413), l_out-13 0.008744, … l_out-23 0.099840, result_norm 0.101369 |
| p4 pos 2 | 0.000001, 0.000001, **l_out-2 0.000592**, l_out-3 0.016212, … result_norm 0.081166 |
| p4 pos 3 | **l_out-0 0.000235**, l_out-1 0.011553, l_out-2 0.026157, … result_norm 0.086210 |
| orig pos 4 | 0.000000, 0.000001, **l_out-2 0.014620**, l_out-3 0.024107, … result_norm 0.037804 |
| orig pos 28 | 0.000001, **l_out-1 0.006023**, l_out-2 0.027309, … result_norm 0.052631 |

**apr ON vs A (EMULATION.md dumps), at the same points:**
- p4 pos 0: `l_out-3` 0.003007
- p4 pos 1: `l_out-3` 0.010622
- the pre-layer-3 values at p4 pos 2–3 and orig pos 4/28 are **identical to 6 digits** (0.000592, 0.000235, 0.014620, 0.006023)
- result_norm: p4 pos 0 0.093794, pos 1 0.150906, pos 2 0.093856, pos 3 0.087119; orig pos 4 0.039121, pos 28 0.049288

**Largest step per (label, pos), apr ON vs C** (`cmp/layer_steps_on_vs_C.tsv`):

| (label, pos) | point | layer | rel L2 | step |
|---|---|---|---|---|
| p4 pos 0 | attn_residual-15 | full-attn mixer | 0.001664 → 0.030645 | +0.028981 |
| **p4 pos 1** | **l_out-21** | DeltaNet FFN | 0.078382 → 0.088171 | **+0.009789** |
| p4 pos 2 | attn_residual-23 | full-attn mixer | — | +0.009281 |
| p4 pos 3 | l_out-23 | full-attn FFN | — | +0.023630 |
| orig pos 4 | l_out-2 | DeltaNet FFN | — | +0.012193 |
| orig pos 28 | l_out-2 | DeltaNet FFN | — | +0.012575 |

**apr OFF vs C:** the largest step is `attn_residual-0` (layer 0 DeltaNet mixer) at 5 of 6 positions, +0.034 to +0.042. That is the same kernel departure as against A (LAYERWISE.md).

**First departure under C.**
- At p4 pos 1 (and pos 0) the curve holds ≤ 0.000002 from layer 0 through layer 11, three full-attention layers included. It first leaves that band at layer 12, a DeltaNet layer, with p4 pos 0 following at layer 13. tables.py's literal "first non-zero printed rel L2" is `linear_attn_out-0` = 0.000002, the same under A.
- At the other four positions it leaves within DeltaNet layers 0–2, before any attention layer, with the same values as under A.

## Verdict (limited to the data)

1. At d1d3c3396 on intel CPU, `f32 KV + FA on` computes exactly what the default computes, byte for byte on 5 prompts. F32 K/V are cast to F16 before the FA op (source above).
2. The default reference's FA/f16 attention **is** the layer-3 residual that EMULATION.md found at p4 pos 0–1. Under FA off + f32 KV, apr ON matches through layer 11 at those positions.
3. It is **not** most of the logits gap. Against C, OFF moves by −1.0 to +5.1% and ON removes 3.6–34.8% (against A: 0.8–32.3%). The gap that remains under C enters in gated DeltaNet layers, at a layer that depends on the position (0–2 at four of six positions, 12–13 at p4 pos 0–1), and grows through the stack. What departs inside those layers is not measured here.
4. apr ON is host-invariant between lambda and intel for orig and p4.

No parity claim is made. The Qwen3.5 parity row stays [U].

## Remaining unknowns

- **[U] Which kernel departs in the DeltaNet layers under C** (p4 pos 1 layer 12, pos 0 layer 13, pos 2 layer 2, pos 3 layer 0, orig pos 4 layer 2, pos 28 layer 1). Measure with the kernel isolation against the C dumps:
  `python3 layerwise/kernel_isolation.py ~/models/Qwen3.5-0.8B-Q4_K_M.gguf ~/src/llama.cpp-d1d3c3396/gguf-py layerwise/layer_types.tsv 0,1,2,12,13 "p4:/tmp/kv3091/ref/C-sub-p4:/tmp/emul3091/runs/on-dump-p4:0,1,2,3" "orig:/tmp/kv3091/ref/C-sub-orig:/tmp/emul3091/runs/on-dump-orig:4,28"`
- **[U] Whether the unported AVX2 dot accumulation / `gemv_q4_K_8x8` repack path explains those departures.** EMULATION.md measured ≤1.8e-5 relative on the Q4_K dot. Two ways to measure:
  - build a d1d3c3396 reference with `-DGGML_NATIVE=OFF -DGGML_CPU_REPACK=OFF` (plus `-DGGML_AVX2=OFF -DGGML_AVX=OFF`) and rerun `kvconfig/run_kvconfig.sh` config C against it; or
  - port `gemv_q4_K_8x8_q8_K` and the AVX2 float order behind `APR_EMULATE_GGML_VECDOT`.
- **Measured (orig): A's `flash_attn = auto` resolves to enabled.** `A-v-orig.log:991` and `:2101` read `resolve_fused_ops: Flash Attention enabled`. **[U] for p1–p4:** their default runs had no `-v`. Byte identity A == B on all 5 prompts (B forces FA on) is consistent with enabled, but it is not a log line. Measure by rerunning `run_kvconfig.sh` §0c with `-v` for p1–p4.
- **[U] C at positions not dumped.** Sub-layer data covers only p4 pos 0–3 and orig pos 4/28. Measure with `--dump-positions` on more positions in `run_kvconfig.sh` §2.
- **[U] B/C and ON at n>1 on prompts other than orig.** Only C orig is n=2 (plus EMULATION.md's ON n=2 on orig/p4). Rerun `run_kvconfig.sh`.
- **[U] Qwen3.5 parity row.** No threshold is set, so it stays fail-closed.

## Commands

```bash
# intel: producer source + build.sh + raw_logits_compare.cpp copied to ~/parity-ref/kvconfig-3091/producer/,
# the lambda binary e56bacdb to /tmp/kv3091/, ids to /tmp/kv3091/ids/
bash ~/parity-ref/kvconfig-3091/run_kvconfig.sh > run_kvconfig.transcript   # build, A regression, B, C, C n=2, C dumps, apr ON on intel
bash ~/parity-ref/kvconfig-3091/producer/build.sh                            # rebuild after copying raw_logits_compare.cpp (see note), rc 0, same a71dde33
# lambda: runs/ rsynced to /tmp/kv3091/ref/
bash kvconfig/compare_kvconfig.sh > compare_kvconfig.transcript              # ON intel-vs-lambda cmp, logits A/B/C x OFF/ON, sub-layer vs C, tables.py
bash scripts/check_llama_pin.sh --self-test                                  # gate, rc 0 (kvconfig/gate.log)
```

**Build note.** In the first run the build step printed `build rc=1`: `raw_logits_compare.cpp` had not been copied, so the SECOND compile in `build.sh` failed. `apr_raw_logits` had already been linked (mtime 22:08:47Z, and the new code's `kvconfig` log line is present in the B/C runs). After copying the file, the rebuild exited rc 0 and produced the same `apr_raw_logits` sha256 `a71dde33…2611` (`intel/rebuild.transcript`).
