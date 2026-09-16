# Qwen3.5-0.8B Q4_K_M: layer-wise apr vs llama.cpp on the prompt-4 chat-markup outlier (PMAT-3091)

Question: where in the forward pass does apr first diverge from llama.cpp on prompt 4, which is where `VARIATION.md` §5 item 5 puts the minimum cosine (0.961309 at pos 1)?
**No threshold is set anywhere. The Qwen3.5 parity row stays [U], fail-closed.**

**Result, in one line:**
- The EMBEDDING output is byte-identical at every measured position.
- **Layer observer pass (§5, local branch `PMAT-3091-layer-observer`):** every sub-layer point is now measured on both engines. Largest relative-L2 step at p4 pos 1: **layer 0 (gated DeltaNet), mixer** (`attn_residual-0`, 0.000000 → 0.039963). The contrast positions step at the same point, and it is their largest step too.
- At that point apr's `ssm_out` (Q5_K) output equals float64 `dequant(W)·input` to rel L2 0.000001. llama's is 0.034–0.042 away at layer 0 (0.017–0.063 over all 42 DeltaNet (layer, pos) rows checked, §5d). llama's CPU `vec_dot_type` for Q5_K is Q8_K.
- The p4-pos-1-specific excess over the contrast positions sits in the FFN sublayers of DeltaNet layers 9–21 (largest at `l_out-21`, §5c), not at layer 0.
- The first pass's `[U]` (no apr per-layer hidden state) is closed by the observer. The parity row stays [U].

## Tree (every measurement below)

| side | tree |
|---|---|
| evidence worktree (layer-3091), base of this commit | `1976315fe` (PMAT-3091-flip-variation) |
| apr subject: #3114 head `31448f6c3`, detached at `/tmp/layer3091/apr-31448`, examples copied in uncommitted | logits binaries as in `VARIATION.md` (`qwen35_raw_logits` `ad4397d5…3d8e`). New: `qwen35_embd_dump` sha256 `f442a70658538b9df66d541e825eac6aa07e7298132d9da860f6baf30340d1b7` |
| reference: llama.cpp `d1d3c3396` | producer `apr_raw_logits` rebuilt with `--dump-tensors`, sha256 `5c5df52d75b514723f706b93aede9939f58e1df178d2b631ad0ad0038994dd06` |
| model | `Qwen3.5-0.8B-Q4_K_M.gguf`, sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` (the same on intel and lambda) |
| **§5 apr subject** (layer observer) | worktree `obs-3091`, branch `PMAT-3091-layer-observer` @ `0543266eeaaaf05a86b94ee5474456a69d22be1b`, stacked on #3114 `31448f6c3`. It is NOT part of #3114's PR. Harness `layerwise/qwen35_layer_obs.rs` (sha256 `89b48ab71d1c1120c615af87b892b9f9ee7b4c097382fefae0c949845cbc78fa`) copied uncommitted into `crates/aprender-serve/examples/`. Release binary sha256 `035531c18c5cb89489caf0ef87e91b09de3f34c7f975e0f50ce2d590e3251186` |
| **§5 reference** | llama.cpp `d1d3c3396`, the same producer binary `5c5df52d…dd06` (not rebuilt), driven by `../qwen35-cpu-reference/producer/run_sublayer.sh` |

## Hosts and load

- llama dumps, the apr embedding dump, and `compare_layerwise.py` ran on intel (`mac-server`, Xeon W-3245), CPU only, `-t 8`, one job at a time, every binary under `timeout` with stdin `< /dev/null`.
  Load was 72.40/78.19/61.19 at the first run (20:06:25Z) and 69.47/77.00/61.24 at the end (20:06:50Z), per `layerwise/run_layerwise.transcript`.
  The comparator ran at 44.43/66.22/59.44.
- The apr example build and `derive_final_norm.py` ran on lambda. Load was 11.72 at the build, and 35.87 → 32.46 over the derivation (20:10:55Z–20:11:37Z).
  The derivation is pure float64 linear algebra over committed or hashed inputs, so the host does not enter the result.

## Layer types (from the GGUF, not assumed)

The GGUF has `qwen35.full_attention_interval = 4` and **no** per-layer type array.
Types were read from the tensors present in each block (`layerwise/layer_types.tsv`):
- full attention (has `attn_q.weight`, no `ssm_a`): **3, 7, 11, 15, 19, 23**
- gated-DeltaNet linear attention (has `ssm_a`): every other layer in 0–23

This matches llama.cpp's loader rule at d1d3c3396 (`src/models/qwen35.cpp`: `is_recr = (i+1) % full_attn_interval != 0`).

## 1. llama side: eval-callback dumps

`producer/apr_raw_logits.cpp` gained `--dump-tensors DIR --dump-positions LIST [--dump-regex RE]`.
- It installs `params.cb_eval` / `cb_eval_user_data`, the `llama_context_params` fields that `examples/eval-callback` at d1d3c3396 sets through `common_params`, with `warmup = false` as that example does.
- It requires `--per-token`, so each decode step is exactly one position.
- In the `ask` phase it records every node name. For matching tensors it copies float32 data with `ggml_backend_tensor_get`.

**Tensor names the callback sees (FIRST list, committed):** `layerwise/llama_tensor_names.tsv`, sha256 `2ac9b2be774a4d360f69dcf1ac92f0aaf1dd7c7bbcc1dc6574c29917f52f20d5`.
- 1314 distinct names, byte-identical between the p4 and original runs.
- Embedding output: `model.input_embed` (GET_ROWS).
- Per layer: `attn_norm-N`, `attn_residual-N`, `attn_post_norm-N`, `ffn_out-N`, `post_ffn-N`, `l_out-N`. Linear layers add `linear_attn_out`, `conv_output_silu`, `state_predelta`, `final_output`, and more. Attention layers add `Qcur_normed`, `attn_pregate`, `attn_gated`, `attn_output`, and more.
- Final: `h_nextn`, then `result_norm` (GET_ROWS), then `result_output` (MUL_MAT, 248320).

The dump set is `model.input_embed`, `l_out-0..23`, `result_norm` and `result_output`, so 27 tensors per position.
The manifests with sha256 of every `.f32` are `layerwise/llama_manifest_p4.sha256.tsv` (108 rows, `f84bce6b…37f`) and `layerwise/llama_manifest_p0.sha256.tsv` (54 rows, `14867c46…62b`).
The raw dumps are not committed: intel `~/parity-ref/layerwise-3091/dump-p{4,0}/` (4.3 M / 2.2 M). Every hash was re-verified after copying to lambda.

**Proof that the callback did not change the computation, and that position mapping is right** (`run_layerwise.transcript`, `curves_*.selfcheck.txt`):
- The producer was rebuilt, then run per-token WITHOUT dumps. p4 matches `p4-per-token.bin` (`0eca077c…6e1e`) under `cmp` with rc 0. The original prompt matches `per-token-run1.bin` (`2401c110…c4bc`) under `cmp` with rc 0.
- The dump runs' logits files are also `cmp` rc 0 against those same two references.
- At all 6 dumped positions, the callback's `result_output` equals row P of the per-token logits file, checked with exact `np.array_equal`.

```bash
# intel
bash producer/build.sh                 # rc 0
WORK=/tmp/layer3091 bash producer/run_layerwise.sh   # script rc 0; transcript layerwise/run_layerwise.transcript
#  inside: timeout 900 apr_raw_logits -m MODEL -f PROMPT -ngl 0 -t 8 -c N -b N --per-token \
#          [--dump-tensors DIR --dump-positions 0,1,2,3 | 4,28] --raw-out F < /dev/null
```

## 2. apr side: which facility reaches the Qwen3.5 CPU path

These were searched at `31448f6c3`:
- `crates/aprender-serve/src/inference_trace/` (the `save_tensor*.rs` files and `gpu_stage_dump.rs`) has no reference to qwen35.
- `run_qwen35_generate` takes no tracer, so `apr run --trace --trace-level payload` has no hook inside the Qwen3.5 layer loop.
- `forward_qwen35.rs` reads no env var.
- `pmat query "qwen35 forward layer hidden state dump" --include-source` returned `forward_single_qwen35`, which has no dump. It also returned `debug_cpu_layer_output` in `ffn_block.rs`, which the qwen35 path never calls.
- Visibility blocks a public-API layer walk: `Qwen35Model.layers` is `pub(crate)`, `Qwen35OwnedLayer` is `pub(crate)`, `forward_deltanet` and `forward_attention` are private, and `OwnedQuantizedModel::fused_matmul_into` is `pub(crate)`.

**Reachable through the public API:** the embedding table (`Qwen35Model::create_base_model` → `token_embedding()`, which is exactly the slice `forward_single_qwen35` copies into `hidden`) and the logits.
`layerwise/qwen35_embd_dump.rs` dumps the former. Its embedding rows' sha256 are in `layerwise/apr_embd_manifest.sha256.tsv` (dumps on intel `~/parity-ref/layerwise-3091/apr/embd-p{4,0}/`).

```bash
# lambda: build (examples copied uncommitted into the #3114 detached tree)
cp layerwise/qwen35_embd_dump.rs /tmp/layer3091/apr-31448/crates/aprender-serve/examples/
CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/apr-3114-meas nice -n 10 timeout 3000 \
  ~/.cargo/bin/cargo build --release -p aprender-serve --example qwen35_embd_dump < /dev/null      # rc 0
# intel
timeout 600 ./qwen35_embd_dump MODEL prompt-4.ids 0,1,2,3 embd-p4 < /dev/null                   # rc 0
timeout 600 ./qwen35_embd_dump MODEL prompt_token_ids.txt 4,28 embd-p0 < /dev/null              # rc 0
```

## 3. Comparison

`layerwise/compare_layerwise.py` (sha256 `837aa8f8961776f0d612393ce4b93cff52cea003a85db6b7f9641004c502c33a` when §3 was run; the **committed** file is the later, extended version `54ccaca6…7adb3` — see §7, whose output without the 10th arg is unchanged) gives, per (tensor, pos), cosine, max |apr−llama| and relative L2 ‖apr−llama‖/‖llama‖.
`l_out-N` and `result_norm` rows carry `U` for apr, plus llama's own L2 norm.
Full per-position curves:
- `layerwise/curves_p4.tsv` (`5c75e221…a029`)
- `layerwise/curves_orig.tsv` (`74bc5277…b4b39f`)

```bash
python3 compare_layerwise.py p4 dump-p4 p4-per-token.bin apr/embd-p4 p4-apr.bin 0,1,2,3 layer_types.tsv MODEL GGUF_PY     # rc 0
python3 compare_layerwise.py orig dump-p0 per-token-run1.bin apr/embd-p0 apr-intel-run1.bin 4,28 layer_types.tsv MODEL GGUF_PY  # rc 0
```

### 3a. Embedding output: identical, so it is not the discriminator

- At all 6 positions (p4 pos 0–3 with ids 27, 91, 316, 4747; orig pos 4 id 33075; orig pos 28 id 1452), apr's embedding row vs llama `model.input_embed` gives **cos 1.000000, max |diff| 0.000000, rel L2 0.000000**.
- A third implementation, gguf-py's `dequantize` of `token_embd.weight` rows, is also max |diff| 0 against both engines.
- `token_embd.weight` is **Q6_K**, shape [1024, 248320]. A GGUF tensor has one ggml type for all its rows, so the markup ids' rows cannot differ in block kind from common tokens' rows.
- Brief step 5's premise does not hold: **the divergence is NOT present at the embedding.**

### 3b. Final norm and the last layer's residual output, derived from the logits

`layerwise/derive_final_norm.py` (sha256 `66f01456…c28f7`) produced `layerwise/final_norm_lstsq.tsv` (`d224c2ca…278345`).
- The method: Qwen3.5-0.8B has no `output.weight`, and both engines use `token_embd` as lm_head.
- `h = argmin ‖W h − logits‖` with W = dequantized `token_embd` (248320 × 1024, QR) recovers each engine's final-norm vector.
- `h / output_norm.weight` is `l_out-23` up to a positive scale, so only cosine is meaningful for that row.
- **CONTROL:** fitting llama's own logits reproduces llama's dumped `result_norm` at cos ≥ 0.999889 (rel L2 ≤ 0.0149) and `l_out-23` direction at cos ≥ 0.999814. The fit's relative residual is < 5e-7 (prints 0.000000) for every fit.

| label / pos | input | logits cos (measured) | logits rel L2 | derived `result_norm` cos (CONTROL / apr) | derived `result_norm` rel L2 (CONTROL / apr) | derived `l_out-23` direction cos (CONTROL / apr) | embd cos |
|---|---|---|---|---|---|---|---|
| p4 0 | `<` 27 | 0.995365 | 0.097638 | 0.999889 / 0.989044 | 0.014919 / 0.147827 | 0.999889 / 0.994118 | 1.000000 |
| **p4 1** | `\|` 91 | **0.961309** | **0.278233** | 0.999975 / **0.944845** | 0.007120 / **0.331704** | 0.999966 / **0.944144** | 1.000000 |
| p4 2 | `im` 316 | 0.973410 | 0.266478 | 0.999963 / 0.970524 | 0.008566 / 0.243297 | 0.999888 / 0.968895 | 1.000000 |
| p4 3 | `_start` 4747 | 0.981265 | 0.214359 | 0.999932 / 0.986881 | 0.011698 / 0.165674 | 0.999814 / 0.986700 | 1.000000 |
| orig 4 | 33075 | 0.994599 | 0.110592 | 0.999924 / 0.996068 | 0.012312 / 0.088610 | 0.999910 / 0.996656 | 1.000000 |
| orig 28 | `ized` 1452 | 0.998569 | 0.053922 | 0.999910 / 0.997429 | 0.013454 / 0.071702 | 0.999887 / 0.997789 | 1.000000 |

The measured logits cosines reproduce `VARIATION.md` B exactly (0.961309, 0.973410, 0.981265 for p4; 0.994599 and 0.998569 for orig). This is an independent recomputation from the same files with a different script.

### 3c. Per-layer curve (brief step 4)

| tensor | measured? |
|---|---|
| `model.input_embed` | yes: identical at all 6 positions |
| `l_out-0` … `l_out-22` | **yes (§5)**: both engines, all 6 positions, in `layerwise/sublayer_{p4,orig}.tsv` |
| `l_out-23` | **yes (§5)**, measured directly. At p4 pos 1: cos 0.944144, rel L2 0.332800. This equals the lstsq-derived direction cos in 3b (0.944144) to 6 digits |
| `result_norm` | **yes (§5)**. At p4 pos 1: cos 0.944845, rel L2 0.331704. 3b derived 0.944845 / 0.331704 |
| `result_output` | yes (3b) |

~~Largest step in relative L2 at p4 pos 1: [U].~~ Measured in §5c.

## 4. Observation (not a verdict, no threshold)

1. The embedding lookup is exact for the markup ids (Q6_K dequant is identical across apr, llama.cpp and gguf-py). The prompt-4 excess is not a token-embedding or quant-block-kind effect.
2. The excess is already in the residual stream when it leaves the last layer.
   - At p4 pos 1, the derived `l_out-23` direction has cos 0.944144; the method's own control there is 0.999966.
   - The contrast positions are at 0.996656 and 0.997789.
   - The lm_head plus final RMSNorm do not create it: the derived `result_norm` and the measured logits track each other at every position (rel L2 0.33 vs 0.28 at p4 pos 1).
3. The ordering across positions is the same at `l_out-23`, `result_norm` and logits: p4 pos 1 < pos 2 < pos 3 < pos 0 < orig pos 4 < orig pos 28.
4. llama's own `l_out-23` L2 norm at p4 pos 1 (8.40) is lower than at p4 pos 0 (10.68) (`curves_p4.tsv`). This is context only.

## 5. Layer observer pass (local measurement branch, orchestrator decision §8)

### 5a. The observer and its name mapping

`Qwen35Model::forward_single_qwen35_observed(&self, token_id, cache, position, obs: &mut dyn FnMut(&str, usize, &[f32]))`:
- `forward_single_qwen35` delegates to it with a no-op closure.
- The sublayers became `forward_deltanet_observed` / `forward_attention_observed`. The old names remain as `#[cfg(test)]` no-op wrappers for the contract tests.
- The observer only reads. The one rewritten statement, `dt = softplus(..) * a`, became two statements with the same two IEEE operations, so that `a_softplus` can be observed.
- Global points pass `QWEN35_OBS_NO_LAYER`. Layer points emit, in graph order, `QWEN35_OBS_DELTANET_POINTS` (18 names) or `QWEN35_OBS_ATTENTION_POINTS` (10 names since PMAT-3091 iteration 6, which added `Qcur_normed` /
  `Kcur_normed` — the per-head q/k RMSNorm outputs, which llama computes but leaves UNNAMED as `node_30` /
  `node_33` in `layerwise/llama_tensor_names.tsv`, so they have no llama counterpart to compare against).

**Mapping, read from source at d1d3c3396 (not from memory): `layerwise/observer_mapping.tsv`** (29 rows: llama name, llama file:line, apr fn, apr file:line at `0543266ee`).
- Mixer output before the residual add: `linear_attn_out-N` (`qwen35.cpp:462`) for DeltaNet, `attn_output-N` (`qwen35.cpp:330`) for attention.
- Residual after the mixer: `attn_residual-N` (`:181`).
- FFN output: `ffn_out-N` (`:192`, first named at `:480`).
- Layer output: `l_out-N` (`:199`).
- **`post_ffn-N` (`:196`) is never a callback name.** `build_cvec` returns the same ADD tensor, which `:199` renames to `l_out-N`. 0 of 1314 names are `post_ffn` (`llama_tensor_names.tsv`).
- **In DeltaNet layers `attn_output-N` is NOT the mixer output.** It is the delta-rule output (`delta-net-base.cpp:552`, 128x16), before the gated norm (`final_output-N`, `:458`) and `ssm_out`. apr observes it at the matching point (`out_h`).
- DeltaNet internals: `conv_output_silu :395`, `q/k/v_conv_predelta :444-446`, `a_softplus :371`, `gate :374` (= softplus·ssm_a), `beta_sigmoid :363`, `z :238`, `state_predelta :389`, `new_state delta-net-base.cpp:553`.

Tests (TDD: RED first as a compile failure, 5 errors, all unresolved observer items), in `forward_qwen35_observer_tests.rs`:
- `observed_path_is_bit_identical_to_plain_forward`: logits bits, ssm and conv state equal at 6 positions × 2 dims.
- `observer_emits_llama_names_in_graph_order_with_exact_residuals`: the exact name/layer sequence, `attn_residual = in + mixer` and `l_out = attn_residual + ffn_out` bit-exact, and `result_output` == logits.
- `cargo test -p aprender-serve --lib qwen35` on lambda (isolated `CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/obs-3091`; `cargo` is a zsh function, called by absolute path): **22 passed, 0 failed** (20 before + 2). `layerwise/observer_gate.txt`.

### 5b. Invariance proof (before any comparison)

`layerwise/observer_invariance.transcript`: lambda, 2026-09-15T20:32:44Z, load 8.64 → 10.83. `qwen35_layer_obs noop` runs the observed path with a no-op closure.

| run | sha256 | vs subject | `cmp` rc |
|---|---|---|---|
| original 78 ids | `f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252` | `apr-intel-run1.bin` `f6f79264…0252` | 0 |
| prompt 4 (82 ids) | `92f1b54d504a53f758a56911c4779ce1900207bd89c78d58597e558ebc6d3ee9` | `p4-apr.bin` `92f1b54d…3ee9` (full hash read on intel) | 0 |

The dump runs (a recording closure) also produced byte-equal logits: `obs-p4.bin` 92f1b54d… and `obs-p0.bin` f6f79264…, `cmp` rc 0 (`layerwise/apr_obs_dump.transcript`). **Observing did not change the computation**, with a no-op closure or with a writing one.

### 5c. Dumps, comparison, curves

```bash
# intel (mac-server), 20:24:03Z-20:24:43Z, load 38.72 -> 45.74; producer NOT rebuilt (sha 5c5df52d... re-checked)
bash producer/run_sublayer.sh
#   no-dump regressions: p4 cmp rc 0 vs 0eca077c..., orig cmp rc 0 vs 2401c110...
#   sub-p4: --dump-positions 0,1,2,3, written=1500 errors=0; logits cmp rc 0
#   sub-p0: --dump-positions 4,28,   written=750  errors=0; logits cmp rc 0
#   375 tensors/pos = 18 DeltaNet x 18 + 6 attention x 8 + 3 global
#   manifests llama_sub_manifest_p4.sha256.tsv (ac5090c6...) / _p0 (e6ac2724...); every .f32 re-verified after rsync to lambda
# lambda, 20:33:13Z-20:33:28Z, load 9.95 -> 13.17
qwen35_layer_obs dump MODEL prompt-4.ids obs-p4.bin 0,1,2,3 obs-p4            # rc 0, written=1500
qwen35_layer_obs dump MODEL prompt_token_ids.txt obs-p0.bin 4,28 obs-p0       # rc 0, written=750
#   manifests apr_obs_manifest_p4.sha256.tsv (d0de99e3...) / _p0 (ea779812...)
# lambda, 20:34:27Z-20:34:44Z, load 5.70 -> 4.43
python3 compare_layerwise.py p4   sub-p4 sub-p4.bin embd-p4 obs-p4.bin 0,1,2,3 layer_types.tsv MODEL GGUF_PY obs-p4 > sublayer_p4.tsv    # rc 0
python3 compare_layerwise.py orig sub-p0 sub-p0.bin embd-p0 obs-p0.bin 4,28   layer_types.tsv MODEL GGUF_PY obs-p0 > sublayer_orig.tsv  # rc 0
python3 layer_steps.py --ref p4:1 sublayer_p4.tsv sublayer_orig.tsv > layer_steps.tsv                                                  # rc 0
```

- `compare_layerwise.py` was extended with an optional 10th arg `APR_OBS_DIR`: the tensor list comes from the llama manifest, and DeltaNet states are also scored transposed. sha256 `54ccaca6fbf6db33f29f113b6689a8a7eae9466297191a15afe37e7755b7adb3`. Without that arg its output is unchanged.
- `sublayer_p4.tsv` (`83d18166…e7c4`) has 1500 rows and `sublayer_orig.tsv` (`2aad1c99…145b`) has 750. **Every row is `measured`, with 0 `U` rows.** The self-check (callback `result_output` == per-token row) is True at all 6 positions.
- `layer_steps.py` sha256 `d9a25a0d6560a2da67a6e98bbc07eaaa57f846e1c5907cf9de325fe814f9f1fe`; `layer_steps.tsv` `cb6a80c5…428e`.

**Residual-stream curve at p4 pos 1** (rel L2 = ‖apr−llama‖/‖llama‖; the step is charged to the point after it; sub = the sub-layer's own output):

| point | layer / type | rel L2 | step | cos | sub-layer output rel L2 |
|---|---|---|---|---|---|
| model.input_embed | - | 0.000000 | - | 1.000000 | - |
| **attn_residual-0** | 0 DeltaNet, mixer | 0.039963 | **+0.039963** | 0.999204 | linear_attn_out-0 0.041934 |
| l_out-0 | 0 DeltaNet, ffn | 0.053324 | +0.013361 | 0.998590 | ffn_out-0 0.063411 |
| l_out-1 | 1 DeltaNet | 0.064590 | +0.010907 (ffn) | 0.997977 | ffn_out-1 0.076211 |
| l_out-3 | 3 attention | 0.080963 | +0.009103 (ffn) | 0.996731 | attn_output-3 0.096991 |
| l_out-7 | 7 attention | 0.112255 | +0.008625 (ffn) | 0.993705 | attn_output-7 0.068198 |
| l_out-9 | 9 DeltaNet | 0.136550 | +0.013941 (ffn) | 0.990639 | ffn_out-9 0.206309 |
| l_out-11 | 11 attention | 0.157639 | +0.015059 (ffn) | 0.987689 | |
| l_out-12 | 12 DeltaNet | 0.177771 | +0.020993 (ffn) | 0.984729 | ffn_out-12 0.263831 |
| l_out-15 | 15 attention | 0.174635 | −0.007523 (ffn) | 0.985146 | |
| l_out-16 | 16 DeltaNet | 0.176957 | +0.019478 (ffn; mixer −0.017156) | 0.984385 | |
| l_out-19 | 19 attention | 0.230613 | +0.018682 (ffn) | 0.973098 | |
| attn_residual-20 | 20 DeltaNet, mixer | 0.248115 | +0.017502 | 0.969102 | linear_attn_out-20 0.320118 |
| attn_residual-21 | 21 DeltaNet, mixer | 0.284240 | +0.022195 | 0.959212 | |
| l_out-21 | 21 DeltaNet, ffn | 0.314435 | +0.030195 | 0.950480 | ffn_out-21 0.354726 |
| l_out-22 | 22 DeltaNet | 0.299569 | −0.009741 (ffn) | 0.954867 | |
| attn_residual-23 | 23 attention, mixer | 0.319942 | +0.020373 | 0.948530 | attn_output-23 0.216937 |
| l_out-23 | 23 attention, ffn | 0.332800 | +0.012858 | 0.944144 | ffn_out-23 0.308102 |
| result_norm | - | 0.331704 | −0.001096 | 0.944845 | - |

All 49 steps at all 6 positions are in `layerwise/layer_steps.tsv`.

### 5d. Observation (no threshold)

1. **Largest rel-L2 step at p4 pos 1:** `attn_residual-0`, layer 0, **gated DeltaNet**, **mixer**: 0.000000 → 0.039963. Within layer 0 the mixer step (0.039963) is larger than the FFN step (0.013361).
2. **The contrast positions step at the SAME point.** `attn_residual-0` is the largest step at orig pos 4 (0.033858), orig pos 28 (0.039555), p4 pos 2 (0.042473) and p4 pos 3 (0.036257). At p4 pos 0 it is second (0.037412), after `attn_residual-15` (0.085038). So the largest step is common to every position; it is not what separates p4 pos 1.
3. **Inside the layer-0 mixer (DeltaNet internals, all 6 positions, `sublayer_*.tsv`):**
   - `attn_norm-0` and `z-0` have rel L2 0.000000. `a_softplus-0`, `gate-0` and `beta_sigmoid-0` are ≤ 0.000132.
   - `conv_output_silu-0` is 0.0032–0.0041, `q/k/v_conv_predelta-0` ≤ 0.0067, and `attn_output-0` (delta-rule output) ≤ 0.0037.
   - `state_predelta-0` / `new_state-0` are ≤ 0.0053 in apr's own memory order. Transposed they give cos ≈ 0.001–0.005, so the two state layouts agree and the transpose does not.
   - `final_output-0` is ≤ 0.0043. **Then `linear_attn_out-0` is 0.034–0.042**, across the single `ssm_out` matmul.
4. **Kernel isolation** (`layerwise/kernel_isolation.py`, sha256 `1345a838aeed2db81ac54bacdec8726b665713816525ea7c82ae0f985f477697`; `kernel_isolation.tsv` `b23162d1…7367`; lambda 20:36:24Z, load 33.06; layers 0, 3, 9, 12, 16, 17, 20, 21, 23 × 6 positions). The reference is float64 `dequant(W)` from gguf-py applied to each engine's OWN dumped input.
   - `ssm_out` is **Q5_K** in every DeltaNet layer checked. **apr's output departs from the reference by ≤ 0.000001 at all 42 (layer, pos) rows; llama's by 0.016736–0.063286** (layer 0: 0.034146–0.042120).
   - apr dispatches Q5_K to `fused_q5k_parallel_matvec_into` (`crates/aprender-serve/src/gguf/inference/fused_matmul_into.rs:56`).
   - llama's CPU `type_traits` give Q5_K `vec_dot_type = GGML_TYPE_Q8_K` (`ggml/src/ggml-cpu/ggml-cpu.c:324`; Q4_K `:314` and Q6_K `:330` are also Q8_K). llama quantizes the activation before this dot product.
   - Q4_K matmuls depart by similar amounts in both engines. `z-0` is equal to 6 digits at all 6 positions (0.004775–0.006506). `z-N` over all 42 rows: apr 0.003710–0.014706, llama 0.003750–0.013164. Attention `attn_output` (layers 3, 23; 12 rows): apr 0.024501–0.040553, llama 0.023194–0.037069. The all-Q4_K FFNs (layers 3, 9, 12, 16; 24 rows): apr 0.017052–0.032559, llama 0.016680–0.031686.
   - FFNs with a **Q6_K** `ffn_down` (layers 0, 17, 20, 21, 23): apr departs less than llama at 29 of 30 (layer, pos) rows. Example: p4 pos 1 layer 21, apr 0.009503 vs llama 0.019409. The exception is orig pos 4 layer 23, apr 0.010004 vs llama 0.009533.
   - Control built into the table: apr `ssm_out` at 1e-6 proves the reference's dequant, orientation and shape are right, and `z-0` equal to 6 digits proves the input slice is the same tensor.
   - **At the largest step, the engine departing from `dequant(W)·x` is the reference, not apr.**
5. **What separates p4 pos 1 from the contrast positions** is the excess of its step over the largest step any contrast position takes at the same point (`layer_steps.tsv` summary):
   - `l_out-21` (DeltaNet FFN, Q6_K down): +0.021363
   - `l_out-16` (DeltaNet FFN, Q4_K): +0.014909
   - `l_out-12` (DeltaNet FFN, Q4_K): +0.013617
   - `attn_residual-20` (DeltaNet mixer): +0.011213
   - `l_out-17` (DeltaNet FFN, Q6_K): +0.010992
   - `l_out-9` (DeltaNet FFN, Q4_K): +0.010598

   The excess is spread over DeltaNet FFN sublayers in layers 9–21, not concentrated at one layer.

## Remaining unknowns, each with the command that would measure it

- ~~First layer where apr diverges; largest step at p4 pos 1; whether orig pos 4/28 step at the same layer.~~ Measured in §5c/§5d.
- ~~Sub-layer localisation inside the stepping layer.~~ Measured in §5d.3.
- **[U] Whether llama's Q8_K activation quantization reproduces llama's Q5_K `ssm_out` departure** (a mechanism proof for §5d.4, not only the source citation).
  Add a Q8_K block quantizer (256-element blocks, `d = max|x|/127`, int8 round) on the input side of `kernel_isolation.py`, run it for layer 0 at all 6 positions, and compare to llama `linear_attn_out-0`.
- **[U] Whether the p4-pos-1 excess in DeltaNet FFN sublayers (§5d.5) is made by the kernels or inherited from their inputs.**
  Run `python3 kernel_isolation.py MODEL GGUF_PY layer_types.tsv 0,1,...,23 p4:sub-p4:obs-p4:0,1,2,3 orig:sub-p0:obs-p0:4,28` and correlate the per-layer (llama − apr) kernel departure with the excess column of `layer_steps.tsv`.
- **[U] The logits gap with the Q5_K activation-quantization difference removed.**
  Write a GGUF copy with `blk.*.ssm_out.weight` re-encoded as F32 (gguf-py `GGUFWriter` over the reader), then re-run `run_sublayer.sh` (with `MODEL=` pointed at the copy) and `qwen35_layer_obs noop/dump`, then `compare_layerwise.py`.
- **[U] llama's own batched-vs-per-token spread per sub-layer** (a noise floor). Needs a batched dump sliced by position (producer extension); measure with `--dump-tensors` without `--per-token`.
- **[U] n > 1 for the apr observer dumps.** The logits of both dump runs are byte-equal to the n=4 subject. Re-run `qwen35_layer_obs dump` and `diff` the `apr_obs_manifest_*.sha256.tsv`.
- **[U] Whether the p4 early-position excess follows the markup content or the position.** Carried over from `VARIATION.md`.
- **The Qwen3.5 parity row stays [U], fail-closed. No threshold is set.**
- Whether to upstream the observer into #3114 is not decided here (orchestrator decision §8: after the measurement).

## Files

| file | what |
|---|---|
| `../qwen35-cpu-reference/producer/apr_raw_logits.cpp` | `--dump-tensors` eval-callback mode |
| `../qwen35-cpu-reference/producer/run_layerwise.sh` | intel driver: regression + dumps + manifest hashing |
| `layerwise/llama_tensor_names.tsv` | every callback node name, with op/type/ne |
| `layerwise/llama_manifest_p{4,0}.sha256.tsv` | dumped tensors: name, layer, pos, shape, bytes, file, sha256 |
| `layerwise/layer_types.tsv` | layer → full attention / gated-DeltaNet, with GGUF tensor evidence |
| `layerwise/qwen35_embd_dump.rs`, `apr_embd_manifest.sha256.tsv` | apr embedding-row harness (public API only) and its output hashes |
| `layerwise/compare_layerwise.py`, `curves_{p4,orig}.tsv`, `curves_*.selfcheck.txt` | per-(tensor, pos) comparison |
| `layerwise/derive_final_norm.py`, `final_norm_lstsq.tsv` | lstsq-derived `result_norm` / `l_out-23` direction, with control |
| `layerwise/run_layerwise.transcript` | intel transcript for the llama runs |
| `../qwen35-cpu-reference/producer/run_sublayer.sh`, `layerwise/run_sublayer.transcript` | §5 intel driver + transcript: sub-layer dumps, regressions, manifests |
| `layerwise/observer_mapping.tsv` | §5a llama name ↔ source line ↔ apr fn ↔ apr line |
| `layerwise/qwen35_layer_obs.rs` | §5 apr harness (noop invariance / dump) over `forward_single_qwen35_observed` |
| `layerwise/observer_invariance.transcript`, `apr_obs_dump.transcript`, `observer_gate.txt` | §5b invariance shas, apr dump runs, qwen35 lib test count |
| `layerwise/{apr_obs,llama_sub}_manifest_{p4,p0}.sha256.tsv` | every dumped `.f32` (1500 + 750 per engine) with sha256 |
| `layerwise/sublayer_{p4,orig}.tsv`, `sublayer_*.selfcheck.txt`, `observer_compare.transcript` | §5c per-(tensor, pos) comparison |
| `layerwise/layer_steps.py`, `layer_steps.tsv` | §5c/§5d residual-stream steps, contrast ranks, p4-pos-1 excess |
| `layerwise/kernel_isolation.py`, `kernel_isolation.tsv`, `kernel_isolation.transcript` | §5d.4 float64 dequant reference vs each engine's kernel |
