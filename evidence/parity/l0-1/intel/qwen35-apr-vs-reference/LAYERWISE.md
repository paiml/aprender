# Qwen3.5-0.8B Q4_K_M: layer-wise apr vs llama.cpp on the prompt-4 chat-markup outlier (PMAT-3091)

Question: where in the forward pass does apr first diverge from llama.cpp on prompt 4, which is where `VARIATION.md` §5 item 5 puts the minimum cosine (0.961309 at pos 1)?
**No threshold is set anywhere. The Qwen3.5 parity row stays [U], fail-closed.**

**Result, in one line:**
- The EMBEDDING output is byte-identical at every measured position.
- The divergence is already present in the last layer's residual output (`l_out-23`), derived below.
- The layer where it first appears is **[U]**. Realizar exposes no per-layer hidden state through a public API, and this ticket may not edit `crates/`. The minimal patch is at the end.

## Tree (every measurement below)

| side | tree |
|---|---|
| evidence worktree (layer-3091), base of this commit | `1976315fe` (PMAT-3091-flip-variation) |
| apr subject: #3114 head `31448f6c3`, detached at `/tmp/layer3091/apr-31448`, examples copied in uncommitted | logits binaries as in `VARIATION.md` (`qwen35_raw_logits` `ad4397d5…3d8e`). New: `qwen35_embd_dump` sha256 `f442a70658538b9df66d541e825eac6aa07e7298132d9da860f6baf30340d1b7` |
| reference: llama.cpp `d1d3c3396` | producer `apr_raw_logits` rebuilt with `--dump-tensors`, sha256 `5c5df52d75b514723f706b93aede9939f58e1df178d2b631ad0ad0038994dd06` |
| model | `Qwen3.5-0.8B-Q4_K_M.gguf`, sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` (the same on intel and lambda) |

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
`layerwise/qwen35_embd_dump.rs` dumps the former. Its embedding rows' sha256 are in the intel transcript of this ticket: pos0 `0f2822b1…5800`, pos1 `b9fec06c…9538`, pos2 `22ba0638…b931`, pos3 `58800e60…7c`, orig pos4 `91c6f7d1…bd1ca`, orig pos28 `42840b96…47ff1`.

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

`layerwise/compare_layerwise.py` (sha256 `837aa8f8961776f0d612393ce4b93cff52cea003a85db6b7f9641004c502c33a`) gives, per (tensor, pos), cosine, max |apr−llama| and relative L2 ‖apr−llama‖/‖llama‖.
`l_out-N` and `result_norm` rows carry `U` for apr, plus llama's own L2 norm.
Full per-position curves:
- `layerwise/curves_p4.tsv` (`5c75e221…a029`)
- `layerwise/curves_orig.tsv` (`74bc5277…c33a`)

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
| `l_out-0` … `l_out-22` | **[U]** for apr. llama's values are dumped and hashed, and llama's L2 norm per layer is in `curves_*.tsv` |
| `l_out-23` | direction only, derived (3b) |
| `result_norm` | derived (3b) |
| `result_output` | yes (3b) |

**Largest step in relative L2 at p4 pos 1: [U].** Between the two ends that are measured or derived, rel L2 goes from 0.000000 at the embedding to 0.331704 at the (derived) `result_norm`. The step is somewhere in the 24 layers, 18 linear and 6 full-attention. Which layer holds it cannot be said without apr-side dumps, so **whether the contrast positions step at the same layer is [U] too**.

## 4. Observation (not a verdict, no threshold)

1. The embedding lookup is exact for the markup ids (Q6_K dequant is identical across apr, llama.cpp and gguf-py). The prompt-4 excess is not a token-embedding or quant-block-kind effect.
2. The excess is already in the residual stream when it leaves the last layer.
   - At p4 pos 1, the derived `l_out-23` direction has cos 0.944144; the method's own control there is 0.999966.
   - The contrast positions are at 0.996656 and 0.997789.
   - The lm_head plus final RMSNorm do not create it: the derived `result_norm` and the measured logits track each other at every position (rel L2 0.33 vs 0.28 at p4 pos 1).
3. The ordering across positions is the same at `l_out-23`, `result_norm` and logits: p4 pos 1 < pos 2 < pos 3 < pos 0 < orig pos 4 < orig pos 28.
4. llama's own `l_out-23` L2 norm at p4 pos 1 (8.40) is lower than at p4 pos 0 (10.68) (`curves_p4.tsv`). This is context only.

## Remaining unknowns, each with the command that would measure it

- **[U] First layer where apr diverges; largest relative-L2 step at p4 pos 1; whether orig pos 4/28 step at the same layer.**
  Needs an apr per-layer hidden dump, which requires editing `crates/` (out of scope here). Minimal patch:
  - Change `crates/aprender-serve/src/gguf/inference/forward/forward_qwen35.rs`, fn `forward_single_qwen35` (lines 700–763 at `31448f6c3`).
  - After each `forward_deltanet` / `forward_attention` call in the layer loop (lines 723–746), call an optional observer with `(il, &hidden)`.
  - Call it once more with `out_normed` after `rms_norm_into` (line 747).
  - This is best exposed as a new `pub fn forward_single_qwen35_observed(&self, token_id, cache, position, obs: &mut dyn FnMut(&str, usize, &[f32]))`, with `forward_single_qwen35` delegating with a no-op, so there is no behaviour change and it is zero-cost when unused.
  - Then a copy of `qwen35_embd_dump.rs` writes `pos<P>/l_out-<il>.f32`, and `compare_layerwise.py` fills the U rows unchanged, because file naming already matches the llama dump.
- **[U] Sub-layer localisation inside the stepping layer.** llama already exposes `attn_norm-N`, `attn_residual-N`, `attn_post_norm-N`, `ffn_out-N`, `linear_attn_out-N` and `attn_output-N`.
  Measure with `--dump-regex 'attn_residual-[0-9]+|ffn_out-[0-9]+|l_out-[0-9]+'` plus the matching observer names on the apr side.
- **[U] llama's own batched-vs-per-token spread per layer**, a noise floor for each layer.
  Would need a batched dump (the eval callback sees the whole ubatch, so dump rows would be sliced by position), measured with `--dump-tensors` without `--per-token` after extending the producer.
- **[U] Whether the p4 early-position excess follows the markup content or the position.** Carried over from `VARIATION.md`.
- **[U] n > 1 for the dump runs.** Byte determinism of the per-token path is n=3 in `PER-TOKEN.md`, and the dump runs' logits equal those bytes. Re-run `run_layerwise.sh` and `cmp` the `.f32` hashes to measure it.

## Files

| file | what |
|---|---|
| `../qwen35-cpu-reference/producer/apr_raw_logits.cpp` | `--dump-tensors` eval-callback mode |
| `../qwen35-cpu-reference/producer/run_layerwise.sh` | intel driver: regression + dumps + manifest hashing |
| `layerwise/llama_tensor_names.tsv` | every callback node name, with op/type/ne |
| `layerwise/llama_manifest_p{4,0}.sha256.tsv` | dumped tensors: name, layer, pos, shape, bytes, file, sha256 |
| `layerwise/layer_types.tsv` | layer → full attention / gated-DeltaNet, with GGUF tensor evidence |
| `layerwise/qwen35_embd_dump.rs` | apr embedding-row harness (public API only) |
| `layerwise/compare_layerwise.py`, `curves_{p4,orig}.tsv`, `curves_*.selfcheck.txt` | per-(tensor, pos) comparison |
| `layerwise/derive_final_norm.py`, `final_norm_lstsq.tsv` | lstsq-derived `result_norm` / `l_out-23` direction, with control |
| `layerwise/run_layerwise.transcript` | intel transcript for the llama runs |
