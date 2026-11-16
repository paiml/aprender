# qwen4_exp per-op golden vectors (QWEN4 S2, #7175, master #7169)

Operator ruling OPT2: one-off Python goldens allowed; only values + command + versions + reference sha are committed.
No Python ships or enters CI. No model weights were downloaded: every parameter is a tiny seeded random tensor.

## Provenance
- transformers checkout: `https://github.com/huggingface/transformers` @ `36e937ae761dea806a5b9fbb382e880378ad5435` (shallow clone of main, 2026-11-16), files under `src/transformers/models/qwen4_exp/`
- python 3.11.10, torch 2.15.0+cpu (CPU), transformers 5.24.0.dev0 (editable from the checkout), numpy 2.4.6
- generator: `gen_goldens.py` sha256 `7c9ab90da8e58abf4e3ce307f6d1da6721594d4db48ae474ed6e917d37694095` (not in repo; copy at /mnt/nvme-raid0/cop-inbox/handoff/aprender-cop/qwen4-gen_goldens.py)
- commands:
  ```
  git clone --depth 1 https://github.com/huggingface/transformers $W/ref
  uv venv --python 3.11 $W/venv
  uv pip install --python $W/venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch numpy
  uv pip install --python $W/venv/bin/python -e $W/ref
  $W/venv/bin/python -I $W/gen_goldens.py $W/out        # W=/mnt/nvme-raid0/tmp/qwen4-goldens
  ```
- Determinism: `torch.use_deterministic_algorithms(True)`, fp32, CPU; every tensor drawn from `torch.Generator().manual_seed(N)` (module params 100/200/300/400, inputs next seeds); two runs produced byte-identical output.
- Param init: HF zero-inits RMSNorm weights (effective scale 1+w), so ALL parameters are overwritten with seeded N(0, s) (matrices s=0.5/sqrt(fan_in), vectors s=0.2) so every term is exercised.

## File format
Each `<op>.json`: `config`, `params` (keys = torch state_dict names, relative to the module), `buffers`, `inputs`, `intermediates`, `output(s)`.
A tensor is `{"shape":[..],"dtype":"f32|i64|bool01","data":[flat row-major]}`. f32 values are exact float32 printed as shortest-roundtrip doubles. Linear weights are torch layout [out,in]; conv1d weights [C,1,K].

## Tiny config (shared; every structural hyperparameter keeps its role)
hidden_size 16 (real 2560), head_dim 8 (256), heads 2/kv 1 (24/2), rope theta 1e4 (1e7) partial_rotary 0.5 (0.25 -> rotary_dim 4), mrope_section [1,1,0] (positions identical on all 3 axes, so inert),
hc_count 4 (4, unchanged), hc_lowrank 8 (320), ngram_size 3 (3, unchanged), heads_per_ngram 2 (8), ngram_vocab_size_base 50 (20,000,000), make_ngram_vocab_size_divisible_by 8 (128),
ple_embed_dim 16 (2560), ple_conv_kernel_size 4 (4, unchanged), seed 1234 (config default; the real checkpoint's value must be read from its config.json), eos_token_id 7,
vocab_size 248320 (UNCHANGED so the n-gram multiplier derivation matches the real model), indexer_n_heads 2 (4), indexer_kv_heads 1 (1), indexer_head_dim 8 (128), indexer_budget 8 (2048), indexer_compress_ratio 2 (4) -> block_topk 4 (512), rms_norm_eps 1e-6, output_gate_type sigmoid.
The module-level `config` block in each JSON lists the exact values that op used.

## Ops
| golden | op | reference (at sha above) | status |
|---|---|---|---|
| indexer.json | sparse-attention lightning indexer, returns per-query selected-token bool mask | modeling_qwen4_exp.py:670-773 (class Qwen4ExpTextQSAIndexer) | ok, isolated; includes block-score intermediate for batch 0 / query 15; batch 1 has 3 left-pad tokens |
| indexed_attention.json | whole Qwen4ExpTextAttention incl. indexer mask, gated q_proj, sigmoid output gate | modeling_qwen4_exp.py:813-897 | ok (smallest enclosing module for the attention-side output gate) |
| ngram.json | n-gram hashed embedding | modeling_qwen4_exp.py:1037-1178 (Qwen4ExpTextNGramEmbedding, _splitmix64, _build_layer_multipliers, _find_nth_prime_after) | ok, isolated; tokens include eos id 7 to exercise segment resets |
| ple.json | per-layer embedding (n-gram emb -> key/value proj -> stream gate -> dilated depthwise conv) | modeling_qwen4_exp.py:1180-1253 (Qwen4ExpTextPLELayer) | ok, isolated; cache=None |
| hc_residual.json | hyper-connection multi-stream residual, plus final mixer and stream init | modeling_qwen4_exp.py:999-1034 (Qwen4ExpTextGatedResidual); combine formula at 1290-1304 (decoder layer); mixer/repeat at 1359-1470 (TextModel) | ok, isolated; the "combined" tensor is the decoder-layer formula replicated in the script from those lines |
| output_gate.json | output_gate_type=sigmoid in the Gated DeltaNet norm | modeling_qwen4_exp.py:170-186 (Qwen4ExpTextRMSNormGated, used at :500) | ok, isolated (the norm only; GDN conv/delta-rule not dumped) |
| MTP head | -- | none | NOT GOLDEN-ABLE, see below |

## MTP: no golden
At this sha `modeling_qwen4_exp.py` has no MTP module: it only lists `^mtp.*` in `_keys_to_ignore_on_load_unexpected` (:1317, :1584), and `Qwen4ExpTextConfig` has no `mtp*` fields (no `num_mtp_layers`, which `PreTrainedConfig.get_mtp_config` requires, configuration_utils.py:1596-1608). The generic `MtpModel`/`MtpLayer` (modeling_layers.py:329-650; enorm/hnorm on embeds and previous hidden, concat, eh_proj, decoder block, post_norm) is not wired to qwen4_exp, and its decoder block would receive hidden_size features where Qwen4ExpTextDecoderLayer demands hc_count*hidden_size. Weight renaming for qwen4_exp MTP checkpoints is not in conversion_mapping.py either (the `MtpModel` mapping at :2303 is generic). A golden would have to invent the structure, so none is committed. Needs the real checkpoint's mtp.* key list, or a later transformers revision.

## Plain-math descriptions
See QWEN4-OPS.md in the cop handoff dir (/mnt/nvme-raid0/cop-inbox/handoff/aprender-cop/QWEN4-OPS.md); the per-op math is repeated below in short form.
- RMSNorm (all norms here): y = x / sqrt(mean(x^2 over group)+eps) * (1 + w), computed in fp32; `group_size` splits the last dim into groups normalised independently.
- indexer: see QWEN4-OPS.md section 1.
- ngram, ple, hc, output gate: see QWEN4-OPS.md sections 2-5.
