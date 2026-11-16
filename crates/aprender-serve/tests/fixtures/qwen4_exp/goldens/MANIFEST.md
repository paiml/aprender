# qwen4_exp per-op golden vectors (QWEN4 S2, #7175, master #7169)

Operator ruling OPT2: one-off Python goldens allowed; only values + command + versions + reference sha are committed.
No Python ships or enters CI. No model weights were downloaded: every parameter is a tiny seeded random tensor.

## Provenance
- transformers checkout: `https://github.com/huggingface/transformers` @ `36e937ae761dea806a5b9fbb382e880378ad5435` (shallow clone of main, 2026-11-16), files under `src/transformers/models/qwen4_exp/`
- python 3.11.10, torch 2.15.0+cpu (CPU), transformers 5.24.0.dev0 (editable from the checkout), numpy 2.4.6
- generator: `gen_goldens.py` sha256 `aabae9c91b9a78a115a4f2e9310cf90e747d3d6823e8994ff7f981a215e8ba34` (the script itself is not in the repo)
- commands:
  ```
  git init $W/ref && git -C $W/ref fetch --depth 1 https://github.com/huggingface/transformers 36e937ae761dea806a5b9fbb382e880378ad5435 && git -C $W/ref checkout FETCH_HEAD
  uv venv --python 3.11 $W/venv
  uv pip install --python $W/venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch numpy
  uv pip install --python $W/venv/bin/python -e $W/ref
  $W/venv/bin/python -I $W/gen_goldens.py $W/out        # W = any scratch dir outside the repo
  ```
- Determinism: `torch.use_deterministic_algorithms(True)`, fp32, CPU; every tensor drawn from `torch.Generator().manual_seed(N)` (indexer/attention params seed 118 and inputs seed 119 (chosen by the generator, see Top-k margin), other ops params 200/300/400 with inputs 201/202/301-304/401-402); two runs produced byte-identical output.
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
| indexer.json | sparse-attention lightning indexer, returns per-query selected-token bool mask | modeling_qwen4_exp.py:670-773 (class Qwen4ExpTextQSAIndexer) | ok, isolated; includes block-score intermediate for batch 0 / query 15; batch 1 has 3 left-pad tokens, and the generator sets the self-visible diagonal for its 3 fully-masked pad query rows by hand (see the JSON note) |
| indexed_attention.json | whole Qwen4ExpTextAttention incl. indexer mask, gated q_proj, sigmoid output gate | modeling_qwen4_exp.py:813-897 | ok (smallest enclosing module for the attention-side output gate); also dumps the attention's own selection_mask |
| ngram.json | n-gram hashed embedding | modeling_qwen4_exp.py:1037-1178 (Qwen4ExpTextNGramEmbedding, _splitmix64, _build_layer_multipliers, _find_nth_prime_after) | ok, isolated; tokens include eos id 7 to exercise segment resets |
| ple.json | per-layer embedding (n-gram emb -> key/value proj -> stream gate -> dilated depthwise conv) | modeling_qwen4_exp.py:1180-1253 (Qwen4ExpTextPLELayer) | ok, isolated; cache=None |
| hc_residual.json | hyper-connection multi-stream residual, plus final mixer and stream init | modeling_qwen4_exp.py:999-1034 (Qwen4ExpTextGatedResidual); combine formula at 1290-1304 (decoder layer); mixer/repeat at 1359-1470 (TextModel) | ok, isolated; the "combined" tensor is the decoder-layer formula replicated in the script from those lines |
| output_gate.json | output_gate_type=sigmoid in the Gated DeltaNet norm | modeling_qwen4_exp.py:170-186 (Qwen4ExpTextRMSNormGated, used at :500) | ok, isolated (the norm only; GDN conv/delta-rule not dumped) |
| MTP head | -- | none | NOT GOLDEN-ABLE, see below |

## MTP: no golden
At this sha `modeling_qwen4_exp.py` has no MTP module: it only lists `^mtp.*` in `_keys_to_ignore_on_load_unexpected` (:1317, :1584), and `Qwen4ExpTextConfig` has no `mtp*` fields (no `num_mtp_layers`, which `PreTrainedConfig.get_mtp_config` requires, configuration_utils.py:1596-1608). The generic `MtpModel`/`MtpLayer` (modeling_layers.py:329-650; enorm/hnorm on embeds and previous hidden, concat, eh_proj, decoder block, post_norm) is not wired to qwen4_exp, and its decoder block would receive hidden_size features where Qwen4ExpTextDecoderLayer demands hc_count*hidden_size. Weight renaming for qwen4_exp MTP checkpoints is not in conversion_mapping.py either (the `MtpModel` mapping at :2303 is generic). A golden would have to invent the structure, so none is committed. Needs the real checkpoint's mtp.* key list, or a later transformers revision.

## Top-k margin (no ties)
The generator asserts, on every (batch, query) row where the number of complete blocks exceeds block_topk (11 rows), that the k-th and (k+1)-th block scores differ by at least 1e-3, and fails otherwise (it scans seeds until this holds). Recorded minimum margin: 0.0198 (min of k-th minus (k+1)-th block score, same value in indexer.json and indexed_attention.json since they share parameters and inputs); also stored in each JSON under `topk_margin`. A Rust top-k therefore matches regardless of tie-breaking.

## Plain-math descriptions (what the Rust implementer builds from)
Notation: H hidden, C hc_count, D head_dim, g group size. fp32 unless stated; the real model is bf16 with fp32 islands (norms, scores, gate math).

RMSNorm (all norms here except the gated norm): y = x * rsqrt(mean(x^2) + eps) * (1 + w), computed in fp32 then cast back. w is an OFFSET from 1 (zero-initialised). With group_size g the last dim is split into groups of g, each normalised independently (hc_norm, norm_key, norm_query, norm_conv use g = H over C*H features).

### 1. Sparse-attention lightning indexer (indexer.json)
Layer type `indexed_attention`. Params: index_qk_proj [(nh+nkv)*dh, H] (no bias), q_layernorm and k_layernorm RMSNorm(dh). Real: nh 4, nkv 1 (must be 1), dh 128, budget 2048, ratio 4, so block_topk = budget/ratio = 512. Per query position t, with V_t the ascending indices of keys visible in the 4D mask:
1. qk = index_qk_proj(h); q = nh heads of dh, k_raw = one head of dh per token. q = RMSNorm(q); q = RoPE(q, cos/sin at t). Partial rotary: only the first rotary_dim = cos.shape[-1] dims rotate (rotate_half, NeoX split halves), the rest pass through.
2. nb = floor(|V_t| / ratio). Blocks are consecutive groups of `ratio` entries of V_t, so they are over visible tokens and left padding shifts them. The tail is the remaining |V_t| mod ratio visible tokens.
3. pooled_b = mean of k_raw over block b (fp32), then RMSNorm_k, then RoPE at the position of the block's first token.
4. score_b = sum over heads of relu(q_head . pooled_b), divided by sqrt(dh). (relu, then SUM over heads, then scale.)
5. Keep the top min(block_topk, nb) blocks. Selected tokens = all tokens of kept blocks plus ALL tail tokens. If nb = 0 only the tail (everything visible).
6. Output bool mask [B,1,Q,KV], true at selected tokens. The attention combines it with the causal mask by AND (bool mask) or by filling ~mask with the dtype minimum (float mask), then runs ordinary softmax attention.
State: k_raw per token must be cached, plus full position ids. Decode re-derives blocks every step from the whole visible history.

### 1b. Attention and its sigmoid output gate (indexed_attention.json)
q_proj outputs [nh*D*2]: per head [query D | gate D]. q_norm and k_norm are per-head RMSNorm(D) applied BEFORE RoPE (partial rotary). GQA softmax attention with scale D^-0.5 and the indexer mask above, then attn_out *= sigmoid(gate) elementwise (gate flattened to nh*D), then o_proj. Real: H 2560, nh 24, nkv 2, D 256, partial_rotary 0.25 (rotary_dim 64), mrope interleaved [11,11,10], theta 1e7.

### 2. N-gram embedding (ngram.json)
ids [B,S] -> [B,S, ple_embed_dim] (ngram_heads = (ngram_size-1)*heads_per_ngram heads, each ple_embed_dim/ngram_heads wide). Table nn.Embedding(padded_total_vocab, ple_embed_dim/ngram_heads), padded to a multiple of make_ngram_vocab_size_divisible_by. In the real checkpoint it is stored as split_ngram_parts shards "ngram_embedding.shard_*.weight" concatenated on dim 0.
Hash, all int64 with wraparound multiply and no overflow check:
- Head h has vocab p_h = the (g+1)-th prime strictly after (ngram_vocab_size_base - 1), where g = ple_layer_index*ngram_heads + h. Offsets are the running sum of p_h.
- Multipliers m_i for i < ngram_size: seed_i = (config.seed + 10007*ple_layer_index + 0x9E3779B97F4A7C15*(i+1)) mod 2^64; m_i = 2*(splitmix64(seed_i) mod max(1, (2^63-1)//vocab_size//2)) + 1. vocab_size is the UNIGRAM vocab (248320 real). splitmix64(v): v += 0x9E3779B97F4A7C15; v = (v ^ (v>>30)) * 0xBF58476D1CE4E5B9; v = (v ^ (v>>27)) * 0x94D049BB133111EB; return v ^ (v>>31) (all mod 2^64).
- Context = previous ngram_size-1 tokens (initially eos); history = [ctx, ids]. shifted_k = history shifted right by k positions, where a position that would cross an eos boundary (segment start is after the previous eos) or is < 0 is replaced by eos (valid iff position-in-segment >= k).
- For n = 2..ngram_size: mixed = XOR over j < n of (shifted_j * m_j); heads for n are slice [(n-2)*hpn, (n-1)*hpn); id = (mixed mod p_h) + offset_h. Concatenate all head ids, look up each, flatten the last two dims.
State: last ngram_size-1 token ids per sequence (padded with eos).
Real: ngram_size 3, heads_per_ngram 8 (16 heads), ple_embed_dim 2560 (160 per head), base 20,000,000, divisor 128. config.seed default 1234; read the real checkpoint's value.

### 3. Per-layer embedding PLE (ple.json)
Only on linear_attention layers listed (1-indexed) in ple_layer_ids (real [2]; ple_layer_index = position in that list, 0). Acts on the C*H hyper stream x before the attention hyper connection: x <- x + PLE(x, ids).
emb = NGram(ids) [B,S,E]. key = RMSNorm_g(key_proj(emb)) [B,S,C*H] viewed [C,H]; value = value_proj(emb) [B,S,H]; q = RMSNorm_g(x) viewed [C,H]. gate_c = (key_c . q_c)/sqrt(H) per stream, then signed sqrt: gate = sqrt(max(|gate|, 1e-6)) * sign(gate). gv_c = sigmoid(gate_c) * value (broadcast over streams), flattened to [B,S,C*H]. out = gv + silu(conv(RMSNorm_g(gv))), where conv is depthwise (channels = groups = C*H), kernel K = ple_conv_kernel_size (4), DILATION = ngram_size (3), no bias, LEFT zero pad of (K-1)*dilation = 9, no right pad. Padding rows: gv rows masked to zero and ids replaced by eos at pads.
State: last 9 columns of the normed gv per channel plus the n-gram context tokens.

### 4. hc multi-stream residual (hc_residual.json)
The residual stream is C copies of H flattened to [B,S,C*H]; the input embedding is repeated C times at model input; after the last layer a mixer (use_combine = false) collapses to H.
GatedResidual(x): xn = RMSNorm_g(x) (g = H, (1+w)); a = silu(Wdown xn / C) [lowrank]; mix = sigmoid(Wup a), viewed [C,H]; mixed_input = mean over C of (mix * xn viewed [C,H]), shape [B,S,H], which feeds the sublayer. If use_combine: inj = 2*sigmoid((Wblock xn)/C), shape [B,S,C]. It returns (mixed_input, x_raw, inj) where x_raw is the UN-normalised input stream.
Decoder layer, for both the attention and the MoE sublayer: (m, x_raw, inj) = HC(x); y = sublayer(m); x = x_raw + flatten(y[..., None, :] * inj[..., :, None]). Each layer has an attn and an mlp hyper connection with their own params.
Final mixer: no block_inject_weight; its output is the gate-weighted mean of the per-stream normed streams, mean_c(mix_c * norm_c(x)). There is NO separate final norm; that output goes straight to the LM head.
Params per HC: hc_norm.weight [C*H], input_mix_weight_down.weight [lowrank, C*H], input_mix_weight_up.weight [C*H, lowrank], block_inject_weight.weight [C, C*H] (absent for the final mixer). Stateless.

### 5. Output gate (output_gate.json, plus 1b)
With output_gate_type = sigmoid: attention side, attn_out *= sigmoid(gate) (see 1b). Gated DeltaNet side, Qwen4ExpTextRMSNormGated: y = (w * x * rsqrt(mean(x^2)+eps)) * sigmoid(z) per head_v_dim row, with PLAIN weight w (init ones, not 1+w); the activation is silu when output_gate_type = silu. Stateless.
