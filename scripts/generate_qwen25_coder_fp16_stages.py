#!/usr/bin/env python3
"""
SHIP-007 layer-0 oracle bisection — generate HF FP16 reference stage tensors
for Qwen2.5-Coder-7B-Instruct.

Loads the canonical HuggingFace `Qwen/Qwen2.5-Coder-7B-Instruct` model at
FP16, runs ONE forward pass on the canonical SHIP-007 prompt
"What is 2+2?", and dumps each of the 13 directly-instrumentable layer-0
forward stages plus the 2 whole-model stages to the same APRT byte format
that `apr trace --save-tensor` produces. The output directory mirrors the
APR layout (`layer-0/<stage>.bin` per-layer + `<stage>.bin` whole-model)
so `apr diff --values <apr>.bin <hf>.bin` works element-wise.

This is the **HF reference** side of the SHIP-007 layer-0 element-wise
diff bisection. Pair with `apr trace --save-tensor` on the canonical
APR teacher to find the stage at which APR forward_traced diverges from
HF FP16 ground truth.

Usage:
    uv run --with torch --with transformers --with safetensors \\
        --with accelerate scripts/generate_qwen25_coder_fp16_stages.py \\
        --output /tmp/qwen25-coder-7b-hf-fp16-stages

Requirements (already met on lambda-labs noah-Lambda-Vector 2026-05-03):
    - HF cache must contain `Qwen/Qwen2.5-Coder-7B-Instruct` FP16
      safetensors (~15 GB, 4 shards). Confirmed at
      ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/.
    - RTX 4090 (24 GB VRAM) — fits 7B FP16 single-device with room to spare.

Captured stages (13 per-layer + 2 whole-model = 15/16):
    Per-layer (layer 0 only by default; --layers expands):
      embedding (model.embed_tokens out)
      attn_norm (input_layernorm out)
      attn_out (self_attn out — post-O-proj, includes Q/K/V/RoPE/softmax/V)
      post_attn_residual (hidden after attn residual add)
      ffn_norm (post_attention_layernorm out)
      ffn_gate (mlp.gate_proj out)
      ffn_up (mlp.up_proj out)
      ffn_silu (silu(gate_proj) — derived in fwd hook)
      ffn_swigl (silu(gate) * up_proj — derived)
      ffn_out (mlp.down_proj out)
      post_ffn_residual (hidden after FFN residual add)
    Whole-model:
      final_norm (model.norm out)
      lm_head (model.lm_head out — last-token logits, dim_product=152064)

    Attention sub-stages (gated on --with-attn-substages, default ON):
      qkv_matmul (concat of pre-bias q_proj+k_proj+v_proj outputs)
      qkv_bias (post-bias)
      q_post_rope (Q after apply_rotary_pos_emb, pre-repeat_kv)
      k_post_rope (K after apply_rotary_pos_emb, pre-repeat_kv)
      attn_scores (Q·Kᵀ * scaling, pre-mask, pre-softmax)
      attn_softmax (softmax(scores+mask), pre-V)
      attention (softmax @ V, pre-O-proj)

The 4 stages between qkv_bias and attn_out require a per-layer monkeypatch
on `Qwen2Attention.forward` because they are intermediate tensors inside
the attention module, not module inputs/outputs that hooks can see. The
patch closes over a shared `captured` dict and emits the 4 stages for each
target layer index; non-target layers fall through to the original eager
forward.

Output schema (APRT byte format, identical to `apr trace --save-tensor`):
    Header: 12 bytes
        magic:        b'APRT' (4 bytes)
        layer:        u32 LE (per-layer index, or 0xFFFFFFFF for whole-model)
        dim_product:  u32 LE (number of f32 elements in body)
    Body: dim_product * 4 bytes (f32 LE)

`apr diff --values <apr>.bin <hf>.bin --limit 64` reports max|diff|, RMS,
cosine, and top-K element-wise divergences. The first stage where cosine
drops below 0.999 (Q4K noise floor for the APR side) is the bug location.
"""

from __future__ import annotations

import argparse
import datetime
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_PROMPT = "What is 2+2?"
EXPECTED_VOCAB_SIZE = 152064
EXPECTED_HIDDEN_DIM = 3584
APRT_MAGIC = b"APRT"
WHOLE_MODEL_LAYER = 0xFFFFFFFF


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model name (default: {DEFAULT_MODEL})")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help=f"Test prompt (default: {DEFAULT_PROMPT!r}); MUST match the APR-side prompt")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory; per-layer stages → <out>/layer-N/<stage>.bin, whole-model → <out>/<stage>.bin")
    p.add_argument("--layers", default="0",
                   help="Comma-separated layer indices to capture (default: '0' = layer-0 only)")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16",
                   help="Model dtype (default: float16, matches APR Q4K dequant precision)")
    p.add_argument("--device", default="cuda",
                   help="Device for forward pass (default: cuda; FP16 7B fits on 24GB VRAM)")
    p.add_argument("--with-attn-substages", dest="with_attn_substages", action="store_true",
                   default=True,
                   help="Capture 4 attention sub-stages via Qwen2Attention.forward monkeypatch "
                        "(q_post_rope, k_post_rope, attn_scores, attn_softmax). Default: ON. "
                        "Required for FALSIFY-ATTN-SUB-004 LIVE bisection (§47.6 step 7).")
    p.add_argument("--no-attn-substages", dest="with_attn_substages", action="store_false",
                   help="Skip the attention sub-stage monkeypatch (legacy 13-stage capture).")
    return p.parse_args()


def write_aprt_file(path: Path, layer: int, values: np.ndarray) -> None:
    """Write APRT header + f32 LE body. Matches `apr trace --save-tensor` byte format exactly."""
    assert values.dtype == np.float32, f"expected float32, got {values.dtype}"
    dim_product = values.size
    header = APRT_MAGIC + struct.pack("<II", layer, dim_product)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        # tobytes() preserves NaN bits + endianness on x86_64 (little-endian native)
        f.write(values.astype(np.float32, copy=False).tobytes())
    actual = os.path.getsize(path)
    expected = 12 + dim_product * 4
    assert actual == expected, f"{path.name}: size {actual} != expected {expected}"


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent, text=True
        ).strip()
    except Exception as e:
        return f"<git rev-parse failed: {e}>"


def to_fp32_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach + .float() (cast to fp32) + cpu + numpy. Standard pattern for fixture export."""
    return t.detach().float().cpu().numpy().astype(np.float32, copy=False)


def install_attn_substages_patch(
    model: torch.nn.Module,
    layers_to_capture: list[int],
    captured: dict[tuple[int, str], np.ndarray],
) -> None:
    """Install per-layer monkeypatches on `Qwen2Attention.forward` to capture
    the 4 attention sub-stages: q_post_rope, k_post_rope, attn_scores, attn_softmax.

    Pre-condition: model must be loaded with `attn_implementation="eager"` so that
    the inlined eager attention path matches the patched forward semantics
    (else attn_scores + attn_softmax may be skipped by sdpa/flash-attn fast paths).

    The patch is applied PER INSTANCE on the requested layers' `.self_attn` modules
    (not globally on the class). Non-target layers retain the original forward.

    Per `contracts/trace-attn-sub-stages-v1.yaml` v1.1.0 SUB-004 invariant:
    "9-element cosine sequence layer-0 [attn_norm, qkv_matmul, qkv_bias, q_post_rope,
    k_post_rope, attn_scores, attn_softmax, attention, attn_out]".
    """
    from transformers.models.qwen2.modeling_qwen2 import (  # type: ignore[import-untyped]
        apply_rotary_pos_emb,
        repeat_kv,
    )

    target_layer_idxs = set(layers_to_capture)

    def traced_forward(self, hidden_states, position_embeddings, attention_mask=None,
                       past_key_values=None, **kwargs):
        # Mirrors transformers 5.x Qwen2Attention.forward but inlines eager_attention_forward
        # so we can capture intermediates. Only target layers go through this path.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Capture #1+#2: post-RoPE Q and K (pre-repeat_kv, GQA-shaped).
        captured[(self.layer_idx, "q_post_rope")] = to_fp32_numpy(query_states)
        captured[(self.layer_idx, "k_post_rope")] = to_fp32_numpy(key_states)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # Inline eager_attention_forward with capture points at scores + softmax.
        key_states_repeated = repeat_kv(key_states, self.num_key_value_groups)
        value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)

        attn_scores = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) * self.scaling

        # Capture #3: pre-mask, pre-softmax scores.
        captured[(self.layer_idx, "attn_scores")] = to_fp32_numpy(attn_scores)

        attn_weights_masked = attn_scores
        if attention_mask is not None:
            attn_weights_masked = attn_weights_masked + attention_mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights_masked, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # Capture #4: post-softmax weights (pre-V multiply).
        captured[(self.layer_idx, "attn_softmax")] = to_fp32_numpy(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states_repeated)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    import types

    for layer_idx in target_layer_idxs:
        attn_module = model.model.layers[layer_idx].self_attn
        attn_module.forward = types.MethodType(traced_forward, attn_module)


def main() -> int:
    args = parse_args()
    layers_to_capture = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"=== SHIP-007 HF FP16 reference stage generation ===")
    print(f"  model:   {args.model}")
    print(f"  prompt:  {args.prompt!r}")
    print(f"  dtype:   {dtype}")
    print(f"  device:  {args.device}")
    print(f"  layers:  {layers_to_capture}")
    print(f"  output:  {args.output}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading model from HF cache (no download — should be instant)...")
    # Force eager attention when capturing sub-stages: sdpa/flash-attn fast paths
    # don't expose pre-softmax scores or post-softmax weights as captureable
    # intermediates. Eager path is the only one whose attn_scores/attn_softmax
    # we can intercept via monkeypatch.
    attn_impl = "eager" if args.with_attn_substages else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": args.device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    print("Tokenizing prompt (no chat template — raw prompt to match apr trace --save-tensor default)...")
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False, return_tensors="pt").to(args.device)
    tokens_list = input_ids[0].tolist()
    print(f"  tokens ({len(tokens_list)}): {tokens_list}")
    print()

    # Storage for stage activations captured via forward hooks.
    # Map: (layer_idx_or_whole_model_sentinel, stage_name) -> np.ndarray fp32
    captured: dict[tuple[int, str], np.ndarray] = {}

    # Per-layer stages we can capture by attaching hooks on existing modules.
    # Each tuple: (stage_name, hook_callable_factory(layer_idx))
    def make_layer_hooks(layer_idx: int) -> list[tuple[str, str, callable]]:
        """Return [(stage_name, where, hook_fn), ...] for the per-layer stages we capture.

        `where` is the module path within the decoder layer (module.named_modules() name)."""
        layer_module = model.model.layers[layer_idx]

        def hook_save(stage_name: str):
            def hook(module, args_, output):
                # Some modules return a tensor directly, others (attn) return a tuple.
                tensor = output[0] if isinstance(output, tuple) else output
                captured[(layer_idx, stage_name)] = to_fp32_numpy(tensor)
            return hook

        # For ffn_silu and ffn_swigl we don't have direct module outputs — the SwiGLU
        # composition `silu(gate_proj(x)) * up_proj(x)` is fused in the MLP forward.
        # We hook gate_proj and up_proj and assemble silu/swigl from those buffers.
        gate_buf: list[torch.Tensor] = []
        up_buf: list[torch.Tensor] = []

        def hook_gate(module, args_, output):
            captured[(layer_idx, "ffn_gate")] = to_fp32_numpy(output)
            gate_buf.append(output)
            silu_out = torch.nn.functional.silu(output)
            captured[(layer_idx, "ffn_silu")] = to_fp32_numpy(silu_out)
            # Compose ffn_swigl when up is also available
            if up_buf:
                captured[(layer_idx, "ffn_swigl")] = to_fp32_numpy(silu_out * up_buf[-1])

        def hook_up(module, args_, output):
            captured[(layer_idx, "ffn_up")] = to_fp32_numpy(output)
            up_buf.append(output)
            if gate_buf:
                silu_g = torch.nn.functional.silu(gate_buf[-1])
                captured[(layer_idx, "ffn_swigl")] = to_fp32_numpy(silu_g * output)

        # QKV captures: HF stores Q/K/V as separate Linears (q_proj, k_proj, v_proj)
        # while APR fuses them into a single qkv_weight matmul. To compare apples
        # to apples we concat HF's three outputs along the last dim AND derive the
        # pre-bias version (qkv_matmul) by subtracting biases.
        # APR's `qkv_matmul` stage = pre-bias matmul output;
        # APR's `qkv_bias` stage   = post-bias.
        # HF's Linear.forward = x @ W^T + b → output IS post-bias; subtract bias to
        # get pre-bias. Per-Linear bias broadcasts across the batch+seq dims.
        q_buf: list[torch.Tensor] = []
        k_buf: list[torch.Tensor] = []
        v_buf: list[torch.Tensor] = []

        def make_qkv_hook(slot: list, bias: torch.Tensor | None):
            def hook(module, args_, output):
                slot.append(output)
                if len(q_buf) and len(k_buf) and len(v_buf):
                    # Both qkv_matmul (pre-bias) and qkv_bias (post-bias)
                    # have the same shape: [batch, seq, q_dim + 2*kv_dim].
                    qkv_post = torch.cat([q_buf[-1], k_buf[-1], v_buf[-1]], dim=-1)
                    captured[(layer_idx, "qkv_bias")] = to_fp32_numpy(qkv_post)
                    # Pre-bias: subtract each Linear's bias (or zero if none).
                    q_b = layer_module.self_attn.q_proj.bias
                    k_b = layer_module.self_attn.k_proj.bias
                    v_b = layer_module.self_attn.v_proj.bias
                    qkv_pre = torch.cat([
                        q_buf[-1] - (q_b if q_b is not None else 0.0),
                        k_buf[-1] - (k_b if k_b is not None else 0.0),
                        v_buf[-1] - (v_b if v_b is not None else 0.0),
                    ], dim=-1)
                    captured[(layer_idx, "qkv_matmul")] = to_fp32_numpy(qkv_pre)
            return hook

        # `attention` stage = INPUT to o_proj (after softmax(Q@Kᵀ/√d)@V, pre-O-proj).
        # Use a forward-pre-hook on o_proj — its `args_[0]` is the raw input tensor.
        def hook_o_proj_pre(module, args_):
            # forward_pre_hook signature: (module, args) where args is a tuple.
            # The input to o_proj is args[0].
            inp = args_[0]
            captured[(layer_idx, "attention")] = to_fp32_numpy(inp)

        return [
            ("attn_norm", layer_module.input_layernorm, hook_save("attn_norm"), False),
            ("q_proj", layer_module.self_attn.q_proj, make_qkv_hook(q_buf, None), False),
            ("k_proj", layer_module.self_attn.k_proj, make_qkv_hook(k_buf, None), False),
            ("v_proj", layer_module.self_attn.v_proj, make_qkv_hook(v_buf, None), False),
            ("attention_pre_o_proj", layer_module.self_attn.o_proj, hook_o_proj_pre, True),
            ("attn_out", layer_module.self_attn, hook_save("attn_out"), False),
            ("ffn_norm", layer_module.post_attention_layernorm, hook_save("ffn_norm"), False),
            ("ffn_gate", layer_module.mlp.gate_proj, hook_gate, False),
            ("ffn_up", layer_module.mlp.up_proj, hook_up, False),
            ("ffn_out", layer_module.mlp.down_proj, hook_save("ffn_out"), False),
        ]

    # Whole-model hooks
    embedding_buf: list[torch.Tensor] = []

    def hook_embedding(module, args_, output):
        captured[(0, "embedding")] = to_fp32_numpy(output)
        embedding_buf.append(output)

    def hook_final_norm(module, args_, output):
        captured[(WHOLE_MODEL_LAYER, "final_norm")] = to_fp32_numpy(output)

    def hook_lm_head(module, args_, output):
        # lm_head output shape: [batch, seq_len, vocab_size]
        # APR captures last-token logits only; mirror that to keep the dim_product match
        last_token = output[0, -1, :]
        captured[(WHOLE_MODEL_LAYER, "lm_head")] = to_fp32_numpy(last_token)

    print("Registering forward hooks...")
    handles = []
    handles.append(model.model.embed_tokens.register_forward_hook(hook_embedding))
    handles.append(model.model.norm.register_forward_hook(hook_final_norm))
    handles.append(model.lm_head.register_forward_hook(hook_lm_head))

    for layer_idx in layers_to_capture:
        for _name, mod, hook, is_pre in make_layer_hooks(layer_idx):
            if is_pre:
                handles.append(mod.register_forward_pre_hook(hook))
            else:
                handles.append(mod.register_forward_hook(hook))

    # post_attn_residual + post_ffn_residual: capture by hooking each layer's forward
    # OUTPUT — Qwen2DecoderLayer returns (hidden_states, ...) and hidden_states is
    # the post_ffn_residual. To get post_attn_residual we'd need a manual instrumentation
    # of the layer (not exposed). Instead we approximate post_ffn_residual via the layer
    # output, and skip post_attn_residual at PARTIAL coverage.
    def make_layer_output_hook(layer_idx: int):
        def hook(module, args_, output):
            # Qwen2DecoderLayer returns (hidden_states, ...) — index [0]
            hs = output[0] if isinstance(output, tuple) else output
            captured[(layer_idx, "post_ffn_residual")] = to_fp32_numpy(hs)
        return hook

    for layer_idx in layers_to_capture:
        handles.append(model.model.layers[layer_idx].register_forward_hook(make_layer_output_hook(layer_idx)))

    print(f"  registered {len(handles)} hooks")

    if args.with_attn_substages:
        print(f"Installing Qwen2Attention.forward monkeypatch on layers {layers_to_capture}...")
        install_attn_substages_patch(model, layers_to_capture, captured)
        print(f"  patched {len(layers_to_capture)} layer(s) for q_post_rope/k_post_rope/attn_scores/attn_softmax")
    print()

    print("Running forward pass (no_grad)...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False, output_hidden_states=False)

    for h in handles:
        h.remove()

    print(f"Captured {len(captured)} stage tensors. Writing APRT files...")
    written: list[Path] = []
    for (layer, stage), values in sorted(captured.items()):
        if layer == WHOLE_MODEL_LAYER:
            path = args.output / f"{stage}.bin"
        else:
            path = args.output / f"layer-{layer}" / f"{stage}.bin"
        write_aprt_file(path, layer, values)
        written.append(path)

    print()
    for p in written:
        sz = os.path.getsize(p)
        print(f"  {p} ({sz:,} bytes)")

    # Smoke-check: lm_head argmax should produce a sensible token
    lm_head_logits = captured[(WHOLE_MODEL_LAYER, "lm_head")]
    argmax_token = int(np.argmax(lm_head_logits))
    argmax_text = tokenizer.decode([argmax_token])
    print()
    print(f"Smoke check (lm_head argmax):")
    print(f"  vocab_size:    {lm_head_logits.shape[0]}")
    print(f"  argmax_token:  {argmax_token}")
    print(f"  argmax_text:   {argmax_text!r}")

    # Provenance file alongside the tensors
    provenance_path = args.output / "_PROVENANCE.txt"
    with open(provenance_path, "w") as f:
        f.write(f"# SHIP-007 HF FP16 reference stages\n")
        f.write(f"model:                 {args.model}\n")
        f.write(f"prompt:                {args.prompt!r}\n")
        f.write(f"tokens:                {tokens_list}\n")
        f.write(f"dtype:                 {dtype}\n")
        f.write(f"device:                {args.device}\n")
        f.write(f"layers_captured:       {layers_to_capture}\n")
        base_stages = [
            "embedding", "attn_norm", "qkv_matmul", "qkv_bias", "attention",
            "attn_out", "post_ffn_residual", "ffn_norm", "ffn_gate", "ffn_up",
            "ffn_silu", "ffn_swigl", "ffn_out",
        ]
        substage_stages = ["q_post_rope", "k_post_rope", "attn_scores", "attn_softmax"]
        if args.with_attn_substages:
            stages_per_layer_list = base_stages + substage_stages
        else:
            stages_per_layer_list = base_stages
        f.write(f"stages_per_layer:      {stages_per_layer_list}\n")
        f.write(f"stages_whole_model:    [final_norm, lm_head]\n")
        f.write(f"with_attn_substages:   {args.with_attn_substages}\n")
        f.write(f"vocab_size:            {lm_head_logits.shape[0]}\n")
        f.write(f"argmax_token:          {argmax_token}\n")
        f.write(f"argmax_text:           {argmax_text!r}\n")
        f.write(f"git_sha:               {get_git_sha()}\n")
        f.write(f"transformers_version:  {__import__('transformers').__version__}\n")
        f.write(f"torch_version:         {torch.__version__}\n")
        f.write(f"generated_utc:         {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
    print(f"\nWrote provenance to {provenance_path}")
    print(f"\n=== Done. {len(written)} APRT stage files in {args.output} ===")
    print("Next: `apr diff --values <apr>.bin <hf>.bin` to find first stage where cos < 0.999")
    return 0


if __name__ == "__main__":
    sys.exit(main())
