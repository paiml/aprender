#!/usr/bin/env python3
"""
Golden Comparison: Layer 0 Forward Pass from transformers
for comparison with realizar implementation.

Bisects: RMSNorm -> Attention -> FFN

Usage:
    uv run --with torch --with transformers --with numpy scripts/reference_layer0.py
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Values from realizar compare_layer0 example (token 791)
REALIZAR_VALUES = {
    "embedding_first5": [-0.028451443, 0.012253761, 0.012253761, -0.022636414, -0.04589653],
    "rmsnorm_first3": [-0.9617414, 0.34859487, 0.4006475],
    "q_first3": [-0.2551602, -0.68396235, -0.046994388],
    "k_first5": [0.37233883, -0.64664435, 1.4780979, -2.3014083, -0.75090396],
    "v_first5": [-0.11200829, -0.06673667, 0.21031746, 0.13783953, -0.5877351],
    "attn_out_first3": [-0.11200829, -0.06673667, 0.21031746],
    "out_proj_first3": [-0.9596884, 0.01161759, -0.05963149],
    "residual1_first3": [-0.98813987, 0.023871351, -0.047377728],
    "ffn_gate_first3": [-0.6621403, -0.15498732, -1.122485],
    "ffn_up_first3": [0.58142006, 0.092911035, 0.32663447],
    "swiglu_first3": [-0.13099347, -0.006643175, -0.09002925],
    "ffn_down_first3": [-0.029813778, 0.17109168, 0.08764197],
    "layer_output_first3": [-1.0179536, 0.19496302, 0.04026424],
}

def rms_norm(x, weight, eps=1e-6):
    """RMSNorm as implemented in Qwen2"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x

def compare(name, ref_values, test_values, tol=0.01):
    """Compare values and report match/divergence"""
    ref = np.array(ref_values)
    test = np.array(test_values[:len(ref_values)])
    diff = np.abs(ref - test)
    max_diff = np.max(diff)
    match = max_diff < tol
    status = "MATCH" if match else "DIVERGE"
    print(f"  {name}: {status} (max_diff={max_diff:.6f})")
    if not match:
        print(f"    Realizar: {ref_values}")
        print(f"    Transf:   {test_values[:len(ref_values)]}")
    return match

def main():
    print(f"=== Golden Comparison: Layer 0 Forward Pass ===")
    print(f"Model: {MODEL_NAME}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Get components
    embed_layer = model.model.embed_tokens
    layer0 = model.model.layers[0]

    print(f"Hidden dim: {model.config.hidden_size}")
    print(f"Num heads: {model.config.num_attention_heads}")
    print(f"Num KV heads: {model.config.num_key_value_heads}")
    print(f"Head dim: {model.config.hidden_size // model.config.num_attention_heads}")

    # Token 791 (same as realizar example)
    token_id = 791
    print(f"\nToken ID: {token_id}")
    print(f"Token string: '{tokenizer.decode([token_id])}'")

    with torch.no_grad():
        # Step 1: Embedding lookup
        embedding = embed_layer.weight[token_id].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        emb_np = embedding.squeeze().numpy()
        print(f"\n=== Step 1: Embedding ===")
        print(f"Embedding first 5: {emb_np[:5].tolist()}")
        compare("embedding", REALIZAR_VALUES["embedding_first5"], emb_np[:5].tolist())

        # Step 2: RMSNorm (input_layernorm)
        # First dump the weight values
        norm_weight = layer0.input_layernorm.weight.detach().cpu().numpy()
        print(f"\n=== Step 2: RMSNorm ===")
        print(f"Norm weight first 5: {norm_weight[:5].tolist()}")
        print(f"Realizar norm weight: [0.6411133, 0.5395508, 0.6201172, 0.80322266, 0.6279297]")

        # Check epsilon
        eps = layer0.input_layernorm.variance_epsilon
        print(f"Epsilon: {eps}")

        # Manual RMSNorm computation to verify
        emb_np = embedding.squeeze().numpy()
        mean_sq = np.mean(emb_np ** 2)
        inv_rms = 1.0 / np.sqrt(mean_sq + eps)
        manual_normed = emb_np * inv_rms * norm_weight
        print(f"Manual RMSNorm first 3: {manual_normed[:3].tolist()}")
        print(f"  inv_rms = {inv_rms:.8f}")
        print(f"  mean_sq = {mean_sq:.8f}")

        normed = layer0.input_layernorm(embedding)
        normed_np = normed.squeeze().numpy()
        print(f"Layer RMSNorm first 3: {normed_np[:3].tolist()}")
        compare("rmsnorm", REALIZAR_VALUES["rmsnorm_first3"], normed_np[:3].tolist())

        # Step 3: Q, K, V projections
        # Qwen2 uses separate Q, K, V projections
        q_proj = layer0.self_attn.q_proj
        k_proj = layer0.self_attn.k_proj
        v_proj = layer0.self_attn.v_proj

        q = q_proj(normed).squeeze()
        k = k_proj(normed).squeeze()
        v = v_proj(normed).squeeze()

        print(f"\n=== Step 3: Q/K/V Projections ===")
        print(f"Q first 3: {q[:3].numpy().tolist()}")
        print(f"K first 5: {k[:5].numpy().tolist()}")
        print(f"V first 5: {v[:5].numpy().tolist()}")
        compare("Q", REALIZAR_VALUES["q_first3"], q[:3].numpy().tolist())
        compare("K", REALIZAR_VALUES["k_first5"], k[:5].numpy().tolist())
        compare("V", REALIZAR_VALUES["v_first5"], v[:5].numpy().tolist())

        # Step 4: For single token, attention output = V (softmax of single element = 1.0)
        # But we need to apply RoPE first and handle GQA
        num_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim = model.config.hidden_size // num_heads
        group_size = num_heads // num_kv_heads

        # Reshape for multi-head
        q_heads = q.view(num_heads, head_dim)  # [12, 128]
        k_heads = k.view(num_kv_heads, head_dim)  # [2, 128]
        v_heads = v.view(num_kv_heads, head_dim)  # [2, 128]

        # Single token attention: output is just V (no actual attention computation needed)
        # With GQA, each Q head attends to its corresponding KV head
        attn_out_heads = []
        for h in range(num_heads):
            kv_head = h // group_size
            attn_out_heads.append(v_heads[kv_head])
        attn_out = torch.stack(attn_out_heads).view(-1)  # [1536]

        print(f"\n=== Step 4: Attention Output (single token = V) ===")
        print(f"Attn out first 3: {attn_out[:3].numpy().tolist()}")
        compare("attn_out", REALIZAR_VALUES["attn_out_first3"], attn_out[:3].numpy().tolist())

        # Step 5: Output projection
        o_proj = layer0.self_attn.o_proj
        out_proj = o_proj(attn_out.unsqueeze(0)).squeeze()

        print(f"\n=== Step 5: Output Projection ===")
        print(f"Out proj first 3: {out_proj[:3].numpy().tolist()}")
        compare("out_proj", REALIZAR_VALUES["out_proj_first3"], out_proj[:3].numpy().tolist())

        # Step 6: Residual connection 1
        residual1 = embedding.squeeze() + out_proj

        print(f"\n=== Step 6: Residual 1 ===")
        print(f"Residual1 first 3: {residual1[:3].numpy().tolist()}")
        compare("residual1", REALIZAR_VALUES["residual1_first3"], residual1[:3].numpy().tolist())

        # Step 7: FFN RMSNorm (post_attention_layernorm)
        ffn_normed = layer0.post_attention_layernorm(residual1.unsqueeze(0).unsqueeze(0))
        ffn_normed = ffn_normed.squeeze()

        # Step 8: FFN (MLP with SwiGLU)
        # Qwen2MLP: gate_proj, up_proj, down_proj with SiLU activation
        gate = layer0.mlp.gate_proj(ffn_normed)
        up = layer0.mlp.up_proj(ffn_normed)

        print(f"\n=== Step 8: FFN Gate/Up ===")
        print(f"FFN gate first 3: {gate[:3].numpy().tolist()}")
        print(f"FFN up first 3: {up[:3].numpy().tolist()}")
        compare("ffn_gate", REALIZAR_VALUES["ffn_gate_first3"], gate[:3].numpy().tolist())
        compare("ffn_up", REALIZAR_VALUES["ffn_up_first3"], up[:3].numpy().tolist())

        # SwiGLU: silu(gate) * up
        swiglu = torch.nn.functional.silu(gate) * up

        print(f"\n=== Step 9: SwiGLU ===")
        print(f"SwiGLU first 3: {swiglu[:3].numpy().tolist()}")
        compare("swiglu", REALIZAR_VALUES["swiglu_first3"], swiglu[:3].numpy().tolist())

        # FFN down projection
        ffn_down = layer0.mlp.down_proj(swiglu)

        print(f"\n=== Step 10: FFN Down ===")
        print(f"FFN down first 3: {ffn_down[:3].numpy().tolist()}")
        compare("ffn_down", REALIZAR_VALUES["ffn_down_first3"], ffn_down[:3].numpy().tolist())

        # Step 11: Residual connection 2 (final layer output)
        layer_output = residual1 + ffn_down

        print(f"\n=== Step 11: Layer 0 Output ===")
        print(f"Layer output first 3: {layer_output[:3].numpy().tolist()}")
        compare("layer_output", REALIZAR_VALUES["layer_output_first3"], layer_output[:3].numpy().tolist())

        # Summary
        print(f"\n=== SUMMARY ===")
        print("If all steps MATCH: realizar Layer 0 is correct, bug is downstream.")
        print("If a step DIVERGES: that's the bug location.")

if __name__ == "__main__":
    main()
