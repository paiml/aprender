#!/usr/bin/env python3
"""
PMAT-COR-002: Golden Vector for Layer 0 Output
Generates reference Layer 0 output for "2+2=" prompt.

Usage:
    uv run --with torch --with transformers --with numpy scripts/golden_layer0_2plus2.py
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
PROMPT = "2+2="  # Raw prompt (will also test with chat template)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("=== PMAT-COR-002: Golden Vector Generation ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Tokenize prompt (raw, no chat template first)
    tokens_raw = tokenizer.encode(PROMPT, add_special_tokens=False)
    print(f"Raw tokens: {tokens_raw}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in tokens_raw]}")

    # Also show chat template version
    chat_prompt = f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    tokens_chat = tokenizer.encode(chat_prompt, add_special_tokens=False)
    print(f"\nChat template: {chat_prompt!r}")
    print(f"Chat tokens: {tokens_chat}")
    print(f"Chat decoded: {[tokenizer.decode([t]) for t in tokens_chat]}")

    # Get model components
    embed_layer = model.model.embed_tokens
    layer0 = model.model.layers[0]
    config = model.config

    print(f"\nModel config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.hidden_size // config.num_attention_heads}")

    # Process first token of raw prompt for Golden Vector
    first_token = tokens_raw[0]
    print(f"\n=== Golden Vector for Token {first_token} ('{tokenizer.decode([first_token])}') ===")

    with torch.no_grad():
        # Get embedding
        embedding = embed_layer.weight[first_token].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        print(f"\nEmbedding:")
        print(f"  Shape: {embedding.shape}")
        print(f"  First 5: {embedding.squeeze()[:5].tolist()}")
        print(f"  Sum: {embedding.sum().item():.6f}")

        # Full Layer 0 forward pass using transformers
        # Need to set up proper attention mask and position ids
        position_ids = torch.tensor([[0]])
        attention_mask = torch.ones((1, 1), dtype=torch.float32)  # Must be float for SDPA

        # Get rotary embeddings from the model
        rotary_emb = model.model.rotary_emb
        cos, sin = rotary_emb(embedding, position_ids)

        # Run layer 0 forward
        hidden_states = embedding
        layer_output = layer0(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            use_cache=False,
        )[0]

        layer0_output = layer_output.squeeze().numpy()
        print(f"\n=== GOLDEN VECTOR: Layer 0 Output ===")
        print(f"  Shape: {layer0_output.shape}")
        print(f"  First 10: {layer0_output[:10].tolist()}")
        print(f"  Last 5: {layer0_output[-5:].tolist()}")
        print(f"  Sum: {layer0_output.sum():.6f}")
        print(f"  Mean: {layer0_output.mean():.6f}")
        print(f"  Std: {layer0_output.std():.6f}")
        print(f"  Min: {layer0_output.min():.6f}")
        print(f"  Max: {layer0_output.max():.6f}")

        # Save full vector for comparison
        np.save("/tmp/golden_layer0_output.npy", layer0_output)
        print(f"\n  Saved to: /tmp/golden_layer0_output.npy")

        # Also output in a format that can be easily pasted
        print(f"\n=== GOLDEN VECTOR (first 20 for quick comparison) ===")
        golden_first20 = layer0_output[:20].tolist()
        print(f"GOLDEN_LAYER0_FIRST20 = {golden_first20}")

        # Now let's also trace intermediate steps
        print(f"\n=== Intermediate Steps (for bisection) ===")

        # Step 1: Input RMSNorm
        normed = layer0.input_layernorm(embedding)
        print(f"\n1. Input RMSNorm:")
        print(f"   First 5: {normed.squeeze()[:5].tolist()}")

        # Step 2: Q, K, V projections
        q = layer0.self_attn.q_proj(normed).squeeze()
        k = layer0.self_attn.k_proj(normed).squeeze()
        v = layer0.self_attn.v_proj(normed).squeeze()
        print(f"\n2. Q projection (with bias):")
        print(f"   First 5: {q[:5].tolist()}")
        print(f"   Sum: {q.sum().item():.6f}")
        print(f"\n   K projection (with bias):")
        print(f"   First 5: {k[:5].tolist()}")
        print(f"   Sum: {k.sum().item():.6f}")
        print(f"\n   V projection (with bias):")
        print(f"   First 5: {v[:5].tolist()}")
        print(f"   Sum: {v.sum().item():.6f}")

        # Check bias values
        print(f"\n   Q bias first 5: {layer0.self_attn.q_proj.bias[:5].tolist()}")
        print(f"   K bias first 5: {layer0.self_attn.k_proj.bias[:5].tolist()}")
        print(f"   V bias first 5: {layer0.self_attn.v_proj.bias[:5].tolist()}")

        # Step 3: Attention (single token = softmax(1) = 1, so output = V mapped through GQA)
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads
        group_size = num_heads // num_kv_heads

        v_heads = v.view(num_kv_heads, head_dim)
        attn_out_heads = []
        for h in range(num_heads):
            kv_head = h // group_size
            attn_out_heads.append(v_heads[kv_head])
        attn_out = torch.stack(attn_out_heads).view(-1)

        print(f"\n3. Attention output (V through GQA):")
        print(f"   First 5: {attn_out[:5].tolist()}")
        print(f"   Sum: {attn_out.sum().item():.6f}")

        # Step 4: O projection
        o_out = layer0.self_attn.o_proj(attn_out.unsqueeze(0)).squeeze()
        print(f"\n4. O projection:")
        print(f"   First 5: {o_out[:5].tolist()}")
        print(f"   Sum: {o_out.sum().item():.6f}")

        # Step 5: Residual 1
        residual1 = embedding.squeeze() + o_out
        print(f"\n5. Residual 1 (embedding + o_out):")
        print(f"   First 5: {residual1[:5].tolist()}")
        print(f"   Sum: {residual1.sum().item():.6f}")

        # Step 6: FFN RMSNorm
        ffn_normed = layer0.post_attention_layernorm(residual1.unsqueeze(0).unsqueeze(0)).squeeze()
        print(f"\n6. FFN RMSNorm:")
        print(f"   First 5: {ffn_normed[:5].tolist()}")

        # Step 7: FFN (gate, up, SwiGLU, down)
        gate = layer0.mlp.gate_proj(ffn_normed)
        up = layer0.mlp.up_proj(ffn_normed)
        swiglu = torch.nn.functional.silu(gate) * up
        ffn_out = layer0.mlp.down_proj(swiglu)

        print(f"\n7. FFN:")
        print(f"   Gate first 5: {gate[:5].tolist()}")
        print(f"   Up first 5: {up[:5].tolist()}")
        print(f"   SwiGLU first 5: {swiglu[:5].tolist()}")
        print(f"   FFN out first 5: {ffn_out[:5].tolist()}")
        print(f"   FFN out sum: {ffn_out.sum().item():.6f}")

        # Step 8: Residual 2 = Layer output
        layer_out_manual = residual1 + ffn_out
        print(f"\n8. Layer output (residual1 + ffn_out):")
        print(f"   First 5: {layer_out_manual[:5].tolist()}")
        print(f"   Sum: {layer_out_manual.sum().item():.6f}")

        # Verify matches transformers output
        cos_sim = cosine_similarity(layer0_output, layer_out_manual.numpy())
        print(f"\n   Cosine similarity with transformers: {cos_sim:.8f}")

        # Final summary
        print(f"\n" + "="*60)
        print(f"GOLDEN VECTOR SUMMARY (Token {first_token} = '{tokenizer.decode([first_token])}')")
        print(f"="*60)
        print(f"First 10: {layer0_output[:10].tolist()}")
        print(f"Statistics: sum={layer0_output.sum():.4f}, mean={layer0_output.mean():.4f}, std={layer0_output.std():.4f}")
        print(f"\nTo compare with apr trace:")
        print(f"  1. Run: apr run ... --trace=transformer_block --trace-layers=0 --trace-verbose")
        print(f"  2. Extract Layer 0 output vector")
        print(f"  3. Compute cosine similarity (target > 0.99)")

if __name__ == "__main__":
    main()
