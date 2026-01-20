#!/usr/bin/env python3
"""
Golden Comparison: Get reference embedding from transformers
for comparison with realizar implementation.

Usage:
    python reference_embedding.py

Requires:
    pip install transformers torch safetensors
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def main():
    print(f"=== Golden Comparison: Transformers Reference ===")
    print(f"Model: {MODEL_NAME}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Tokenize "def"
    print("\n=== Tokenization of 'def' ===")
    tokens = tokenizer.encode("def", add_special_tokens=False)
    print(f"Token count: {len(tokens)}")
    for i, tok in enumerate(tokens):
        tok_str = tokenizer.decode([tok])
        print(f"  [{i}] token_id={tok}, string='{tok_str}'")

    # Additional tokenization tests
    for test_str in ["def", "def ", " def", "def fibonacci"]:
        toks = tokenizer.encode(test_str, add_special_tokens=False)
        print(f"'{test_str}' -> {toks}")

    # Load model to get embeddings
    print("\nLoading model (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use f32 for exact comparison
        trust_remote_code=True,
    )

    # Get embedding layer
    embed_layer = model.model.embed_tokens

    print(f"\n=== Embedding Layer ===")
    print(f"  shape: {embed_layer.weight.shape}")
    print(f"  dtype: {embed_layer.weight.dtype}")

    # Get embedding for token 750 (should be "def")
    token_id = 750
    emb = embed_layer.weight[token_id].detach().cpu().numpy()

    print(f"\n=== Embedding for token {token_id} ===")
    print("First 10 values:")
    for i, v in enumerate(emb[:10]):
        print(f"  [{i:4}] = {v:.8f}")

    # Stats
    import numpy as np
    print(f"\nStats:")
    print(f"  sum:  {np.sum(emb):.8f}")
    print(f"  norm: {np.linalg.norm(emb):.8f}")
    print(f"  min:  {np.min(emb):.8f}")
    print(f"  max:  {np.max(emb):.8f}")

    # Last 10 values
    print("\nLast 10 values:")
    for i in range(len(emb) - 10, len(emb)):
        print(f"  [{i:4}] = {emb[i]:.8f}")

    # JSON output for comparison
    print(f"\n=== JSON (first 10) for comparison ===")
    print("[" + ", ".join(f"{v:.8f}" for v in emb[:10]) + "]")

    # Reference tokens
    print(f"\n=== Reference tokens (first 4 values each) ===")
    for tid in [0, 1, 100, 151644]:
        e = embed_layer.weight[tid].detach().cpu().numpy()
        print(f"Token {tid:6}: [{e[0]:.6f}, {e[1]:.6f}, {e[2]:.6f}, {e[3]:.6f}]")

    # Now compare with realizar values
    realizar_first_10 = [
        0.01059240, -0.01039481, -0.01039481, 0.02458388, 0.02458388,
        -0.00339907, -0.03837776, -0.01039481, -0.00339907, -0.00339907
    ]

    print(f"\n=== Comparison with realizar ===")
    print("Realizar first 10:", realizar_first_10)
    print("Transform first 10:", [f"{v:.8f}" for v in emb[:10]])

    # Calculate difference
    diff = np.array(realizar_first_10) - emb[:10]
    print(f"\nDifference (realizar - transformers):")
    for i, d in enumerate(diff):
        print(f"  [{i}] = {d:.10f}")

    max_diff = np.max(np.abs(diff))
    print(f"\nMax absolute difference: {max_diff:.10e}")

    # Cosine similarity
    cos_sim = np.dot(realizar_first_10, emb[:10]) / (
        np.linalg.norm(realizar_first_10) * np.linalg.norm(emb[:10])
    )
    print(f"Cosine similarity (first 10): {cos_sim:.10f}")

    if max_diff < 1e-5:
        print("\n[PASS] Embeddings match within tolerance!")
    else:
        print("\n[FAIL] Embeddings DIVERGE!")

if __name__ == "__main__":
    main()
