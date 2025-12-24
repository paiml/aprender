#!/usr/bin/env python3
"""
Generate Golden Traces for Qwen2 Model Verification

This script generates PyTorch baseline outputs for verifying the Rust implementation
against the reference HuggingFace model. It produces tensor statistics at each layer
that can be compared with aprender's Qwen2Model output.

Usage:
    python scripts/generate_golden_traces.py --model Qwen/Qwen2-0.5B-Instruct \
        --output golden/qwen2-0.5b.json

Requirements:
    pip install torch transformers numpy

References:
    - Spec §C1: Golden Trace Verification (logits deviate > 1e-4 from PyTorch reference)
    - Spec §I14: Golden Trace Match (any tensor diff > tolerance)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Check for required dependencies
try:
    import torch
    import numpy as np
except ImportError:
    print("Error: Required packages not found.")
    print("Install with: pip install torch transformers numpy")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers package not found.")
    print("Install with: pip install transformers")
    sys.exit(1)


@dataclass
class TensorStats:
    """Statistics for a tensor at a specific layer."""
    name: str
    shape: list[int]
    dtype: str
    mean: float
    std: float
    min_val: float
    max_val: float
    abs_mean: float
    has_nan: bool
    has_inf: bool
    sample_values: list[float]  # First 10 values for spot-checking


def compute_tensor_stats(name: str, tensor: torch.Tensor) -> TensorStats:
    """Compute statistics for a tensor."""
    t = tensor.detach().float()
    flat = t.flatten()

    return TensorStats(
        name=name,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        mean=float(t.mean().item()),
        std=float(t.std().item()),
        min_val=float(t.min().item()),
        max_val=float(t.max().item()),
        abs_mean=float(t.abs().mean().item()),
        has_nan=bool(torch.isnan(t).any().item()),
        has_inf=bool(torch.isinf(t).any().item()),
        sample_values=[float(x) for x in flat[:10].tolist()],
    )


def generate_golden_trace(
    model_name: str,
    test_inputs: list[str],
    device: str = "cpu",
    max_new_tokens: int = 10,
) -> dict:
    """
    Generate golden traces from HuggingFace model.

    Returns a dictionary containing:
    - config: Model configuration
    - inputs: Test input information
    - layer_stats: Per-layer tensor statistics
    - logits: Final logit statistics
    - generations: Generated text for each input
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for maximum precision
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Collect model config
    config = {
        "model_name": model_name,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
        "num_hidden_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size,
        "intermediate_size": model.config.intermediate_size,
        "rope_theta": getattr(model.config, "rope_theta", 10000.0),
    }

    traces = []

    for test_input in test_inputs:
        print(f"Processing: {test_input[:50]}...")

        # Tokenize
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Hook to capture intermediate activations
        layer_activations = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                layer_activations[name] = compute_tensor_stats(name, out)
            return hook

        # Register hooks for key layers
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"layer.{i}.attention")))
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"layer.{i}.mlp")))
            hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"layer.{i}.input_norm")))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook(f"layer.{i}.post_attn_norm")))

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True, output_attentions=True)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Collect logits statistics
        logits = outputs.logits
        logit_stats = compute_tensor_stats("logits", logits)

        # Collect embedding statistics
        embed_stats = compute_tensor_stats("embedding", outputs.hidden_states[0])

        # Final hidden state
        final_hidden = compute_tensor_stats("final_hidden", outputs.hidden_states[-1])

        # Generate some tokens
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        # Convert layer_activations to serializable format
        layer_stats_serializable = {
            name: {
                "shape": stats.shape,
                "dtype": stats.dtype,
                "mean": stats.mean,
                "std": stats.std,
                "min": stats.min_val,
                "max": stats.max_val,
                "abs_mean": stats.abs_mean,
                "has_nan": stats.has_nan,
                "has_inf": stats.has_inf,
                "sample_values": stats.sample_values,
            }
            for name, stats in layer_activations.items()
        }

        trace = {
            "input": test_input,
            "input_ids": input_ids[0].tolist(),
            "num_tokens": int(input_ids.shape[1]),
            "embedding": {
                "shape": embed_stats.shape,
                "mean": embed_stats.mean,
                "std": embed_stats.std,
            },
            "layer_stats": layer_stats_serializable,
            "final_hidden": {
                "shape": final_hidden.shape,
                "mean": final_hidden.mean,
                "std": final_hidden.std,
            },
            "logits": {
                "shape": logit_stats.shape,
                "mean": logit_stats.mean,
                "std": logit_stats.std,
                "min": logit_stats.min_val,
                "max": logit_stats.max_val,
                "sample_values": logit_stats.sample_values,
            },
            "generated_text": generated_text,
            "generated_ids": generated[0].tolist(),
        }
        traces.append(trace)

    return {
        "version": "1.0.0",
        "generator": "generate_golden_traces.py",
        "config": config,
        "traces": traces,
        "tolerance": {
            "logit_abs": 1e-4,  # Spec C1: > 1e-4 deviation fails
            "stat_percent": 1.0,  # Spec A2: > 1% deviation fails
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden traces for Qwen2 model verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate traces for Qwen2-0.5B
    python generate_golden_traces.py -m Qwen/Qwen2-0.5B-Instruct -o golden/qwen2.json

    # Use GPU
    python generate_golden_traces.py -m Qwen/Qwen2-0.5B-Instruct -o golden/qwen2.json --device cuda

    # Custom test inputs
    python generate_golden_traces.py -m Qwen/Qwen2-0.5B-Instruct -o golden/qwen2.json \\
        --inputs "What is 2+2?" "Explain Rust in one sentence."
""",
    )
    parser.add_argument(
        "-m", "--model",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2-0.5B-Instruct)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("golden/qwen2-0.5b.json"),
        help="Output file path (default: golden/qwen2-0.5b.json)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain machine learning in one sentence.",
            "Write a haiku about Rust programming.",
            "2 + 2 = ",
        ],
        help="Test input strings",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate per input (default: 20)",
    )

    args = parser.parse_args()

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Generate traces
    print("=" * 60)
    print("Golden Trace Generator for Aprender")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print(f"Test inputs: {len(args.inputs)}")
    print("=" * 60)

    traces = generate_golden_trace(
        model_name=args.model,
        test_inputs=args.inputs,
        device=args.device,
        max_new_tokens=args.max_tokens,
    )

    # Save to JSON
    with open(args.output, "w") as f:
        json.dump(traces, f, indent=2)

    print()
    print("=" * 60)
    print(f"✓ Golden traces saved to: {args.output}")
    print(f"✓ Config: {traces['config']['num_hidden_layers']} layers, "
          f"{traces['config']['hidden_size']} hidden, "
          f"{traces['config']['vocab_size']} vocab")
    print(f"✓ Traces: {len(traces['traces'])} test cases")
    print("=" * 60)

    # Print verification command
    print()
    print("To verify against Rust implementation:")
    print(f"  apr test-model model.apr --golden-trace {args.output}")


if __name__ == "__main__":
    main()
