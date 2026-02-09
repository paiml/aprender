# Case Study: Hex Forensics — Format-Aware Binary Inspection

## Why Binary Forensics?

When model inference produces garbage, you need to see the actual bytes. Not a high-level
summary — the *raw data*. Traditional tools like `xxd` show bytes but don't understand
model formats. `apr hex` bridges this gap: format-aware binary inspection that annotates
GGUF headers, dequantizes Q4K/Q6K blocks, computes value distributions, and flags
anomalies — all in a single command.

**Toyota Way**: *Genchi Genbutsu* — go and see the actual data at the source of the problem.

## Quick Reference

```bash
# Auto-detect format, show summary + hex dump
apr hex model.gguf

# Annotated file header (magic, version, tensor_count, metadata)
apr hex model.gguf --header

# Raw bytes with ASCII column (like xxd, but format-aware)
apr hex model.gguf --raw --width 32 --limit 512

# Quantization super-block structure (Q4K/Q6K/Q8_0)
apr hex model.gguf --blocks --tensor "attn_q"

# Value distribution histogram + entropy + kurtosis
apr hex model.gguf --distribution --tensor "output.weight"

# Per-region byte entropy (corruption detection)
apr hex model.gguf --entropy

# GGUF → APR layout contract overlay
apr hex model.gguf --contract

# List all tensors with dtype and shape
apr hex model.gguf --list

# JSON output for scripting
apr hex model.gguf --json --tensor "attn_q"
```

## Supported Formats

| Format | Modes | Notes |
|--------|-------|-------|
| GGUF | All 8 modes | Full support including blocks, contract |
| APR | header, raw, list, stats, distribution, entropy | Native format |
| SafeTensors | header, raw, list, entropy | JSON header + tensor data |

Format is auto-detected from magic bytes:
- `47 47 55 46` = `GGUF`
- `41 50 52 00` = `APR\0`
- First 8 bytes as u64 LE < 100MB = SafeTensors header length

## Mode Deep Dives

### `--header`: Annotated File Header

Shows the file header with byte offsets, raw hex, and decoded values:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GGUF File Header
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  00000000: 47 47 55 46                 magic: "GGUF"
  00000004: 03 00 00 00                 version: 3
  00000008: 23 01 00 00 00 00 00 00     tensor_count: 291
  00000010: 1A 00 00 00 00 00 00 00     metadata_kv_count: 26
```

Color coding: dimmed offsets, yellow hex bytes, bold white labels, cyan values.

### `--blocks`: Quantization Super-Block View

Annotates the internal structure of quantized blocks:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Block View: blk.0.ffn_down.weight (Q6_K, [4864, 896])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q6_K Super-Block #0 (256 elements, 210 bytes):
  00000000: 52 F2 40 26 24 D2 1B 22 ..  ql[0-127]: low 4 bits
  00000080: A6 9E A4 95 66 9A 8B AA ..  qh[0-63]: high 2 bits
  000000C0: D4 CC BD DC 67 CD 80 99 ..  scales[0-15]: 16 sub-block scales
  000000D0: 5F 01                       d (scale): 0.00002 (f16)
```

Supported dtypes: Q4_K (144B/256elem), Q6_K (210B/256elem), Q8_0 (34B/32elem).

### `--distribution`: Value Histogram

Dequantizes tensor values and shows the distribution:

```
Distribution: blk.0.attn_norm.weight
  [  -0.532,   -0.425)                                            0.2%
  [  -0.104,    0.003)  ██████████████████████████               34.5%
  [   0.003,    0.110)  ████████████████████████████████████████ 52.1%
  [   0.110,    0.216)  ██████                                    8.8%

  Entropy: 1.62 bits
  Kurtosis: 6.06
  Min: -0.531738
  Max: 0.537109
  Mean: 0.031080
  Std: 0.095854
```

### `--entropy`: Byte Entropy Analysis

Computes Shannon entropy with sliding window anomaly detection:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Byte Entropy Analysis (GGUF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total entropy: 7.9429 bits (0.0 = uniform, 8.0 = random)
  File size: 468.64 MiB
  Expected range: Q4K/Q6K: 7.5-8.0, F32: 5.0-7.5, F16: 6.0-7.5
  ─── Sliding Window (4KB)
  Min entropy: 3.0821 at 0x0
  Max entropy: 7.9293 at 0xEF010F0
```

Anomalous regions (entropy < 1.0) indicate corruption or all-zeros.

### `--contract`: Layout Contract Overlay

Shows the GGUF→APR tensor name mapping with transpose requirements:

```
╭───────────────────────────┬────────────────────────────────────────┬───────────┬──────────╮
│ GGUF Name                 │ APR Name                               │ Transpose │ Critical │
├───────────────────────────┼────────────────────────────────────────┼───────────┼──────────┤
│ output.weight             │ lm_head.weight                         │ Yes       │ CRITICAL │
│ token_embd.weight         │ model.embed_tokens.weight              │ Yes       │ -        │
│ blk.0.attn_norm.weight    │ model.layers.{n}.input_layernorm.weight│ No        │ -        │
╰───────────────────────────┴────────────────────────────────────────┴───────────┴──────────╯
```

## Algorithms

### Shannon Entropy

```
H = -Σ p(x) * log2(p(x))
```

Where `p(x)` is the frequency of byte value `x` in the data. Range: 0.0 (all bytes
identical) to 8.0 (perfectly uniform random). Quantized weights typically show 7.5-8.0;
values below 5.0 suggest corruption or padding.

### f16 → f32 Conversion

IEEE 754 half-precision uses 1 sign bit, 5 exponent bits, 10 mantissa bits. The
conversion handles three cases: zero/subnormal (denormalize), normal (bias adjustment
`exp + 112`), and special (Inf/NaN propagation). The bias trick `exp + 112` (where
112 = 127 - 15) avoids unsigned integer underflow.

### Q4_K / Q6_K Dequantization

Each super-block stores 256 elements with a shared scale factor `d` (f16) and per-element
quantized values. Dequantization: `value = d * (quant - zero_point)`. The block view
shows the raw structure so you can verify the dequantization pipeline is reading the
correct offsets.

## Example

```bash
cargo run --example hex_forensics
```

See `examples/hex_forensics.rs` for standalone implementations of all algorithms.

## Debugging Workflow

1. **Start with `--header`** — verify format, version, tensor count
2. **Use `--list`** — find tensor names and shapes
3. **Use `--blocks`** — verify quantization structure reads correct offsets
4. **Use `--distribution`** — check for NaN, zero clusters, unexpected ranges
5. **Use `--entropy`** — detect corruption or zero-padding regions
6. **Use `--contract`** — verify GGUF→APR name mapping and transpose flags
