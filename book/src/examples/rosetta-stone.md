# Case Study: Rosetta Stone — Universal Model Format Converter

## Overview

The Rosetta Stone pattern provides universal model format conversion between APR, GGUF,
and SafeTensors formats. It handles format detection, direct conversion paths, multi-step
chains, and tokenizer preservation.

**Run command:**
```bash
cargo run --example rosetta_stone
```

## Key Concepts

- **Format Detection**: Identifies APR, GGUF, SafeTensors from magic bytes and extensions
- **Direct Conversion**: Single-step A to B conversion (e.g., SafeTensors to APR)
- **Multi-Step Chains**: A to B to C when no direct path exists
- **Round-Trip Verification**: Validates lossless conversion via tensor comparison
- **Tokenizer Preservation (PMAT-APR-TOK-001)**: Embedded tokenizers travel with the model

## Tokenizer Preservation

APR format embeds tokenizers during conversion, making models truly portable:

| Source Format | Tokenizer Source |
|--------------|------------------|
| SafeTensors to APR | Reads sibling `tokenizer.json` (vocab, BOS/EOS tokens) |
| GGUF to APR | Extracts vocabulary from GGUF metadata |
| APR inference | Uses embedded tokenizer for automatic token decoding |

Verification: `strings model.apr | grep tokenizer.vocabulary`

## Usage

```rust
use aprender::format::rosetta::{
    ConversionOptions, ConversionPath, FormatType, RosettaStone, TensorInfo,
};

fn main() {
    // Detect format from file
    let format = FormatType::detect("model.safetensors");

    // Plan conversion path
    let path = RosettaStone::plan_conversion(
        FormatType::SafeTensors,
        FormatType::Apr,
    );

    // Execute with options
    let options = ConversionOptions::default()
        .with_quantization("q4k");

    RosettaStone::convert("input.safetensors", "output.apr", &options)
        .expect("conversion succeeded");
}
```

## CLI Equivalent

```bash
apr convert model.safetensors -o model.apr
apr convert model.safetensors --quantize q4k -o model-q4k.apr
apr convert model.gguf -o model.apr
```

## Toyota Way Alignment

- **Genchi Genbutsu**: Inspect actual tensor data before/after conversion
- **Jidoka**: Stop on any conversion anomaly (dimension mismatch, NaN)
- **Kaizen**: Multi-step chains for iterative improvement

## See Also

- [APR Format Deep Dive](./apr-format-deep-dive.md)
- [Model Serialization](./model-serialization.md)
- [APR CLI Commands](./apr-cli-commands.md)
