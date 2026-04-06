# SentencePiece Tokenization

SentencePiece is a language-independent tokenization approach used by LLaMA, TinyLlama, and many other modern LLMs. Realizar implements SentencePiece tokenization for GGUF models.

## Key Concepts

### The `▁` (U+2581) Character

SentencePiece uses the special character `▁` (Unicode Lower One Eighth Block, U+2581) to represent word boundaries:

- **Spaces are converted to `▁`** before tokenization
- Tokens like `▁Paris`, `▁capital`, `▁the` include the word boundary marker
- This allows the tokenizer to distinguish between word-initial and word-internal occurrences

### Example

```text
Input:  "The capital of France is Paris"
After:  "The▁capital▁of▁France▁is▁Paris"
Tokens: [1576, 7483, 310, 3444, 338, 3681]
        "The" "▁capital" "▁of" "▁France" "▁is" "▁Paris"
```

## Implementation

The `GGUFModel::encode()` function implements SentencePiece tokenization:

```rust
pub fn encode(&self, text: &str) -> Option<Vec<u32>> {
    // SentencePiece preprocessing: replace spaces with ▁ (U+2581)
    let processed = if text.starts_with(' ') {
        text.replace(' ', "▁")
    } else {
        // Replace internal spaces with ▁
        let mut result = String::new();
        for ch in text.chars() {
            if ch == ' ' {
                result.push('▁');
            } else {
                result.push(ch);
            }
        }
        result
    };

    // Greedy longest match tokenization
    // ... (uses character boundaries for UTF-8 safety)
}
```

### UTF-8 Safety

The tokenizer uses character boundaries rather than byte indices to safely handle multi-byte UTF-8 characters like `▁` (3 bytes):

```rust
// Collect character byte offsets for proper slicing
let char_indices: Vec<usize> = remaining
    .char_indices()
    .map(|(i, _)| i)
    .chain(std::iter::once(remaining.len()))
    .collect();

// Try all prefixes up to 32 chars
for char_count in 1..=char_indices.len().min(32) {
    let byte_end = char_indices[char_count];
    let prefix = &remaining[..byte_end];
    // ... lookup in vocabulary
}
```

## Vocabulary Access

GGUF models store the vocabulary in metadata:

```rust
// Get vocabulary from GGUF metadata
let vocab = model.vocabulary().unwrap();

// Vocabulary is indexed by token ID
assert_eq!(vocab[3681], "▁Paris");
assert_eq!(vocab[1576], "The");

// Special tokens
let bos_id = model.bos_token_id(); // Usually 1
let eos_id = model.eos_token_id(); // Usually 2
```

## Testing Tokenization

```rust
use realizar::gguf::GGUFModel;

let model = GGUFModel::from_bytes(&data).unwrap();

// Test encoding
let tokens = model.encode("The capital of France is").unwrap();
assert_eq!(tokens, vec![1576, 7483, 310, 3444, 338]);

// Test decoding
let text = model.decode(&tokens);
assert_eq!(text, "The▁capital▁of▁France▁is");
```

## Critical Fix: Space Handling

A critical bug was fixed where spaces were tokenized as byte tokens (`<0x20>`) instead of being converted to `▁`:

**Before (broken):**
```text
"The capital" -> [1576, 35, 5030, 2410]  // 35 = <0x20> byte token
                  "The" " " "cap" "ital"
```

**After (correct):**
```text
"The capital" -> [1576, 7483]
                  "The" "▁capital"
```

This fix improved model output from gibberish to correct predictions:
- "The capital of France is" + model -> "Paris" (rank 1)

## See Also

- [BPE Tokenization](./bpe.md)
- [Vocabulary Management](./vocabulary.md)
- [GGUF Format](../formats/gguf.md)
