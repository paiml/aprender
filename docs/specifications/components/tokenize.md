# Tokenize Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §20
**Status**: Active
**CLI**: `apr tokenize`
**Implementation**: `crates/apr-cli/src/commands/tokenize.rs`
**Library**: `aprender::text`

---

## 1. Overview

Tokenizer operations: BPE vocabulary training, token breakdown with vocabulary
lookup, and corpus analysis.

## 2. CLI Interface

```
apr tokenize plan --corpus <FILE> --vocab-size <N>
apr tokenize apply --plan <YAML> --output <PATH>
apr tokenize <FILE> --text "Hello world" [--vocab]
```

## 3. BPE Training Pipeline

### Plan Phase
```bash
apr tokenize plan \
  --corpus training-data.txt \
  --vocab-size 32000 \
  --min-frequency 2
```

Analyzes corpus, estimates training time, validates parameters.

### Apply Phase
```bash
apr tokenize apply --plan tokenizer-plan.yaml -o tokenizer.json
```

Trains BPE tokenizer from corpus.

## 4. Token Analysis

```bash
apr tokenize model.apr --text "Hello world"
# Output: token IDs, decoded tokens, byte pairs
```

## 5. Features

- BPE (Byte Pair Encoding) vocabulary learning
- Special token configuration (BOS, EOS, PAD, UNK)
- Unicode normalization (NFC, NFKC)
- Pre-tokenization rules (whitespace, punctuation)

---

## Provable Contracts

### Contract: `tokenizer-v1.yaml`

```yaml
metadata:
  description: "Tokenizer — BPE training, encode/decode roundtrip"
  references:
    - "Sennrich et al. (2016) BPE: Neural Machine Translation of Rare Words"

equations:
  bpe_merge:
    formula: "merge(corpus, pair) replaces all occurrences of pair with new token"
    invariants:
      - "Vocab size increases by 1 per merge"
      - "Most frequent pair selected for merge"
      - "Corpus content preserved after merge"

  encode_decode_roundtrip:
    formula: "decode(encode(text)) == text"
    invariants:
      - "Lossless for all valid UTF-8 input"
      - "Unknown bytes handled via byte fallback"
      - "Special tokens not generated from normal text"

  vocab_size:
    formula: "|vocab| == base_vocab + num_merges"
    invariants:
      - "base_vocab = 256 (byte-level BPE)"
      - "num_merges = requested_vocab_size - base_vocab - num_special"
      - "Final vocab size matches requested"

proof_obligations:
  - type: invariant
    property: "Encode/decode roundtrip"
    formal: "decode(encode(text)) == text for valid UTF-8"
  - type: invariant
    property: "Vocab size correctness"
    formal: "|vocab| == requested_vocab_size"
  - type: invariant
    property: "BPE merge frequency"
    formal: "merged pair has highest frequency among candidates"
  - type: invariant
    property: "Special token isolation"
    formal: "encode(normal_text) never produces BOS/EOS/PAD token IDs"

falsification_tests:
  - id: FALSIFY-TOK-001
    rule: "Roundtrip"
    prediction: "decode(encode(text)) == text for ASCII and Unicode"
    if_fails: "Encoding loses information or decode misaligns"
  - id: FALSIFY-TOK-002
    rule: "Vocab size"
    prediction: "trained tokenizer has exactly requested vocab size"
    if_fails: "Merge count or special token count wrong"
  - id: FALSIFY-TOK-003
    rule: "Special token isolation"
    prediction: "encoding 'Hello world' never returns BOS/EOS IDs"
    if_fails: "Special tokens in merge table or collision"
  - id: FALSIFY-TOK-004
    rule: "Empty string"
    prediction: "encode('') returns empty token list, not error"
    if_fails: "Edge case handling for empty input"
  - id: FALSIFY-TOK-005
    rule: "Byte fallback"
    prediction: "invalid UTF-8 bytes encoded via byte tokens"
    if_fails: "Byte fallback missing, panic on invalid input"

kani_harnesses:
  - id: KANI-TOK-001
    obligation: "Vocab size"
    property: "base + merges + special == total for bounded merge count"
    bound: 32
    strategy: exhaustive
```

### Binding Requirements

```yaml
  - contract: tokenizer-v1.yaml
    equation: encode_decode_roundtrip
    module_path: "aprender::text::tokenizer"
    function: "encode / decode"
    status: implemented

  - contract: tokenizer-v1.yaml
    equation: bpe_merge
    module_path: "aprender::text::bpe"
    function: train_bpe
    status: implemented
```
