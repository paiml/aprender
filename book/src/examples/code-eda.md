# Case Study: Code-Aware EDA (Easy Data Augmentation)

Syntax-aware data augmentation for source code, preserving semantic validity while generating diverse training samples.

## Quick Start

```rust
use aprender::synthetic::code_eda::{CodeEda, CodeEdaConfig, CodeLanguage};
use aprender::synthetic::{SyntheticGenerator, SyntheticConfig};

// Configure for Rust code
let config = CodeEdaConfig::default()
    .with_language(CodeLanguage::Rust)
    .with_rename_prob(0.15)
    .with_comment_prob(0.1);

let generator = CodeEda::new(config);

// Augment code samples
let seeds = vec![
    "let x = 42;\nprintln!(\"{}\", x);".to_string(),
];

let synth_config = SyntheticConfig::default()
    .with_augmentation_ratio(2.0)
    .with_quality_threshold(0.3)
    .with_seed(42);

let augmented = generator.generate(&seeds, &synth_config)?;
```

## Why Code-Specific Augmentation?

Traditional EDA (Wei & Zou, 2019) works on natural language but fails on code:

| Text EDA | Code EDA |
|----------|----------|
| Random word swap | Preserves syntax |
| Synonym replacement | Variable renaming |
| Random deletion | Dead code removal |
| Random insertion | Comment insertion |

**Key difference:** Code has structure. `x = 1; y = 2;` can become `y = 2; x = 1;` only if statements are independent.

## Augmentation Operations

### 1. Variable Renaming (VR)

Replace identifiers with semantic synonyms:

```rust
// Original
let x = calculate();
let i = 0;
let buf = Vec::new();

// Augmented
let value = calculate();  // x → value
let index = 0;            // i → index
let buffer = Vec::new();  // buf → buffer
```

**Built-in synonym mappings:**

| Original | Alternatives |
|----------|--------------|
| `x` | `value`, `val` |
| `y` | `result`, `res` |
| `i` | `index`, `idx` |
| `j` | `inner`, `jdx` |
| `n` | `count`, `num` |
| `tmp` | `temp`, `scratch` |
| `buf` | `buffer`, `data` |
| `len` | `length`, `size` |
| `err` | `error`, `e` |

**Reserved keywords are never renamed:**

- Rust: `let`, `mut`, `fn`, `impl`, `struct`, `enum`, `trait`, etc.
- Python: `def`, `class`, `import`, `if`, `for`, `while`, etc.

### 2. Comment Insertion (CI)

Add language-appropriate comments:

```rust
// Rust
let x = 42;
// TODO: review    ← inserted
let y = x + 1;
```

```python
# Python
x = 42
# NOTE: temp       ← inserted
y = x + 1
```

### 3. Statement Reorder (SR)

Swap adjacent independent statements:

```rust
// Original
let a = 1;
let b = 2;
let c = 3;

// Augmented (swap a,b)
let b = 2;
let a = 1;
let c = 3;
```

**Delimiter detection:**
- Rust: semicolons (`;`)
- Python: newlines (`\n`)

### 4. Dead Code Removal (DCR)

Remove comments and collapse whitespace:

```rust
// Original
let x = 1;  // important value
let y = 2;  /* temp */

// Augmented
let x = 1;
let y = 2;
```

## Configuration

### `CodeEdaConfig`

```rust
CodeEdaConfig::default()
    .with_rename_prob(0.15)      // Variable rename probability
    .with_comment_prob(0.1)      // Comment insertion probability
    .with_reorder_prob(0.05)     // Statement reorder probability
    .with_remove_prob(0.1)       // Dead code removal probability
    .with_num_augments(4)        // Augmentations per input
    .with_min_tokens(5)          // Skip short code
    .with_language(CodeLanguage::Rust)
```

### Supported Languages

```rust
pub enum CodeLanguage {
    Rust,    // Full syntax awareness
    Python,  // Full syntax awareness
    Generic, // Language-agnostic operations only
}
```

## Quality Metrics

### Token Overlap

Measures semantic preservation via Jaccard similarity:

```rust
let generator = CodeEda::new(CodeEdaConfig::default());

let original = "let x = 42;";
let augmented = "let value = 42;";

let overlap = generator.token_overlap(original, augmented);
// overlap ≈ 0.75 (shared: let, =, 42, ;)
```

### Quality Score

Penalizes extremes (too similar or too different):

| Overlap | Quality | Interpretation |
|---------|---------|----------------|
| > 0.95 | 0.5 | Too similar, little augmentation |
| 0.3-0.95 | overlap | Good augmentation |
| < 0.3 | 0.3 | Too different, likely corrupted |

### Diversity Score

Measures batch diversity (inverse of average pairwise overlap):

```rust
let batch = vec![
    "let x = 1;".to_string(),
    "fn foo() {}".to_string(),
];

let diversity = generator.diversity_score(&batch);
// diversity > 0.5 (different code patterns)
```

## Integration with aprender-shell

The `aprender-shell` CLI supports CodeEDA for shell command augmentation:

```bash
# Train with code-aware augmentation
aprender-shell augment --use-code-eda

# View augmentation statistics
aprender-shell stats --augmented
```

## Use Cases

### 1. Defect Prediction Training

Augment labeled commit diffs to improve classifier robustness:

```rust
let buggy_code = vec![
    "if (x = null) return;".to_string(),  // Assignment instead of comparison
];

let augmented = generator.generate(&buggy_code, &config)?;
// Train classifier on original + augmented samples
```

### 2. Code Clone Detection

Generate synthetic near-clones for contrastive learning:

```rust
let original = "fn add(a: i32, b: i32) -> i32 { a + b }";

// Generate variations with same semantics
let clones = generator.generate(&[original.to_string()], &config)?;
```

### 3. Code Completion Training

Augment training data for autocomplete models:

```rust
let completions = vec![
    "git commit -m 'fix bug'".to_string(),
    "cargo build --release".to_string(),
];

// 2x training data with variations
let augmented = generator.generate(&completions, &SyntheticConfig::default()
    .with_augmentation_ratio(2.0))?;
```

## Deterministic Generation

CodeEDA uses a seeded PRNG for reproducibility:

```rust
let generator = CodeEda::new(CodeEdaConfig::default());

let aug1 = generator.augment("let x = 1;", 42);
let aug2 = generator.augment("let x = 1;", 42);

assert_eq!(aug1, aug2);  // Same seed = same output
```

## Custom Synonyms

Extend the synonym dictionary:

```rust
use aprender::synthetic::code_eda::VariableSynonyms;

let mut synonyms = VariableSynonyms::new();
synonyms.add_synonym(
    "conn".to_string(),
    vec!["connection".to_string(), "db".to_string()],
);
synonyms.add_synonym(
    "ctx".to_string(),
    vec!["context".to_string(), "cx".to_string()],
);
```

## Performance

CodeEDA is designed for batch augmentation efficiency:

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Tokenization | O(n) | Single pass, no regex |
| Variable rename | O(n) | HashMap lookup |
| Comment insertion | O(n) | Single pass |
| Statement reorder | O(n) | Split + swap |
| Quality score | O(n) | Token set operations |

Typical throughput: **50,000+ augmentations/second** on modern hardware.

## References

- Wei & Zou (2019). "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"
- D'Ambros et al. (2012). "Evaluating Defect Prediction Approaches" (defect prediction context)
- [Synthetic Data Generation](./synthetic-data-generation.md) - General EDA for text

## See Also

- [CodeFeatureExtractor](./code-feature-extractor.md) - 8-dimensional commit feature extraction
- [Shell Completion](./shell-completion.md) - AI-powered shell autocomplete
- [Shell Completion Benchmarks](./shell-completion-benchmarks.md) - Sub-10ms latency verification
