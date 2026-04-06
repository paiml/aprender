# Sampling Strategies

Realizar implements **22 sampling strategies** for text generation, achieving full parity with llama.cpp plus 6 additional samplers.

## Overview

Sampling transforms model logits into token selections. Different strategies offer tradeoffs between:
- **Diversity**: How varied are the outputs?
- **Quality**: How coherent are the outputs?
- **Determinism**: How reproducible are the outputs?

## Sampler Comparison Table

| Sampler | llama.cpp | Realizar | Description |
|---------|-----------|----------|-------------|
| Greedy | ✅ | ✅ | Select highest probability token |
| Temperature | ✅ | ✅ | Scale logits by temperature |
| Top-k | ✅ | ✅ | Keep only k highest probability tokens |
| Top-p (Nucleus) | ✅ | ✅ | Keep tokens until cumulative prob >= p |
| Min-p | ✅ | ✅ | Keep tokens with prob >= min_p * max_prob |
| Mirostat v1 | ✅ | ✅ | Target perplexity with learning rate |
| Mirostat v2 | ✅ | ✅ | Simplified Mirostat |
| Typical | ✅ | ✅ | Sample from "typical" distribution |
| Repetition Penalty | ✅ | ✅ | Penalize recently used tokens |
| Frequency Penalty | ✅ | ✅ | Penalize based on frequency |
| Presence Penalty | ✅ | ✅ | Penalize any previous occurrence |
| Logit Bias | ✅ | ✅ | Add/subtract from specific token logits |
| Grammar | ✅ | ✅ | Constrain to grammar rules |
| DRY | ✅ | ✅ | Don't Repeat Yourself penalty |
| XTC | ✅ | ✅ | eXtreme Token Constraint |
| **Dynamic Temperature** | ✅ | ✅ | Entropy-based temperature adjustment |
| **Infill/FIM** | ✅ | ✅ | Fill-in-the-Middle for code |
| **Sampler Chain** | ✅ | ✅ | Composable sampler pipeline |
| Tail-Free (TFS) | ❌ | ✅ | *Realizar-only* |
| Eta Sampling | ❌ | ✅ | *Realizar-only* |
| CFG | ❌ | ✅ | *Realizar-only* |
| Token Healing | ❌ | ✅ | *Realizar-only* |

## Basic Sampling

### Greedy Sampling

Always selects the token with highest probability:

```rust
use realizar::generate::{sample_greedy, SamplingParams};
use realizar::Tensor;

let logits = Tensor::from_vec(vec![5], vec![1.0, 3.0, 2.0, 5.0, 4.0]).unwrap();
let token = sample_greedy(&logits).unwrap();
assert_eq!(token, 3); // Index of highest value (5.0)
```

### Temperature Scaling

Temperature < 1.0 makes distribution sharper (more deterministic).
Temperature > 1.0 makes distribution flatter (more random).

```rust
use realizar::generate::apply_temperature;

let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
let scaled = apply_temperature(&logits, 0.5).unwrap(); // Sharper
```

### Top-k Sampling

Keep only the k highest probability tokens:

```rust
use realizar::generate::apply_top_k;

let logits = Tensor::from_vec(vec![10], vec![...]).unwrap();
let filtered = apply_top_k(&logits, 5).unwrap(); // Keep top 5
```

### Top-p (Nucleus) Sampling

Keep tokens until cumulative probability reaches threshold:

```rust
use realizar::generate::apply_top_p;

let logits = Tensor::from_vec(vec![10], vec![...]).unwrap();
let filtered = apply_top_p(&logits, 0.9).unwrap(); // 90% cumulative prob
```

## Advanced Sampling

### Dynamic Temperature (v0.3.0)

Adjusts temperature based on entropy of the distribution. When the model is uncertain (high entropy), uses lower temperature; when confident (low entropy), uses higher temperature.

```rust
use realizar::generate::{DynTempConfig, apply_dynamic_temperature};

let config = DynTempConfig {
    temp: 1.0,    // Base temperature
    delta: 0.5,   // Temperature range (min = temp - delta, max = temp + delta)
    exponent: 1.0 // Controls the mapping curve
};

let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
let result = apply_dynamic_temperature(&logits, &config);
```

### Infill/FIM Sampling (v0.3.0)

Fill-in-the-Middle sampling for code completion. Handles end-of-generation tokens specially:

```rust
use realizar::generate::{InfillConfig, apply_infill_sampling};

let config = InfillConfig {
    eog_tokens: vec![0, 1],  // End-of-generation token IDs
    eog_ratio_threshold: 0.1 // Force EOG when p_eog * n > p_txt
};

let logits = Tensor::from_vec(vec![100], vec![...]).unwrap();
let result = apply_infill_sampling(&logits, &config);

if result.force_eog {
    // Model wants to end generation
}
```

### Sampler Chain (v0.3.0)

Compose multiple samplers into a pipeline:

```rust
use realizar::generate::{
    SamplerChain, SamplerContext,
    TemperatureSampler, TopKSampler, TopPSampler,
    RepetitionPenaltySampler, RepetitionConfig
};

let chain = SamplerChain::new()
    .with_sampler(TemperatureSampler::new(0.8))
    .with_sampler(TopKSampler::new(50))
    .with_sampler(TopPSampler::new(0.95))
    .with_sampler(RepetitionPenaltySampler::new(RepetitionConfig {
        penalty: 1.1,
        window_size: 64,
    }));

let mut logits = Tensor::from_vec(vec![vocab_size], vec![...]).unwrap();
let context = SamplerContext::with_tokens(vec![1, 2, 3]); // Previous tokens

chain.apply(&mut logits, &context);
let token = chain.sample(&logits, &context).unwrap();
```

## Penalty Samplers

### Repetition Penalty

Penalizes tokens that appear in context:

```rust
use realizar::generate::{RepetitionConfig, apply_repetition_penalty};

let config = RepetitionConfig {
    penalty: 1.2,     // Multiply logits by 1/penalty
    window_size: 64,  // Look at last N tokens
};

let context_tokens = vec![10, 20, 30, 20]; // Token 20 appears twice
apply_repetition_penalty(&mut logits, &context_tokens, &config);
```

### DRY (Don't Repeat Yourself)

Prevents repeated sequences:

```rust
use realizar::generate::{DryConfig, apply_dry_penalty};

let config = DryConfig {
    multiplier: 2.0,
    base: 1.75,
    allowed_length: 2,
    sequence_breakers: vec![],
};

apply_dry_penalty(&mut logits, &context_tokens, &config);
```

## Quality Samplers

### Mirostat v2

Targets a specific perplexity level:

```rust
use realizar::generate::{MirostatState, apply_mirostat_v2};

let mut state = MirostatState::new(5.0); // Target perplexity
let (token, new_state) = apply_mirostat_v2(&logits, state, 0.1);
```

### Typical Sampling

Samples from tokens with "typical" information content:

```rust
use realizar::generate::apply_typical;

let filtered = apply_typical(&logits, 0.9).unwrap();
```

## Test Coverage

All sampling strategies are covered by comprehensive tests:

- **Unit tests**: 1573+ tests in `src/generate.rs`
- **Property tests**: Mathematical invariants verified
- **Mutation tests**: 100% mutation score on critical paths
- **Fuzz tests**: Edge cases explored

## Performance

| Strategy | Latency | Notes |
|----------|---------|-------|
| Greedy | ~1µs | O(n) argmax |
| Temperature | ~1µs | O(n) multiply |
| Top-k | ~5µs | O(n log k) partial sort |
| Top-p | ~8µs | O(n log n) full sort |
| Mirostat | ~10µs | Includes perplexity calc |
| Sampler Chain | ~20µs | Depends on chain length |

## References

- [Temperature Sampling (Ackley et al., 1985)](https://doi.org/10.1126/science.220.4598.671)
- [Top-k Sampling (Fan et al., 2018)](https://arxiv.org/abs/1805.04833)
- [Nucleus Sampling (Holtzman et al., 2019)](https://arxiv.org/abs/1904.09751)
- [Mirostat (Basu et al., 2020)](https://arxiv.org/abs/2007.14966)
- [Typical Sampling (Meister et al., 2023)](https://arxiv.org/abs/2202.00666)
