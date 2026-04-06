# Generative AI Evaluation Metrics

Entrenar provides domain-specific metrics for evaluating
generative models across speech recognition, text generation,
code synthesis, and information retrieval. All functions live
in `entrenar::eval::generative`.

## Metric Enum Variants

These metrics integrate into the unified `Metric` enum used by the evaluator and leaderboard:

```rust,ignore
use entrenar::eval::evaluator::Metric;
use entrenar::eval::RougeVariant;

let metrics = vec![
    Metric::WER,                          // ASR
    Metric::RTFx,                         // ASR speed
    Metric::BLEU,                         // Translation / text gen
    Metric::ROUGE(RougeVariant::Rouge1),  // Summarization (unigram)
    Metric::ROUGE(RougeVariant::Rouge2),  // Summarization (bigram)
    Metric::ROUGE(RougeVariant::RougeL),  // Summarization (LCS)
    Metric::Perplexity,                   // Language modeling
    Metric::MMLUAccuracy,                 // LLM benchmark
    Metric::PassAtK(1),                   // Code generation
    Metric::NDCGAtK(10),                  // Retrieval ranking
];

// Polarity: higher_is_better() returns false for WER and Perplexity
assert!(!Metric::WER.higher_is_better());
assert!(!Metric::Perplexity.higher_is_better());
assert!(Metric::BLEU.higher_is_better());
assert!(Metric::PassAtK(1).higher_is_better());
```

## WER (Word Error Rate)

Word-level Levenshtein edit distance normalized by reference length. Standard metric for ASR evaluation.

**Formula:** `WER = (Substitutions + Deletions + Insertions) / |reference_words|`

```rust,ignore
use entrenar::eval::generative::word_error_rate;

let wer = word_error_rate(
    "the cat sat on the mat",    // reference
    "the cat sit on a mat",      // hypothesis
);
// wer = 2/6 = 0.333 (two substitutions: sat->sit, the->a)

// Identical strings yield 0.0
assert_eq!(word_error_rate("hello world", "hello world"), 0.0);

// Can exceed 1.0 if hypothesis is much longer than reference
let wer = word_error_rate("short", "this is a very long hypothesis");
assert!(wer > 1.0);
```

Returns `f64::INFINITY` when the reference is empty but the hypothesis is not.

## RTFx (Real-Time Factor Inverse)
Processing speed metric for audio models. `RTFx = audio_duration / processing_time`. Higher means faster.

```rust,ignore
use entrenar::eval::generative::real_time_factor_inverse;

// Processed 60s of audio in 0.6s => 100x real-time
let rtfx = real_time_factor_inverse(0.6, 60.0);
assert_eq!(rtfx, 100.0);
```

## BLEU

Modified n-gram precision with brevity penalty
(Papineni et al., 2002). Standard metric for machine
translation. Returns a value in `[0, 1]`.

```rust,ignore
use entrenar::eval::generative::bleu_score;

let references = &[
    "the cat is on the mat",
    "there is a cat on the mat",
];
let hypothesis = "the cat is on the mat";

// max_n=4 for standard BLEU-4
let score = bleu_score(references, hypothesis, 4);
assert!(score > 0.5);
```

Internally computes clipped n-gram counts for orders 1
through `max_n`, takes the geometric mean, and applies a
brevity penalty for short hypotheses.

## ROUGE (1/2/L)

Recall-Oriented Understudy for Gisting Evaluation. Measures
n-gram overlap (ROUGE-N) or longest common subsequence
(ROUGE-L) as F1 scores.

```rust,ignore
use entrenar::eval::generative::{rouge_n, rouge_l};

let reference = "the cat sat on the mat";
let hypothesis = "the cat is on the mat";

// ROUGE-1: unigram overlap F1
let r1 = rouge_n(reference, hypothesis, 1);

// ROUGE-2: bigram overlap F1
let r2 = rouge_n(reference, hypothesis, 2);

// ROUGE-L: LCS-based F1
let rl = rouge_l(reference, hypothesis);

// ROUGE-1 >= ROUGE-2 (unigrams overlap more than bigrams)
assert!(r1 >= r2);
```

Both functions return `0.0` when either input is empty or too short for the requested n-gram order.

## Perplexity

Exponentiated average negative log-probability. Lower values
indicate a better language model. Minimum possible value is
`1.0` for a perfect predictor.

**Formula:** `PPL = exp(-1/N * sum(log_probs))`

```rust,ignore
use entrenar::eval::generative::perplexity;

// Log-probabilities from a language model
let log_probs = vec![-0.1, -0.2, -0.15, -0.3, -0.05];
let ppl = perplexity(&log_probs);
assert!(ppl >= 1.0);

// Perfect predictions (log_prob = 0 for all tokens) => PPL = 1.0
let perfect = perplexity(&[0.0, 0.0, 0.0]);
assert!((perfect - 1.0).abs() < 1e-10);

// Empty input returns infinity
assert!(perplexity(&[]).is_infinite());
```

## MMLUAccuracy

LLM benchmark accuracy covering MMLU, MMLU-PRO, BBH, and
similar multiple-choice evaluations. Represented as
`Metric::MMLUAccuracy` in the enum. Tracked as a distinct
variant to enable leaderboard-specific column mapping.

## pass@k

Unbiased estimator for functional correctness of code
generation (Chen et al., 2021). Computes the probability that
at least one of `k` samples passes all test cases.

**Formula:** `pass@k = 1 - C(n-c, k) / C(n, k)`

where `n` = total samples, `c` = correct samples, `k` = top-k threshold.

```rust,ignore
use entrenar::eval::generative::pass_at_k;

// Generated 200 samples, 50 passed, estimate pass@1
let p1 = pass_at_k(200, 50, 1);
assert!((p1 - 0.25).abs() < 0.01);  // ~25%

// pass@10 is higher (more chances to find a correct sample)
let p10 = pass_at_k(200, 50, 10);
assert!(p10 > p1);

// All correct => 1.0
assert_eq!(pass_at_k(100, 100, 1), 1.0);

// None correct => 0.0
assert_eq!(pass_at_k(100, 0, 1), 0.0);
```

Computation uses log-space arithmetic to avoid overflow for large `n`.

## NDCG@k

Normalized Discounted Cumulative Gain. Measures ranking quality
for search and retrieval by comparing the actual ranking against
the ideal (sorted) ranking. Returns a value in `[0, 1]` where
`1.0` is a perfect ranking.

**Formula:** `NDCG@k = DCG@k / IDCG@k` where `DCG = sum((2^rel_i - 1) / log2(i+1))`

```rust,ignore
use entrenar::eval::generative::ndcg_at_k;

// Relevance scores in the order returned by the system
let scores = &[3.0, 2.0, 3.0, 0.0, 1.0, 2.0];

// Perfect ranking at top-3 would be [3, 3, 2, ...]
let ndcg = ndcg_at_k(scores, 3);
assert!(ndcg > 0.0 && ndcg <= 1.0);

// Perfect ranking => 1.0
let perfect = ndcg_at_k(&[3.0, 2.0, 1.0], 3);
assert!((perfect - 1.0).abs() < 1e-10);
```

## Polarity Summary

| Metric | Direction | Domain |
|--------|-----------|--------|
| WER | Lower is better | ASR |
| RTFx | Higher is better | ASR speed |
| BLEU | Higher is better | Translation |
| ROUGE (1/2/L) | Higher is better | Summarization |
| Perplexity | Lower is better | Language modeling |
| MMLUAccuracy | Higher is better | LLM benchmarks |
| pass@k | Higher is better | Code generation |
| NDCG@k | Higher is better | Retrieval |
