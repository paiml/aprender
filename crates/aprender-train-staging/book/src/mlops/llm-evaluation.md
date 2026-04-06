# LLM Evaluation

Evaluate LLM outputs for relevance, coherence, groundedness, and harmfulness.

## Toyota Principle: Genchi Genbutsu

"Go and see" - directly observe model outputs to understand quality. Systematic evaluation enables data-driven
improvement.

## Quick Start

```rust
use entrenar::monitor::llm::{InMemoryLLMEvaluator, LLMEvaluator, EvalResult};

let mut evaluator = InMemoryLLMEvaluator::new();

// Evaluate a response
let result = evaluator.evaluate_response(
    "run-123",
    "What is machine learning?",           // prompt
    "Machine learning is a subset of AI...", // response
    Some("ML is artificial intelligence..."), // ground truth (optional)
)?;

println!("Relevance: {:.2}", result.relevance);
println!("Coherence: {:.2}", result.coherence);
println!("Groundedness: {:.2}", result.groundedness);
println!("Harmfulness: {:.2}", result.harmfulness);
```

## Evaluation Metrics

### Relevance (0.0 - 1.0)

Measures how well the response addresses the prompt:

```rust
// High relevance: response directly answers the question
// Low relevance: response is off-topic or tangential

// Computed by word overlap between prompt and response
let relevance = result.relevance;
```

### Coherence (0.0 - 1.0)

Measures logical flow and readability:

```rust
// High coherence: well-structured, logical flow
// Low coherence: disjointed, contradictory

// Computed by sentence structure analysis
let coherence = result.coherence;
```

### Groundedness (0.0 - 1.0)

Measures faithfulness to source material:

```rust
// High groundedness: claims supported by context
// Low groundedness: hallucinated or unsupported claims

// Requires ground truth for comparison
let groundedness = result.groundedness;
```

### Harmfulness (0.0 - 1.0)

Measures presence of harmful content:

```rust
// Low harmfulness: safe, appropriate content
// High harmfulness: toxic, dangerous, or inappropriate

// Keyword-based detection
let harmfulness = result.harmfulness;
```

## Prompt Tracking

Track prompt versions for A/B testing:

```rust
use entrenar::monitor::llm::{PromptVersion, PromptId};

let prompt = PromptVersion {
    id: PromptId::new("prompt-v1"),
    template: "Answer the following question: {question}".to_string(),
    version: 1,
    metadata: Some(serde_json::json!({
        "author": "alice",
        "description": "Basic QA prompt"
    })),
};

evaluator.track_prompt("run-123", &prompt)?;

// List prompts for a run
let prompts = evaluator.get_prompts("run-123")?;
```

## Batch Evaluation

```rust
let responses = vec![
    ("What is AI?", "AI is...", Some("Artificial intelligence...")),
    ("Explain ML", "ML uses...", Some("Machine learning...")),
    ("Define DL", "DL is...", Some("Deep learning...")),
];

let mut total_relevance = 0.0;

for (prompt, response, ground_truth) in responses {
    let result = evaluator.evaluate_response(
        "run-123",
        prompt,
        response,
        ground_truth,
    )?;

    total_relevance += result.relevance;
}

let avg_relevance = total_relevance / responses.len() as f64;
println!("Average relevance: {:.2}", avg_relevance);
```

## LLM Metrics Logging

```rust
use entrenar::monitor::llm::LLMMetrics;

let metrics = LLMMetrics {
    prompt_tokens: 50,
    completion_tokens: 150,
    total_tokens: 200,
    latency_ms: 500,
    model: "gpt-4".to_string(),
    temperature: Some(0.7),
    top_p: Some(0.9),
};

evaluator.log_llm_call("run-123", metrics)?;

// Retrieve metrics
let all_metrics = evaluator.get_metrics("run-123")?;
for m in all_metrics {
    println!("Tokens: {}, Latency: {}ms", m.total_tokens, m.latency_ms);
}
```

## Aggregate Metrics

```rust
use entrenar::monitor::llm::AggregateMetrics;

let aggregate = evaluator.aggregate_metrics("run-123")?;

println!("Total calls: {}", aggregate.total_calls);
println!("Total tokens: {}", aggregate.total_tokens);
println!("Avg latency: {:.0}ms", aggregate.avg_latency_ms);
println!("Avg relevance: {:.2}", aggregate.avg_relevance);
println!("Avg coherence: {:.2}", aggregate.avg_coherence);
```

## Integration with Training

```rust
use entrenar::train::callback::LLMEvalCallback;

let callback = LLMEvalCallback::new()
    .with_eval_samples(100)
    .with_ground_truth_path("data/test.jsonl");

trainer.add_callback(callback);

// Evaluation runs automatically at end of each epoch
trainer.fit(&model, &dataset)?;
```

## Cargo Run Example

```bash
# Evaluate single response
cargo run --example llm_eval -- \
    --prompt "What is ML?" \
    --response "Machine learning is..."

# Evaluate from file
cargo run --example llm_eval -- \
    --input responses.jsonl \
    --output eval_results.json

# With ground truth
cargo run --example llm_eval -- \
    --input responses.jsonl \
    --ground-truth ground_truth.jsonl
```

## Custom Evaluators

```rust
use entrenar::monitor::llm::{LLMEvaluator, EvalResult};

struct CustomEvaluator {
    // Custom state
}

impl LLMEvaluator for CustomEvaluator {
    fn evaluate_response(
        &mut self,
        run_id: &str,
        prompt: &str,
        response: &str,
        ground_truth: Option<&str>,
    ) -> Result<EvalResult> {
        // Custom evaluation logic
        let relevance = custom_relevance(prompt, response);
        let coherence = custom_coherence(response);
        let groundedness = custom_groundedness(response, ground_truth);
        let harmfulness = custom_harmfulness(response);

        Ok(EvalResult::new(relevance, coherence, groundedness, harmfulness))
    }

    // ... implement other methods
}
```

## Evaluation Report

```rust
// Generate evaluation report
let report = evaluator.generate_report("run-123")?;

println!("{}", report);
```

Output:
```
=== LLM Evaluation Report ===
Run: run-123
Total evaluations: 100

Metrics Summary:
  Relevance:    0.85 ± 0.12
  Coherence:    0.92 ± 0.08
  Groundedness: 0.78 ± 0.15
  Harmfulness:  0.02 ± 0.05

Token Usage:
  Total tokens: 25,000
  Avg per call: 250

Latency:
  Avg: 450ms
  P95: 850ms
  P99: 1200ms
```

## Best Practices

1. **Always include ground truth** - Enables groundedness measurement
2. **Evaluate on diverse prompts** - Avoid overfitting to specific patterns
3. **Track prompt versions** - Enable A/B testing
4. **Log token usage** - Monitor costs
5. **Set quality thresholds** - Fail builds on low scores

## See Also

- [MLOps Overview](./overview.md)
- [Experiment Tracking](./experiment-tracking.md)
- [Quality Gates (Jidoka)](../monitor/quality-gates.md)
