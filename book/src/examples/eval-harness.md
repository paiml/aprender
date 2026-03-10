# Case Study: Evaluation Harness

**Ticket**: GH-454 | **Contract**: `finetune.md §eval`

## Overview

A lightweight evaluation framework compatible with the lm-evaluation-harness
approach. Supports multiple-choice tasks (HellaSwag, MMLU, ARC), perplexity
evaluation, and metric aggregation across benchmarks.

## API

```rust
use aprender::online::eval_harness::{
    EvalExample, EvalTask, HarnessConfig, TaskType,
    run_harness, compute_perplexity, compute_accuracy,
};

// Define a multiple-choice task
let mut task = EvalTask::new("hellaswag", TaskType::MultipleChoice);
task.add_example(EvalExample {
    context: "A person is making a sandwich. They".to_string(),
    choices: vec![
        " spread butter on the bread.".to_string(),
        " flew into space.".to_string(),
    ],
    gold_idx: Some(0),
    reference: None,
});

// Run with a scoring function
let config = HarnessConfig {
    tasks: vec![task],
    length_normalize: false,
    max_examples: 0,
};

let report = run_harness(&config, |context, completion| {
    // Return log-likelihood of completion given context
    model.score(context, completion)
})?;

println!("Accuracy: {:.1}%", report.macro_accuracy * 100.0);
```

## Task Types

| Type | Use Case | Metric |
|------|----------|--------|
| `MultipleChoice` | HellaSwag, MMLU, ARC | Accuracy |
| `Perplexity` | WikiText, PTB | PPL |
| `Generation` | Translation, summarization | (placeholder) |
| `Classification` | Sentiment, NLI | Accuracy |

## Key Properties

- **Generic scoring**: Any `Fn(&str, &str) -> f64` can be used as scorer
- **Macro averaging**: Report aggregates accuracy across tasks equally
- **Max examples**: Limit evaluation to N examples per task for quick checks
- **Mock data**: `mock_hellaswag()` provides test data without external files

## Falsification Tests

| Test | Property |
|------|----------|
| FALSIFY-EVAL-001 | Perplexity is always positive |
| FALSIFY-EVAL-002 | Accuracy is bounded in [0.0, 1.0] |
| FALSIFY-EVAL-003 | Report aggregation is consistent |

## Run the Example

```bash
cargo run --example eval_harness
```
