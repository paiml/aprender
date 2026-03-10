//! Evaluation Harness for Standard Benchmarks (GH-454)
//!
//! Demonstrates the evaluation harness with multiple-choice and perplexity tasks,
//! scoring functions, and metric aggregation.
//!
//! Run: cargo run --example eval_harness

use aprender::online::eval_harness::{
    compute_accuracy, compute_perplexity, mock_hellaswag, run_harness, EvalExample, EvalTask,
    HarnessConfig, TaskType,
};

fn main() {
    println!("=== Evaluation Harness (GH-454) ===\n");

    // ── 1. Scoring Primitives ──
    println!("── 1. Scoring Primitives ──");
    let ppl = compute_perplexity(-23.0, 10);
    println!("  Perplexity (LL=-23, N=10): {:.2}", ppl);

    let preds = vec![0, 1, 2, 0];
    let gold = vec![0, 1, 0, 0];
    let acc = compute_accuracy(&preds, &gold);
    println!("  Accuracy: {:.1}% ({}/{})", acc * 100.0, 3, 4);

    // ── 2. Multiple-Choice Evaluation ──
    println!("\n── 2. Multiple-Choice (HellaSwag mock) ──");
    let task = mock_hellaswag();
    println!("  Task: {} ({} examples)", task.name, task.len());

    let config = HarnessConfig {
        tasks: vec![task],
        length_normalize: false,
        max_examples: 0,
    };

    // Score function: prefer completions that are contextually related
    let report = run_harness(&config, |ctx, completion| {
        // Simple heuristic: shorter completions with common words score higher
        let base = -(completion.len() as f64) * 0.1;
        // Bonus for sensible completions
        if ctx.contains("sandwich") && completion.contains("butter") {
            base + 5.0
        } else if ctx.contains("cat") && completion.contains("purred") {
            base + 5.0
        } else {
            base
        }
    })
    .unwrap();

    let mc = &report.tasks[0];
    println!("  Accuracy: {:.1}%", mc.accuracy.unwrap_or(0.0) * 100.0);
    println!("  Predictions: {:?}", mc.predictions);

    // ── 3. Perplexity Evaluation ──
    println!("\n── 3. Perplexity Evaluation ──");
    let mut ppl_task = EvalTask::new("wikitext_sample", TaskType::Perplexity);
    ppl_task.add_example(EvalExample {
        context: String::new(),
        choices: vec![],
        gold_idx: None,
        reference: Some("The Mona Lisa is a half-length portrait painting by the Italian artist Leonardo da Vinci".to_string()),
    });
    ppl_task.add_example(EvalExample {
        context: String::new(),
        choices: vec![],
        gold_idx: None,
        reference: Some(
            "Machine learning is a subset of artificial intelligence that focuses on algorithms"
                .to_string(),
        ),
    });

    let ppl_config = HarnessConfig {
        tasks: vec![ppl_task],
        length_normalize: false,
        max_examples: 0,
    };

    let ppl_report = run_harness(&ppl_config, |_, text| {
        // Mock log-likelihood: -0.3 per token (reasonable for a decent model)
        -(text.split_whitespace().count() as f64) * 0.3
    })
    .unwrap();

    let ppl_metrics = &ppl_report.tasks[0];
    println!(
        "  Perplexity: {:.2}",
        ppl_metrics.perplexity.unwrap_or(f64::NAN)
    );
    println!(
        "  Avg LL: {:.4}",
        ppl_metrics.avg_log_likelihood.unwrap_or(f64::NAN)
    );

    // ── 4. Multi-Task Aggregation ──
    println!("\n── 4. Multi-Task Report ──");
    let mc_task = mock_hellaswag();
    let mut cls_task = EvalTask::new("sentiment", TaskType::Classification);
    cls_task.add_example(EvalExample {
        context: "This movie was great".to_string(),
        choices: vec![" positive".to_string(), " negative".to_string()],
        gold_idx: Some(0),
        reference: None,
    });
    cls_task.add_example(EvalExample {
        context: "Terrible experience".to_string(),
        choices: vec![" positive".to_string(), " negative".to_string()],
        gold_idx: Some(1),
        reference: None,
    });

    let multi_config = HarnessConfig {
        tasks: vec![mc_task, cls_task],
        length_normalize: false,
        max_examples: 0,
    };

    let multi_report = run_harness(&multi_config, |ctx, completion| {
        if ctx.contains("sandwich") && completion.contains("butter") {
            0.0
        } else if ctx.contains("cat") && completion.contains("purred") {
            0.0
        } else if ctx.contains("great") && completion.contains("positive") {
            0.0
        } else if ctx.contains("Terrible") && completion.contains("negative") {
            0.0
        } else {
            -5.0
        }
    })
    .unwrap();

    println!("  Tasks evaluated: {}", multi_report.num_tasks);
    println!("  Total examples:  {}", multi_report.total_examples);
    println!(
        "  Macro accuracy:  {:.1}%",
        multi_report.macro_accuracy * 100.0
    );
    for t in &multi_report.tasks {
        println!(
            "    {}: acc={:.1}%",
            t.task_name,
            t.accuracy.unwrap_or(0.0) * 100.0
        );
    }

    println!("\n=== Eval harness verified ===");
}
