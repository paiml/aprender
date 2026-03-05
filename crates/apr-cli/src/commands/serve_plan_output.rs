//! Text output formatter for `apr serve plan`.

use crate::commands::serve_plan::{PlanVerdict, ServePlan};

pub fn print_serve_plan_text(plan: &ServePlan, display_name: &str) {
    let quant = plan.model.quantization.as_deref().unwrap_or("unknown");

    let hw_label = match plan.hardware {
        Some(ref h) => h.gpu_name.clone(),
        None => "CPU".to_string(),
    };

    println!(
        "\n\u{2501}\u{2501}\u{2501} Inference Plan: {} ({}) on {} \u{2501}\u{2501}\u{2501}\n",
        plan.model.name, quant, hw_label,
    );

    // Model info
    println!("MODEL");
    println!(
        "  Parameters:      {:>14}",
        format_params(plan.model.params)
    );
    println!("  Quantization:    {:>14}", quant);
    println!("  Format:          {:>14}", plan.model.format);
    println!("  File size:       {:>11.0} MB", plan.model.file_size_mb);
    println!();

    // Memory/VRAM budget
    let budget_title = if plan.hardware.is_some() {
        "VRAM BUDGET"
    } else {
        "MEMORY BUDGET"
    };
    println!("{budget_title}");
    println!(
        "  Model weights ({quant}): {:>8.0} MB",
        plan.memory_budget.weights_mb,
    );
    println!(
        "  KV cache (batch={}, {} seq): {:>1.0} MB",
        plan.memory_budget.batch_size, plan.memory_budget.seq_len, plan.memory_budget.kv_cache_mb,
    );
    println!(
        "  Activations:          {:>8.0} MB",
        plan.memory_budget.activations_mb,
    );
    if plan.memory_budget.overhead_mb > 0.0 {
        println!(
            "  CUDA overhead:        {:>8.0} MB",
            plan.memory_budget.overhead_mb,
        );
    }
    println!("  {}", "\u{2500}".repeat(38));

    if let (Some(gpu_total), Some(util)) = (
        plan.memory_budget.gpu_total_mb,
        plan.memory_budget.utilization_pct,
    ) {
        println!(
            "  Total:           {:>8.0} MB / {:.0} MB  ({:.1}%)",
            plan.memory_budget.total_mb, gpu_total, util,
        );
    } else {
        println!("  Total:           {:>8.0} MB", plan.memory_budget.total_mb,);
    }

    if let Some(max_batch) = plan.memory_budget.max_batch {
        println!("  Max batch size:  {:>8}", max_batch);
    }
    println!();

    // Roofline analysis (GPU only)
    if let Some(ref roof) = plan.roofline {
        println!("ROOFLINE ANALYSIS");
        println!("  Memory bandwidth:    {:>8.0} GB/s", roof.bandwidth_gbps,);
        println!(
            "  Bandwidth ceiling:   {:>8.0} tok/s",
            roof.bandwidth_ceiling_tps,
        );
        println!("  Bottleneck:          {:>8}", roof.bottleneck);
        println!();
    }

    // Throughput estimate
    println!("THROUGHPUT ESTIMATE");
    if plan.hardware.is_none() {
        println!("  (DDR5 ~50 GB/s assumed)");
    }
    println!(
        "  Single decode:       ~{:.0} tok/s",
        plan.throughput.single_decode_tps,
    );
    if let Some(batched) = plan.throughput.batched_tps {
        println!(
            "  Batch={} projected:  ~{:.0} tok/s",
            plan.throughput.batch_size, batched,
        );
    }
    println!();

    // Contracts
    println!("CONTRACTS");
    for check in &plan.contracts {
        let icon = if check.passed { "\u{2713}" } else { "\u{2717}" };
        println!("  {icon} {}: {} ({})", check.id, check.name, check.detail);
    }
    println!();

    // Verdict
    let verdict_str = match plan.verdict {
        PlanVerdict::Ready => "READY",
        PlanVerdict::Warnings => "WARNINGS",
        PlanVerdict::Blocked => "BLOCKED",
    };
    println!("VERDICT: {verdict_str}");
    println!();

    // Next step hint
    let is_hf = plan.model.format == "HuggingFace";
    match plan.verdict {
        PlanVerdict::Ready | PlanVerdict::Warnings => {
            if is_hf {
                println!(
                    "Next: apr pull {display_name} && apr serve run <local_path> --gpu --batch"
                );
            } else if plan.hardware.is_some() {
                println!("Next: apr serve run {display_name} --gpu --batch");
            } else {
                println!("Next: apr serve run {display_name}");
            }
        }
        PlanVerdict::Blocked => {
            if let Some(max) = plan.memory_budget.max_batch {
                if max > 0 {
                    println!("Hint: reduce batch size to {max} or use a smaller quantization");
                } else {
                    println!(
                        "Hint: model exceeds available VRAM. Use CPU mode or a smaller model."
                    );
                }
            }
        }
    }
    println!();
}

fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.2}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1e6)
    } else {
        format!("{params}")
    }
}
