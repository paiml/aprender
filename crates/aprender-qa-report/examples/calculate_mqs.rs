//! Example: Calculating Model Qualification Score (MQS)
//!
//! This example demonstrates how to calculate MQS scores from test evidence
//! using the Popperian falsification methodology.
//!
//! Run with: `cargo run --example calculate_mqs -p apr-qa-report`

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_sign_loss)]

use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use apr_qa_report::MqsCalculator;
use apr_qa_runner::{Evidence, EvidenceCollector};

fn main() {
    let model_id = "meta-llama/Llama-3.2-1B-Instruct";
    let model = ModelId::new("meta-llama", "Llama-3.2-1B-Instruct");

    // Create sample evidence from test runs
    let mut collector = EvidenceCollector::new();

    // Gateway tests (G1-G4) - critical infrastructure tests
    add_gateway_evidence(&mut collector, &model);

    // Functional tests (F-*) - capability tests
    add_functional_evidence(&mut collector, &model);

    // Performance tests (P-*) - speed/throughput tests
    add_performance_evidence(&mut collector, &model);

    // Calculate MQS score
    let calculator = MqsCalculator::new();

    match calculator.calculate(model_id, &collector) {
        Ok(score) => {
            println!("Model Qualification Score (MQS) Report");
            println!("======================================");
            println!();
            println!("Model: {}", score.model_id);
            println!("Grade: {}", score.grade);
            println!();
            println!("Gateway Status:");
            for gateway in &score.gateways {
                let status = if gateway.passed { "PASS" } else { "FAIL" };
                println!("  [{}] {}: {}", status, gateway.id, gateway.description);
            }
            println!();
            println!("Score Breakdown:");
            println!("  Raw Score:         {}", score.raw_score);
            println!("  Normalized Score:  {:.1}%", score.normalized_score);
            println!(
                "  Tests Passed:      {}/{}",
                score.tests_passed, score.total_tests
            );
            println!(
                "  Qualification:     {}",
                if score.qualifies() {
                    "QUALIFIED"
                } else {
                    "NOT QUALIFIED"
                }
            );
            println!();

            // Calculate Popperian score for philosophical context
            let popperian_calc = apr_qa_report::popperian::PopperianCalculator::new();
            let popperian = popperian_calc.calculate(model_id, &collector);

            println!("Popperian Analysis:");
            println!("  Corroborated:      {}", popperian.corroborated);
            println!("  Falsified:         {}", popperian.falsified);
            println!("  Inconclusive:      {}", popperian.inconclusive);
            println!(
                "  Corroboration:     {:.1}%",
                popperian.corroboration_ratio * 100.0
            );
            println!("  Black Swans:       {}", popperian.black_swan_count);
            println!();
            println!("Summary: {}", popperian.falsification_summary());
        }
        Err(e) => {
            eprintln!("Error calculating MQS: {e}");
        }
    }
}

fn add_gateway_evidence(collector: &mut EvidenceCollector, model: &ModelId) {
    // G1: Model loads
    collector.add(Evidence::corroborated(
        "G1-LOAD",
        create_scenario(model, "Loading model..."),
        "Model loaded in 2.5s",
        2500,
    ));

    // G2: Basic inference
    collector.add(Evidence::corroborated(
        "G2-INFER",
        create_scenario(model, "Hello, world!"),
        "Generated response: Hello! How can I help you today?",
        150,
    ));

    // G3: No crashes
    collector.add(Evidence::corroborated(
        "G3-STABLE",
        create_scenario(model, "Test stability"),
        "Process completed with exit code 0",
        100,
    ));

    // G4: Output not garbage
    collector.add(Evidence::corroborated(
        "G4-VALID",
        create_scenario(model, "What is 2+2?"),
        "The answer is 4.",
        80,
    ));
}

fn add_functional_evidence(collector: &mut EvidenceCollector, model: &ModelId) {
    // Add 10 functional tests with mostly passing results
    for i in 0..10 {
        let gate_id = format!("F-FUNC-{:03}", i + 1);
        let prompt = format!("Functional test prompt {}", i + 1);

        if i < 9 {
            // 90% pass rate
            collector.add(Evidence::corroborated(
                gate_id,
                create_scenario(model, &prompt),
                "Test passed successfully",
                100 + i as u64 * 10,
            ));
        } else {
            collector.add(Evidence::falsified(
                gate_id,
                create_scenario(model, &prompt),
                "Output quality below threshold",
                "Low quality output",
                200,
            ));
        }
    }
}

fn add_performance_evidence(collector: &mut EvidenceCollector, model: &ModelId) {
    // Add performance tests
    collector.add(Evidence::corroborated(
        "P-PERF-001",
        create_scenario(model, "Performance test 1"),
        "Achieved 25 tokens/sec",
        1000,
    ));

    collector.add(Evidence::corroborated(
        "P-PERF-002",
        create_scenario(model, "Performance test 2"),
        "TTFT: 150ms",
        150,
    ));
}

fn create_scenario(model: &ModelId, prompt: &str) -> QaScenario {
    QaScenario::new(
        model.clone(),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        prompt.to_string(),
        42,
    )
}
