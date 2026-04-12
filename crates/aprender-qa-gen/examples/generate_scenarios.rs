//! Example: Generating QA Scenarios
//!
//! This example demonstrates how to use the `ScenarioGenerator` to create
//! test scenarios for model qualification testing.
//!
//! Run with: `cargo run --example generate_scenarios -p apr-qa-gen`

#![allow(clippy::missing_panics_doc)]

use apr_qa_gen::{Backend, Format, Modality, ModelId, ScenarioGenerator};

fn main() {
    // Create a model ID for a hypothetical model (org, name)
    let model = ModelId::new("meta-llama", "Llama-3.2-1B-Instruct");

    // Create a scenario generator with default settings
    let generator = ScenarioGenerator::new(model.clone());

    // Generate scenarios for a specific configuration
    let scenarios = generator.generate_for(Modality::Run, Backend::Cpu, Format::Gguf);

    println!(
        "Generated {} scenarios for {}:",
        scenarios.len(),
        model.hf_repo()
    );
    println!();

    // Display first 5 scenarios
    for (i, scenario) in scenarios.iter().take(5).enumerate() {
        println!("Scenario {}:", i + 1);
        println!("  ID:       {}", scenario.id);
        println!("  Modality: {:?}", scenario.modality);
        println!("  Backend:  {:?}", scenario.backend);
        println!("  Format:   {:?}", scenario.format);
        println!("  Prompt:   {}", scenario.prompt);
        println!("  Seed:     {}", scenario.seed);
        println!();
    }

    // Generate all scenarios (all modality/backend/format combinations)
    let all_scenarios = generator.generate();
    println!(
        "Total scenarios generated (all combinations): {}",
        all_scenarios.len()
    );
}
