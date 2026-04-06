/// Pipeline demonstration
/// Based on sovereign-ai-spec.md section 2.8 and section 11
use batuta::pipeline::{
    AnalysisStage, BuildStage, OptimizationStage, TranspilationPipeline, TranspilationStage,
    ValidationStage, ValidationStrategy,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🚀 Batuta Pipeline Demo");
    println!("Based on sovereign-ai-spec.md section 2.8\n");

    // Create pipeline with Jidoka (stop-on-error) validation
    let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
        .add_stage(Box::new(AnalysisStage))
        .add_stage(Box::new(TranspilationStage::new(true, true)))
        .add_stage(Box::new(OptimizationStage::new(false, true, 500)))
        .add_stage(Box::new(ValidationStage::new(false, false)))
        .add_stage(Box::new(BuildStage::new(false, None, false)));

    // Define input/output paths
    let input = PathBuf::from(".");
    let output = PathBuf::from("./target/transpiled");

    println!("Input: {:?}", input);
    println!("Output: {:?}\n", output);

    // Run pipeline
    println!("Running 5-phase pipeline...\n");

    match pipeline.run(&input, &output).await {
        Ok(result) => {
            println!("✅ Pipeline completed successfully!\n");
            println!("Output: {:?}", result.output_path);
            println!("Optimizations: {:?}", result.optimizations);
            println!("Validation: {}", if result.validation_passed { "✓" } else { "✗" });
        }
        Err(e) => {
            eprintln!("❌ Pipeline failed: {}", e);
            eprintln!("\nNote: This is expected if transpiler tools (depyler, decy, bashrs)");
            eprintln!("are not installed. Install them or run `batuta transpile` interactively.");
        }
    }

    Ok(())
}
