//! Contract Pipeline Demo
//!
//! Demonstrates the provable-contracts pipeline:
//!   1. YAML contracts define equations with preconditions/postconditions
//!   2. build.rs reads contracts/*.yaml and emits env vars
//!   3. #[contract] proc macro injects debug_assert!() from those env vars
//!   4. Runtime (debug builds) enforces contract assertions automatically
//!
//! Run: cargo run --example contract_pipeline_demo

fn main() {
    println!("=== Simular Contract Pipeline Demo ===\n");

    // Show which contracts are wired up via build.rs env vars
    println!("Contract bindings (from contracts/*.yaml via build.rs):");
    println!(
        "  gradient-v1 / gradient_clipping  -> engine::jidoka::PreFlightCheck::check_gradient_norm"
    );
    println!("  checkpoint-v1 / checkpoint_roundtrip -> replay::CheckpointManager::checkpoint");
    println!("  loss-functions-v1 / mse_loss     -> domains::ml::TrainingSimulator::step");
    println!();

    // Show the env vars that build.rs emits (available at compile time)
    println!("Build-time env vars emitted by build.rs:");
    if let Some(v) = option_env!("CONTRACT_GRADIENT_V1_GRADIENT_CLIPPING_PRE_COUNT") {
        println!("  CONTRACT_GRADIENT_V1_GRADIENT_CLIPPING_PRE_COUNT = {v}");
    }
    if let Some(v) = option_env!("CONTRACT_GRADIENT_V1_GRADIENT_CLIPPING_PRE_0") {
        println!("  CONTRACT_GRADIENT_V1_GRADIENT_CLIPPING_PRE_0     = {v}");
    }
    if let Some(v) = option_env!("CONTRACT_CHECKPOINT_V1_CHECKPOINT_ROUNDTRIP_PRE_COUNT") {
        println!("  CONTRACT_CHECKPOINT_V1_CHECKPOINT_ROUNDTRIP_PRE_COUNT = {v}");
    }
    if let Some(v) = option_env!("CONTRACT_CHECKPOINT_V1_CHECKPOINT_ROUNDTRIP_PRE_0") {
        println!("  CONTRACT_CHECKPOINT_V1_CHECKPOINT_ROUNDTRIP_PRE_0     = {v}");
    }
    if let Some(v) = option_env!("CONTRACT_LOSS_FUNCTIONS_V1_MSE_LOSS_PRE_COUNT") {
        println!("  CONTRACT_LOSS_FUNCTIONS_V1_MSE_LOSS_PRE_COUNT = {v}");
    }
    if let Some(v) = option_env!("CONTRACT_LOSS_FUNCTIONS_V1_MSE_LOSS_PRE_0") {
        println!("  CONTRACT_LOSS_FUNCTIONS_V1_MSE_LOSS_PRE_0     = {v}");
    }
    if let Some(v) = option_env!("CONTRACT_LOSS_FUNCTIONS_V1_MSE_LOSS_PRE_1") {
        println!("  CONTRACT_LOSS_FUNCTIONS_V1_MSE_LOSS_PRE_1     = {v}");
    }
    println!();

    println!("Pipeline flow:");
    println!("  contracts/*.yaml");
    println!("    -> build.rs emit_contract_assertions()");
    println!("    -> cargo:rustc-env=CONTRACT_*_PRE_N=<rust expr>");
    println!("    -> #[contract(\"yaml-stem\", equation = \"eq\")]");
    println!("    -> proc macro reads env vars at compile time");
    println!("    -> debug_assert!(<rust expr>) injected into function body");
    println!();
    println!("In debug builds, preconditions are checked at every call site.");
    println!("In release builds, debug_assert! compiles to nothing (zero cost).");
    println!();
    println!("=== Demo complete ===");
}
