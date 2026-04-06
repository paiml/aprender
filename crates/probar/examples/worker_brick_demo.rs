//! WorkerBrick Code Generation Demo (PROBAR-SPEC-009-P7)
//!
//! Demonstrates automatic generation of Web Worker code from Rust brick definitions:
//! - Zero hand-written JavaScript
//! - Type-safe message protocols
//! - State machine validation
//! - TypeScript type generation
//!
//! Run with: cargo run --example worker_brick_demo -p jugar-probar

use jugar_probar::brick::worker::{
    BrickWorkerMessage, BrickWorkerMessageDirection, FieldType, WorkerBrick,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     WorkerBrick Code Generation Demo (PROBAR-SPEC-009-P7)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_audio_worker();
    demo_transcription_worker();
    demo_game_physics_worker();
    demo_typescript_types();
}

/// Demonstrate audio processing worker generation
fn demo_audio_worker() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. Audio Processing Worker");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Define audio worker with SharedArrayBuffer for streaming
    let audio_worker = WorkerBrick::new("audio_processor")
        // Messages TO the worker
        .message(
            BrickWorkerMessage::new("init", BrickWorkerMessageDirection::ToWorker)
                .field("sampleRate", FieldType::Number)
                .field("bufferSize", FieldType::Number)
                .field("sharedBuffer", FieldType::SharedArrayBuffer),
        )
        .message(
            BrickWorkerMessage::new("process", BrickWorkerMessageDirection::ToWorker)
                .field("inputBuffer", FieldType::Float32Array)
                .optional_field("gain", FieldType::Number),
        )
        .message(BrickWorkerMessage::new(
            "stop",
            BrickWorkerMessageDirection::ToWorker,
        ))
        // Messages FROM the worker
        .message(BrickWorkerMessage::new(
            "ready",
            BrickWorkerMessageDirection::FromWorker,
        ))
        .message(
            BrickWorkerMessage::new("processed", BrickWorkerMessageDirection::FromWorker)
                .field("outputBuffer", FieldType::Float32Array)
                .field("peakLevel", FieldType::Number),
        )
        .message(
            BrickWorkerMessage::new("error", BrickWorkerMessageDirection::FromWorker)
                .field("message", FieldType::String),
        )
        // State machine transitions
        .transition("uninitialized", "init", "initializing")
        .transition("initializing", "ready", "ready")
        .transition("ready", "process", "processing")
        .transition("processing", "processed", "ready")
        .transition("ready", "stop", "stopped")
        .transition("processing", "error", "error");

    println!("  Worker Definition:");
    println!("    ├─ Name: audio_processor");
    println!(
        "    ├─ ToWorker messages: {}",
        audio_worker.to_worker_messages().len()
    );
    println!(
        "    └─ FromWorker messages: {}",
        audio_worker.from_worker_messages().len()
    );

    println!("\n  State Machine:");
    println!("    uninitialized ─init─► initializing ─ready─► ready");
    println!("                                                 │");
    println!("                                      ┌──process─┘");
    println!("                                      ▼");
    println!("                            stopped ◄─stop── ready");
    println!("                                      │");
    println!("                            error ◄───┴─── processing");

    // Generate JavaScript
    let js_code = audio_worker.to_worker_js();
    println!("\n  Generated JavaScript ({} chars):", js_code.len());
    println!("  ┌─────────────────────────────────────────────────────────┐");
    for (i, line) in js_code.lines().take(12).enumerate() {
        let truncated = if line.len() > 55 {
            format!("{}...", &line[..52])
        } else {
            line.to_string()
        };
        println!("  │ {:2}. {:<53} │", i + 1, truncated);
    }
    println!(
        "  │     ... ({} more lines)                               │",
        js_code.lines().count() - 12
    );
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();
}

/// Demonstrate transcription worker (whisper.apr style)
fn demo_transcription_worker() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. Transcription Worker (whisper.apr style)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let transcription_worker = WorkerBrick::new("transcription")
        // Initialize with model URL
        .message(
            BrickWorkerMessage::new("load_model", BrickWorkerMessageDirection::ToWorker)
                .field("modelUrl", FieldType::String)
                .field("quantization", FieldType::String),
        )
        // Transcribe audio chunk
        .message(
            BrickWorkerMessage::new("transcribe", BrickWorkerMessageDirection::ToWorker)
                .field("audioData", FieldType::Float32Array)
                .field("language", FieldType::String)
                .optional_field("timestamp", FieldType::Number),
        )
        // Cancel current transcription
        .message(BrickWorkerMessage::new(
            "cancel",
            BrickWorkerMessageDirection::ToWorker,
        ))
        // Model loaded response
        .message(
            BrickWorkerMessage::new("model_loaded", BrickWorkerMessageDirection::FromWorker)
                .field("vocabSize", FieldType::Number)
                .field("modelSize", FieldType::Number),
        )
        // Transcription result
        .message(
            BrickWorkerMessage::new("transcript", BrickWorkerMessageDirection::FromWorker)
                .field("text", FieldType::String)
                .field("confidence", FieldType::Number)
                .optional_field("segments", FieldType::String),
        )
        // Progress update
        .message(
            BrickWorkerMessage::new("progress", BrickWorkerMessageDirection::FromWorker)
                .field("percent", FieldType::Number)
                .field("stage", FieldType::String),
        )
        // State transitions
        .transition("uninitialized", "load_model", "loading")
        .transition("loading", "model_loaded", "ready")
        .transition("ready", "transcribe", "transcribing")
        .transition("transcribing", "transcript", "ready")
        .transition("transcribing", "cancel", "ready");

    // Generate Rust bindings
    let rust_code = transcription_worker.to_rust_bindings();
    println!("  Generated Rust Bindings ({} chars):", rust_code.len());
    println!("  ┌─────────────────────────────────────────────────────────┐");
    for (i, line) in rust_code.lines().take(20).enumerate() {
        let truncated = if line.len() > 55 {
            format!("{}...", &line[..52])
        } else {
            line.to_string()
        };
        println!("  │ {:2}. {:<53} │", i + 1, truncated);
    }
    println!(
        "  │     ... ({} more lines)                               │",
        rust_code.lines().count() - 20
    );
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();
}

/// Demonstrate game physics worker
fn demo_game_physics_worker() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Game Physics Worker");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let physics_worker = WorkerBrick::new("physics")
        // Initialize physics world
        .message(
            BrickWorkerMessage::new("init_world", BrickWorkerMessageDirection::ToWorker)
                .field("gravity", FieldType::Number)
                .field("worldSize", FieldType::Number),
        )
        // Add rigid body
        .message(
            BrickWorkerMessage::new("add_body", BrickWorkerMessageDirection::ToWorker)
                .field("id", FieldType::Number)
                .field("mass", FieldType::Number)
                .field("posX", FieldType::Number)
                .field("posY", FieldType::Number),
        )
        // Step simulation
        .message(
            BrickWorkerMessage::new("step", BrickWorkerMessageDirection::ToWorker)
                .field("deltaTime", FieldType::Number),
        )
        // World state update
        .message(
            BrickWorkerMessage::new("state_update", BrickWorkerMessageDirection::FromWorker)
                .field("positions", FieldType::Float32Array)
                .field("velocities", FieldType::Float32Array),
        )
        // Collision event
        .message(
            BrickWorkerMessage::new("collision", BrickWorkerMessageDirection::FromWorker)
                .field("bodyA", FieldType::Number)
                .field("bodyB", FieldType::Number)
                .field("impulse", FieldType::Number),
        )
        // State machine
        .transition("uninitialized", "init_world", "ready")
        .transition("ready", "add_body", "ready")
        .transition("ready", "step", "simulating")
        .transition("simulating", "state_update", "ready");

    // Show TypeScript types
    let ts_types = physics_worker.to_typescript_defs();
    println!("  Generated TypeScript Types ({} chars):", ts_types.len());
    println!("  ┌─────────────────────────────────────────────────────────┐");
    for (i, line) in ts_types.lines().take(25).enumerate() {
        let truncated = if line.len() > 55 {
            format!("{}...", &line[..52])
        } else {
            line.to_string()
        };
        println!("  │ {:2}. {:<53} │", i + 1, truncated);
    }
    if ts_types.lines().count() > 25 {
        println!(
            "  │     ... ({} more lines)                               │",
            ts_types.lines().count() - 25
        );
    }
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();
}

/// Demonstrate TypeScript type generation
fn demo_typescript_types() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. Field Type Mappings");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Rust → JavaScript → TypeScript mappings:");
    println!();
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │ FieldType             │ TypeScript      │ Rust          │");
    println!("  ├─────────────────────────────────────────────────────────┤");

    let types = [
        FieldType::String,
        FieldType::Number,
        FieldType::Boolean,
        FieldType::SharedArrayBuffer,
        FieldType::Float32Array,
        FieldType::Optional(Box::new(FieldType::Number)),
    ];

    for ft in &types {
        println!(
            "  │ {:<21} │ {:<15} │ {:<13} │",
            format!("{:?}", ft),
            ft.to_typescript(),
            ft.to_rust()
        );
    }
    println!("  └─────────────────────────────────────────────────────────┘");

    println!("\n  Zero-JS Benefits:");
    println!("    ├─ All worker code generated from Rust specs");
    println!("    ├─ Type safety enforced at compile time");
    println!("    ├─ State machine prevents invalid transitions");
    println!("    ├─ Message protocol validated by types");
    println!("    └─ No manual JavaScript maintenance");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
