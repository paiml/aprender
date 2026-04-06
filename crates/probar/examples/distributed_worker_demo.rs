//! Distributed Worker Demo (PROBAR-SPEC-009-P10)
//!
//! Demonstrates distributed brick execution with:
//! - Work-stealing scheduler
//! - Data locality tracking
//! - Multi-backend dispatch (CPU/GPU/SIMD/Remote)
//! - PUB/SUB coordination
//!
//! Run with: cargo run --example distributed_worker_demo -p jugar-probar

use jugar_probar::brick::distributed::{
    Backend, BackendSelector, BrickCoordinator, BrickDataTracker, BrickInput, BrickMessage,
    DistributedBrick, MultiBrickExecutor, WorkStealingScheduler, WorkerId,
};
use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::sync::Arc;
use std::time::Duration;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      Distributed Worker Demo (PROBAR-SPEC-009-P10)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_backend_selection();
    demo_data_locality();
    demo_work_stealing();
    demo_pub_sub();
    demo_multi_backend_execution();
}

/// Demo brick for testing distributed execution
#[derive(Debug, Clone)]
struct MatMulBrick {
    name: &'static str,
    size: usize,
}

impl MatMulBrick {
    fn new(name: &'static str, size: usize) -> Self {
        Self { name, size }
    }
}

impl Brick for MatMulBrick {
    fn brick_name(&self) -> &'static str {
        self.name
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::default()
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![],
            failed: vec![],
            verification_time: Duration::ZERO,
        }
    }

    fn to_html(&self) -> String {
        format!("<div>MatMul {}x{}</div>", self.size, self.size)
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

/// Demonstrate backend selection based on input size
fn demo_backend_selection() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. Backend Selection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let selector = BackendSelector::new()
        .with_gpu_threshold(1_000_000) // 1M elements for GPU
        .with_simd_threshold(10_000) // 10K for SIMD
        .with_cpu_max_threshold(100_000_000); // 100M max local

    println!("  BackendSelector Thresholds:");
    println!("    ├─ GPU: >= 1,000,000 elements");
    println!("    ├─ SIMD: >= 10,000 elements");
    println!("    └─ CPU max: 100,000,000 elements\n");

    let test_cases = [
        (100, "Small tensor"),
        (5_000, "Medium tensor"),
        (50_000, "Large tensor"),
        (500_000, "Very large tensor"),
        (2_000_000, "Huge tensor"),
    ];

    println!("  Selection Results (GPU available=true):");
    println!("  ┌────────────────────┬────────────────────┬──────────┐");
    println!("  │ Description        │ Element Count      │ Backend  │");
    println!("  ├────────────────────┼────────────────────┼──────────┤");
    for (count, desc) in &test_cases {
        let backend = selector.select(*count, true);
        println!("  │ {:<18} │ {:>18} │ {:?}    │", desc, count, backend);
    }
    println!("  └────────────────────┴────────────────────┴──────────┘");

    println!("\n  Backend Availability:");
    println!("    ├─ CPU: {} (always)", Backend::Cpu.is_available());
    println!("    ├─ SIMD: {} (always)", Backend::Simd.is_available());
    println!(
        "    ├─ GPU: {} (requires feature)",
        Backend::Gpu.is_available()
    );
    println!(
        "    └─ Remote: {} (not implemented)",
        Backend::Remote.is_available()
    );

    println!("\n  Performance Estimates (relative):");
    println!("    ├─ GPU: {}", Backend::Gpu.performance_estimate());
    println!("    ├─ SIMD: {}", Backend::Simd.performance_estimate());
    println!("    ├─ CPU: {}", Backend::Cpu.performance_estimate());
    println!("    └─ Remote: {}", Backend::Remote.performance_estimate());
    println!();
}

/// Demonstrate data locality tracking
fn demo_data_locality() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. Data Locality Tracking");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let tracker = BrickDataTracker::new();

    // Simulate workers with different data
    let worker0 = WorkerId::new(0);
    let worker1 = WorkerId::new(1);
    let worker2 = WorkerId::new(2);

    println!("  Registering data locations:");
    println!("    Worker 0: model_weights (100MB), embeddings (50MB)");
    println!("    Worker 1: model_weights (100MB), cache (20MB)");
    println!("    Worker 2: embeddings (50MB), cache (20MB)");
    println!();

    // Track data locations
    tracker.track_data("model_weights", worker0, 100 * 1024 * 1024);
    tracker.track_data("model_weights", worker1, 100 * 1024 * 1024);
    tracker.track_data("embeddings", worker0, 50 * 1024 * 1024);
    tracker.track_data("embeddings", worker2, 50 * 1024 * 1024);
    tracker.track_data("cache", worker1, 20 * 1024 * 1024);
    tracker.track_data("cache", worker2, 20 * 1024 * 1024);

    // Query data locations
    println!("  Data Location Queries:");
    for key in &["model_weights", "embeddings", "cache"] {
        let workers = tracker.get_workers_for_data(key);
        let worker_ids: Vec<_> = workers.iter().map(|w| w.0).collect();
        println!("    {} → workers {:?}", key, worker_ids);
    }
    println!();

    // Calculate affinity for a task needing model_weights + embeddings
    let dependencies = vec!["model_weights".into(), "embeddings".into()];
    let affinity = tracker.calculate_affinity(&dependencies);

    println!("  Affinity Scores (for model_weights + embeddings):");
    let mut sorted: Vec<_> = affinity.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (worker, score) in sorted {
        println!("    Worker {}: {:.2}", worker.0, score);
    }
    println!("    → Best worker: 0 (has both datasets)");
    println!();
}

/// Demonstrate work-stealing scheduler
fn demo_work_stealing() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Work-Stealing Scheduler");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let data_tracker = Arc::new(BrickDataTracker::new());
    let scheduler = WorkStealingScheduler::new(Arc::clone(&data_tracker));

    println!("  Scheduler Configuration:");
    println!("    ├─ Data locality aware: true");
    println!("    └─ Work-stealing enabled: true");
    println!();

    // Create distributed bricks
    let brick1 = MatMulBrick::new("encoder", 512);
    let brick2 = MatMulBrick::new("decoder", 512);
    let brick3 = MatMulBrick::new("attention", 64);

    let dist1 = DistributedBrick::new(brick1)
        .with_backend(Backend::Simd)
        .with_data_dependencies(vec!["model_weights".into()]);

    let dist2 = DistributedBrick::new(brick2)
        .with_backend(Backend::Simd)
        .with_preferred_worker(WorkerId::new(1));

    let dist3 = DistributedBrick::new(brick3).with_backend(Backend::Cpu);

    // Submit tasks with TaskSpec
    let task_id1 = scheduler.submit(dist1.to_task_spec(), "input_1".into());
    let task_id2 = scheduler.submit_priority(dist2.to_task_spec(), "input_2".into(), 10);
    let task_id3 = scheduler.submit(dist3.to_task_spec(), "input_3".into());

    println!("  Submitted Tasks:");
    println!(
        "    ├─ Task {}: encoder (SIMD, deps: model_weights)",
        task_id1
    );
    println!("    ├─ Task {}: decoder (SIMD, priority 10)", task_id2);
    println!("    └─ Task {}: attention (CPU)", task_id3);
    println!();

    // Show scheduler statistics
    let stats = scheduler.stats();
    println!("  Scheduler Statistics:");
    println!("    ├─ Total submitted: {}", stats.total_submitted);
    println!("    ├─ Total completed: {}", stats.total_completed);
    println!("    └─ Total stolen: {}", stats.total_stolen);
    println!();

    // Explain work stealing
    println!("  Work Stealing Algorithm:");
    println!("    1. Each worker has a local deque (double-ended queue)");
    println!("    2. Workers pop from their own queue (LIFO for locality)");
    println!("    3. Idle workers steal from others (FIFO for fairness)");
    println!("    4. Data locality scores influence steal targets");
    println!();
}

/// Demonstrate PUB/SUB coordination
fn demo_pub_sub() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. PUB/SUB Coordination");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let coordinator = BrickCoordinator::new();

    // Subscribe to events
    let weight_sub = coordinator.subscribe("weights");
    let state_sub = coordinator.subscribe("state");

    println!("  Subscriptions:");
    println!("    ├─ Topic: {} (for weight updates)", weight_sub.topic());
    println!("    └─ Topic: {} (for state changes)", state_sub.topic());
    println!();

    // Publish some messages
    coordinator.publish(
        "weights",
        BrickMessage::WeightUpdate {
            brick_name: "encoder".into(),
            weights: vec![0u8; 100], // Simulated weights
            version: 1,
        },
    );

    coordinator.publish(
        "state",
        BrickMessage::StateChange {
            brick_name: "decoder".into(),
            event: "model_loaded".into(),
        },
    );

    coordinator.publish(
        "weights",
        BrickMessage::WeightUpdate {
            brick_name: "attention".into(),
            weights: vec![0u8; 50],
            version: 1,
        },
    );

    println!("  Published Messages:");
    println!("    1. WeightUpdate(encoder, v1)");
    println!("    2. StateChange(decoder, model_loaded)");
    println!("    3. WeightUpdate(attention, v1)");
    println!();

    // Drain messages
    let weight_msgs = weight_sub.drain();
    let state_msgs = state_sub.drain();

    println!("  Received Messages:");
    println!("    weights topic: {} messages", weight_msgs.len());
    for msg in &weight_msgs {
        if let BrickMessage::WeightUpdate {
            brick_name,
            version,
            ..
        } = msg
        {
            println!("      • {} v{}", brick_name, version);
        }
    }
    println!("    state topic: {} messages", state_msgs.len());
    for msg in &state_msgs {
        if let BrickMessage::StateChange { brick_name, event } = msg {
            println!("      • {} → {}", brick_name, event);
        }
    }
    println!();
}

/// Demonstrate multi-backend execution
fn demo_multi_backend_execution() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. Multi-Backend Execution");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let data_tracker = Arc::new(BrickDataTracker::new());
    let executor = MultiBrickExecutor::new(Arc::clone(&data_tracker)).with_gpu_available(false);

    // Create test brick
    let brick = MatMulBrick::new("test_matmul", 256);

    // Test with different input sizes
    let test_inputs = [
        (100, "Small (100 elements)"),
        (20_000, "Medium (20K elements)"),
        (100_000, "Large (100K elements)"),
    ];

    println!("  Executing brick with different input sizes:\n");

    for (size, desc) in &test_inputs {
        let input = BrickInput::new(vec![1.0f32; *size], vec![*size]);

        let result = executor.execute(&brick, input);

        match result {
            Ok(output) => {
                println!("    {}:", desc);
                println!("      ├─ Backend: {:?}", output.metrics.backend);
                println!(
                    "      ├─ Execution time: {:?}",
                    output.metrics.execution_time
                );
                println!("      └─ Output size: {} elements", output.data.len());
            }
            Err(e) => {
                println!("    {}: Error - {:?}", desc, e);
            }
        }
        println!();
    }

    // Demonstrate distributed brick execution
    println!("  Distributed Brick Execution:");
    let dist_brick = DistributedBrick::new(MatMulBrick::new("distributed_matmul", 512))
        .with_backend(Backend::Simd)
        .with_data_dependencies(vec!["encoder_weights".into()])
        .with_preferred_worker(WorkerId::new(0));

    // Track that worker 0 has the weights
    data_tracker.track_weights("distributed_matmul", WorkerId::new(0));

    let input = BrickInput::new(vec![1.0f32; 50_000], vec![50_000]);
    let result = executor.execute_distributed(&dist_brick, input);

    match result {
        Ok(output) => {
            println!("    ├─ Backend: {:?}", output.metrics.backend);
            println!("    ├─ Worker: {:?}", output.metrics.worker_id);
            println!("    └─ Execution time: {:?}", output.metrics.execution_time);
        }
        Err(e) => {
            println!("    Error: {:?}", e);
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
