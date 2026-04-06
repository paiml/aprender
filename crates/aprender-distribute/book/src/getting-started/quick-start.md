# Quick Start

This guide will walk you through creating your first distributed computing application with Repartir in under 5 minutes.

## Setup

Create a new Rust project:

```bash
cargo new repartir-demo
cd repartir-demo
```

Add dependencies to `Cargo.toml`:

```toml
[dependencies]
repartir = "1.0"
tokio = { version = "1.35", features = ["rt-multi-thread", "macros"] }
```

## CPU Task Example

Replace `src/main.rs` with:

```rust
use repartir::executor::{CpuExecutor, Executor};
use repartir::task::{Task, Backend};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Create CPU executor
    let executor = CpuExecutor::new();
    println!("CPU Executor: {} cores", executor.capacity());

    // Create a task
    let task = Task::builder()
        .binary("/usr/bin/uname")
        .args(vec!["-a".to_string()])
        .backend(Backend::Cpu)
        .build()?;

    // Execute and get result
    let result = executor.execute(task).await?;

    println!("Exit code: {}", result.exit_code());
    println!("Output: {}", result.stdout_str()?);
    println!("Duration: {:?}", result.duration());

    Ok(())
}
```

Run it:

```bash
cargo run
```

Output:

```
CPU Executor: 8 cores
Exit code: 0
Output: Linux hostname 6.8.0-47-generic #47-Ubuntu SMP PREEMPT_DYNAMIC ...
Duration: 12ms
```

## Using the Scheduler

The scheduler manages multiple tasks with priorities:

```rust
use repartir::{Scheduler, Task, Backend, Priority};
use repartir::executor::{CpuExecutor, Executor};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    let scheduler = Scheduler::new();
    let executor = CpuExecutor::new();

    // Submit high-priority task
    let high_task = Task::builder()
        .binary("/bin/echo")
        .arg("High priority")
        .priority(Priority::High)
        .build()?;

    let high_id = scheduler.submit(high_task).await?;

    // Submit normal-priority task
    let normal_task = Task::builder()
        .binary("/bin/echo")
        .arg("Normal priority")
        .priority(Priority::Normal)
        .build()?;

    let normal_id = scheduler.submit(normal_task).await?;

    // Execute in priority order
    while let Some(task) = scheduler.next().await {
        let task_id = task.id();
        let result = executor.execute(task).await?;

        scheduler.store_result(task_id, result).await?;
        println!("Completed task {}: {}",
            task_id,
            scheduler.get_result(&task_id).await?.stdout_str()?
        );
    }

    Ok(())
}
```

## Next Steps

- **[First Task](./first-task.md)**: Deep dive into the Task model
- **[Core Concepts](./core-concepts.md)**: Understand executors, schedulers, and messaging
- **[CPU Executor](../executors/cpu.md)**: Learn CPU execution in detail
- **[GPU Executor](../executors/gpu.md)**: Run compute shaders on GPU
