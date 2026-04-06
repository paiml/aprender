//! Test remote worker on Mac Pro
//!
//! Run with: `cargo run --release --features remote --example test_mac_worker`

use repartir::executor::remote::RemoteExecutor;
use repartir::executor::Executor;
use repartir::task::{Backend, Task};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    tracing_subscriber::fmt::init();

    println!("Connecting to Mac Pro worker at 192.168.50.100:9000...");

    // Create remote executor and add worker
    let executor = RemoteExecutor::new().await?;
    executor.add_worker("192.168.50.100:9000").await?;

    println!("Connected! Executor capacity: {} workers", executor.capacity());

    // Create a simple task
    let task = Task::builder()
        .binary("/bin/echo")
        .arg("Hello from Mac Pro Xeon W-3245!")
        .backend(Backend::Remote)
        .build()?;

    println!("Submitting task...");

    // Execute task
    let result = executor.execute(task).await?;

    if result.is_success() {
        println!("Task succeeded!");
        println!("Output: {}", result.stdout_str()?.trim());
    } else {
        println!("Task failed!");
        println!("Exit code: {:?}", result.exit_code());
        println!("Stderr: {}", result.stderr_str()?);
    }

    Ok(())
}
