//! Test remote worker on Mac Pro
//!
//! Run with: `cargo run --release --features remote --example test_mac_worker [-- <host:port>]`

use repartir::executor::remote::RemoteExecutor;
use repartir::executor::Executor;
use repartir::task::{Backend, Task};
use std::io::IsTerminal;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    tracing_subscriber::fmt::init();

    // A release build on a terminal keeps the Mac Pro default. A debug build or a run with
    // no terminal (CI, a pipe) needs the worker named: the default is one LAN host.
    let default_worker = (!cfg!(debug_assertions) && std::io::stdout().is_terminal())
        .then(|| "192.168.50.100:9000".to_string());
    let Some(worker) = std::env::args().nth(1).or(default_worker) else {
        eprintln!("Usage: test_mac_worker <host:port>   (e.g. 192.168.50.100:9000)");
        std::process::exit(2);
    };
    println!("Connecting to Mac Pro worker at {worker}...");

    // Create remote executor and add worker
    let executor = RemoteExecutor::new().await?;
    executor.add_worker(&worker).await?;

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
