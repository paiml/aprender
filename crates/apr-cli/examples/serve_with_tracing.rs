//! Example: Using apr serve with X-Trace-Level header for debugging
//!
//! This example demonstrates how to use the apr serve endpoint with tracing
//! enabled to debug inference performance.
//!
//! # Running the server
//! ```bash
//! apr serve /path/to/model.gguf --port 8080
//! ```
//!
//! # Using tracing headers
//! ```bash
//! # Brick-level tracing (token operations)
//! curl -X POST http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -H "X-Trace-Level: brick" \
//!   -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}'
//!
//! # Step-level tracing (forward pass steps)
//! curl -X POST http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -H "X-Trace-Level: step" \
//!   -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}'
//!
//! # Layer-level tracing (per-layer timing)
//! curl -X POST http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -H "X-Trace-Level: layer" \
//!   -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}'
//! ```
//!
//! # Trace Response Format
//!
//! When tracing is enabled, the response includes additional trace fields:
//! ```json
//! {
//!   "id": "chatcmpl-...",
//!   "choices": [...],
//!   "usage": {...},
//!   "brick_trace": {
//!     "level": "brick",
//!     "operations": 5,
//!     "total_time_us": 12345,
//!     "breakdown": [
//!       {"name": "token_0", "time_us": 2469},
//!       {"name": "token_1", "time_us": 2469},
//!       ...
//!     ]
//!   }
//! }
//! ```
//!
//! # Without tracing
//!
//! When no X-Trace-Level header is provided, the response is a clean
//! OpenAI-compatible format with no trace fields.

use std::process::Command;

fn main() {
    println!("APR Serve Tracing Example");
    println!("=========================");
    println!();
    println!("This example demonstrates the X-Trace-Level header for debugging inference.");
    println!();
    println!("Start the server:");
    println!("  apr serve /path/to/model.gguf --port 8080");
    println!();
    println!("Then use one of these trace levels:");
    println!();
    println!("1. Brick-level (token operations):");
    println!(r#"   curl -H "X-Trace-Level: brick" http://localhost:8080/v1/chat/completions ..."#);
    println!();
    println!("2. Step-level (forward pass steps):");
    println!(r#"   curl -H "X-Trace-Level: step" http://localhost:8080/v1/chat/completions ..."#);
    println!();
    println!("3. Layer-level (per-layer timing):");
    println!(r#"   curl -H "X-Trace-Level: layer" http://localhost:8080/v1/chat/completions ..."#);
    println!();

    // Check if server is running
    let health = Command::new("curl")
        .args(["-s", "http://localhost:8080/health"])
        .output();

    match health {
        Ok(output) if output.status.success() => {
            let response = String::from_utf8_lossy(&output.stdout);
            println!("Server health: {response}");
        }
        _ => {
            println!("Server not running. Start with: apr serve <model> --port 8080");
        }
    }
}
