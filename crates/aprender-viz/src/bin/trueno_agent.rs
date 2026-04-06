//! trueno-agent - Remote monitoring agent for multi-system support.
//!
//! This agent runs on remote nodes and reports metrics back to the leader.

fn main() {
    eprintln!("trueno-agent requires the 'monitor-remote' feature.");
    eprintln!("Build with: cargo build --features monitor-remote");
    std::process::exit(1);
}
