//! Classification witness: declared `needs-net`. The peer CI does not have.
fn main() {
    eprintln!(
        "ERROR Failed to connect to worker at 192.0.2.1:9000: Connection refused (os error 111)"
    );
    std::process::exit(1);
}
