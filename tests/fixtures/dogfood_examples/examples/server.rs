//! Classification witness: `long-running`. Declared long-running in
//! Cargo.toml.in; stays up until the wrapper kills it, like a server or a TUI.
fn main() {
    println!("listening on 127.0.0.1:0 (fixture)");
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
