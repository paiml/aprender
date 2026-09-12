//! Classification witness: `timeout`. Never returns, never prints, ignores
//! nothing — only the `timeout --signal=KILL` wrapper ends this process, which
//! is what makes it the mutation witness for the wrapper itself.
fn main() {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
