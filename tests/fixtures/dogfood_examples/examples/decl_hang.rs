//! Classification witness: `timeout`. Declared needs-data, but hangs. Only
//! `long-running` may turn a kill into a skip.
fn main() {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
