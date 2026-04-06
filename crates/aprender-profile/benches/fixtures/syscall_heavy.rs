/// Benchmark fixture: syscall-heavy workload
/// Performs many file system operations to generate syscalls
use std::fs;
use std::io::Write;

fn main() {
    let temp_dir = std::env::temp_dir();

    // Perform 100 file operations (write + read + remove)
    for i in 0..100 {
        let path = temp_dir.join(format!("bench_file_{}.txt", i));

        // Write (generates open, write, close syscalls)
        let mut file = fs::File::create(&path).expect("Failed to create file");
        file.write_all(b"benchmark data\n").expect("Failed to write");
        drop(file);

        // Read (generates open, read, close syscalls)
        let _content = fs::read_to_string(&path).expect("Failed to read");

        // Remove (generates unlink syscall)
        fs::remove_file(&path).expect("Failed to remove");
    }
}
