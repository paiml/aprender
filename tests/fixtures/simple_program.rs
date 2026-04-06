// Sprint 34: Simple test program for integration tests
//
// Generates basic syscalls for testing OTLP export

use std::fs;

fn main() {
    // Write to file (generates syscalls)
    fs::write("/tmp/renacer_test.txt", "Hello from test program!")
        .expect("Failed to write file");

    // Read from file
    let content = fs::read_to_string("/tmp/renacer_test.txt")
        .expect("Failed to read file");

    // Print to verify (generates write syscall)
    println!("Read: {}", content);

    // Clean up
    fs::remove_file("/tmp/renacer_test.txt")
        .expect("Failed to remove file");
}
