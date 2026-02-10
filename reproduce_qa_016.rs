use std::io::Write;
use tempfile::NamedTempFile;
use std::process::Command;

fn main() {
    // Create a temporary file with the "APRN" magic bytes
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    file.write_all(b"APRN").expect("write magic bytes");
    // Add truncated header to simulate corruption
    file.write_all(&[0u8; 10]).expect("write truncated header");

    let path = file.path().to_str().expect("valid UTF-8 path").to_string();
    
    // Execute apr validate
    let output = Command::new("cargo")
        .args(&["run", "-p", "apr-cli", "--", "validate", &path])
        .output()
        .expect("Failed to execute apr");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("STDOUT:\n{}", stdout);
    println!("STDERR:\n{}", stderr);
    println!("Exit Status: {}", output.status);

    if stdout.contains("FAIL") || stdout.contains("incomplete") {
        println!("Reproduction Successful: Output contains expected failure message.");
    } else {
        println!("Reproduction Failed: Output does not match expectation.");
    }
}
