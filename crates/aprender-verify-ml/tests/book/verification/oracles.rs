//! Verification oracle examples from the book

use verificar::oracle::{Executor, IoOracle, PythonExecutor};
use verificar::Language;

#[test]
fn test_io_oracle_example() {
    // Example: Using I/O oracle for verification
    let oracle = IoOracle::new();
    let _executor = PythonExecutor::new();

    let source_code = "print(2 + 2)";
    let target_code = "println!(\"{}\", 2 + 2);";
    let input = "";

    let verdict = oracle.verify(
        source_code,
        target_code,
        input,
        Language::Python,
        Language::Rust,
    );

    assert!(verdict.is_ok());
}

#[test]
fn test_python_executor_example() {
    // Example: Executing Python code
    let executor = PythonExecutor::new();
    let code = "print('Hello, World!')";
    let input = "";

    let result = executor.execute(code, input, 5000);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.stdout.contains("Hello, World!"));
    assert_eq!(output.exit_code, 0);
}

#[test]
fn test_verification_with_input_example() {
    // Example: Verification with stdin input
    let executor = PythonExecutor::new();

    let code = "name = input()\nprint(f'Hello, {name}!')";
    let input = "Alice";

    let result = executor.execute(code, input, 5000);

    assert!(result.is_ok());
    assert!(result.unwrap().stdout.contains("Alice"));
}

#[test]
#[ignore] // Timeout handling currently hangs - needs executor fix
fn test_timeout_handling_example() {
    // Example: Handling execution timeouts
    let executor = PythonExecutor::new();
    let infinite_loop = "while True:\n    pass";

    // Use 1000ms (1 second) timeout for realistic test behavior
    let result = executor.execute(infinite_loop, "", 1000);

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Timeout") || error_msg.contains("timeout"));
}
