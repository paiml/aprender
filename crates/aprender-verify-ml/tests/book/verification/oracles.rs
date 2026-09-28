//! Verification oracle examples from the book

use verificar::oracle::{Executor, IoOracle, PythonExecutor};
use verificar::Language;

/// The book examples run a real `python3`. The sovereign-ci test image has none, so
/// there `execute` fails at spawn and the example asserted on it (#4554). Skip with
/// the reason on stderr, the same guard the executor's own unit tests use, so the
/// examples still run wherever an interpreter exists.
fn python_or_skip(test: &str) -> Option<PythonExecutor> {
    let executor = PythonExecutor::new();
    if executor.is_available() {
        Some(executor)
    } else {
        eprintln!("SKIP {test}: python3 is not available on this host");
        None
    }
}

#[test]
fn test_io_oracle_example() {
    // Example: Using I/O oracle for verification
    let oracle = IoOracle::new();
    let Some(_executor) = python_or_skip("test_io_oracle_example") else {
        return;
    };

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

    verdict.expect("I/O oracle verdict");
}

#[test]
fn test_python_executor_example() {
    // Example: Executing Python code
    let Some(executor) = python_or_skip("test_python_executor_example") else {
        return;
    };
    let code = "print('Hello, World!')";
    let input = "";

    let output = executor
        .execute(code, input, 5000)
        .expect("python3 runs the example");
    assert!(output.stdout.contains("Hello, World!"));
    assert_eq!(output.exit_code, 0);
}

#[test]
fn test_verification_with_input_example() {
    // Example: Verification with stdin input
    let Some(executor) = python_or_skip("test_verification_with_input_example") else {
        return;
    };

    let code = "name = input()\nprint(f'Hello, {name}!')";
    let input = "Alice";

    let output = executor
        .execute(code, input, 5000)
        .expect("python3 runs the example");
    assert!(output.stdout.contains("Alice"));
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
