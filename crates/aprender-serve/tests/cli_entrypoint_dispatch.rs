//! `cli::entrypoint` dispatch: each arm reaches its handler and surfaces the handler's result.
//!
//! Complements the in-crate Info/Viz/Bench-list tests with the arms they leave
//! uncovered: List, BenchCompare (error propagation) and Run/Chat on a model
//! path that does not exist.
//!
//! Refs: #5707

use realizar::cli::{entrypoint, Cli, Commands};

const MISSING: &str = "/nonexistent-5707/model.gguf";

#[tokio::test]
async fn list_with_remote_returns_ok_without_touching_the_local_registry() {
    let cli = Cli {
        command: Commands::List {
            remote: Some("http://registry.invalid".to_string()),
            format: "table".to_string(),
        },
    };
    assert!(entrypoint(cli).await.is_ok());
}

#[tokio::test]
async fn bench_compare_propagates_the_read_error_naming_the_file() {
    let cli = Cli {
        command: Commands::BenchCompare {
            file1: "/nonexistent-5707/a.json".to_string(),
            file2: "/nonexistent-5707/b.json".to_string(),
            threshold: 5.0,
        },
    };
    let err = entrypoint(cli).await.expect_err("missing files must fail");
    assert!(
        err.to_string().contains("/nonexistent-5707/a.json"),
        "{err}"
    );
}

#[tokio::test]
async fn run_on_a_missing_local_model_is_an_error() {
    let cli = Cli {
        command: Commands::Run {
            model: MISSING.to_string(),
            prompt: Some("hi".to_string()),
            max_tokens: 1,
            temperature: 0.0,
            format: "text".to_string(),
            system: None,
            raw: true,
            gpu: false,
            verbose: false,
            trace: None,
        },
    };
    assert!(entrypoint(cli).await.is_err());
}

#[tokio::test]
async fn chat_on_a_missing_local_model_is_an_error() {
    let cli = Cli {
        command: Commands::Chat {
            model: MISSING.to_string(),
            system: None,
            history: None,
        },
    };
    assert!(entrypoint(cli).await.is_err());
}
