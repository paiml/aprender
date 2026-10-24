//! #5691: `run_batch_loop` turns every failing prompt into one JSONL error row
//! and counts it, never aborting the batch.

use super::{run_batch_loop, BatchInferenceConfig, BatchModel};
use std::path::PathBuf;

fn config() -> BatchInferenceConfig {
    BatchInferenceConfig {
        model_path: PathBuf::from("unused.gguf"),
        max_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        no_gpu: true,
        verbose: false,
        stop_tokens: vec![],
    }
}

/// A model with no backend: `generate` fails with "No model available".
fn empty_model() -> BatchModel {
    BatchModel {
        #[cfg(feature = "cuda")]
        gpu: None,
        #[cfg(feature = "gpu")]
        wgpu: None,
        cpu: None,
    }
}

fn run(
    input: &str,
    encode: fn(&str) -> Option<Vec<u32>>,
) -> (Vec<serde_json::Value>, usize, usize, usize) {
    let mut out = Vec::new();
    let stats = run_batch_loop(
        &config(),
        input.as_bytes(),
        &mut out,
        &[],
        12.5,
        empty_model(),
        &encode,
        &|_: &[u32]| String::new(),
    )
    .expect("loop survives per-prompt failures");
    let rows = String::from_utf8(out)
        .expect("utf8")
        .lines()
        .map(|l| serde_json::from_str(l).expect("each row is JSON"))
        .collect();
    (rows, stats.total_prompts, stats.successful, stats.failed)
}

#[test]
fn bad_json_encode_failure_and_missing_model_each_emit_one_error_row_5691() {
    let input = "\n   \nnot json\n{\"prompt\":\"hi\",\"task_id\":\"enc\"}\n";
    let (rows, total, ok, failed) = run(input, |_| None);
    // blank lines are skipped and uncounted; the other two fail
    assert_eq!((total, ok, failed), (2, 0, 2));
    assert_eq!(rows.len(), 2);
    assert!(rows[0]["error"]
        .as_str()
        .expect("err")
        .starts_with("JSON parse error"));
    assert!(rows[0]["task_id"].is_null());
    assert_eq!(rows[1]["error"], "Tokenizer encode failed");
    assert_eq!(rows[1]["task_id"], "enc");

    // encode succeeds, the model has no backend: the error names it, the task id survives
    let (rows, total, ok, failed) = run("{\"prompt\":\"hi\",\"task_id\":\"t1\"}\n", |_| {
        Some(vec![1, 2])
    });
    assert_eq!((total, ok, failed), (1, 0, 1));
    assert_eq!(rows[0]["task_id"], "t1");
    assert_eq!(rows[0]["tokens_generated"], 0);
    assert!(rows[0]["error"]
        .as_str()
        .expect("err")
        .contains("No model available"));
}

#[test]
fn stats_carry_model_load_and_every_tenth_prompt_is_counted_5691() {
    let input = "x\n".repeat(10);
    let mut out = Vec::new();
    let stats = run_batch_loop(
        &config(),
        input.as_bytes(),
        &mut out,
        &[],
        12.5,
        empty_model(),
        &|_: &str| None,
        &|_: &[u32]| String::new(),
    )
    .expect("ok");
    assert_eq!(
        (stats.total_prompts, stats.failed, stats.successful),
        (10, 10, 0)
    );
    assert!((stats.model_load_ms - 12.5).abs() < f64::EPSILON);
    assert!(stats.total_inference_ms.abs() < f64::EPSILON);
}

#[test]
fn an_unwritable_sink_is_an_io_error_not_a_panic_5691() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let err = run_batch_loop(
        &config(),
        "oops\n".as_bytes(),
        &mut Broken,
        &[],
        0.0,
        empty_model(),
        &|_: &str| None,
        &|_: &[u32]| String::new(),
    )
    .expect_err("write failure surfaces");
    assert!(format!("{err}").contains("Write error"), "{err}");
}
