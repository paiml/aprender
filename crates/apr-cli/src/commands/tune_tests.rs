use super::*;
use std::fs;

// =========================================================================
// TuneMethod tests
// =========================================================================

#[test]
fn test_tune_method_parse() {
    assert!(matches!(
        "lora".parse::<TuneMethod>().expect("parse::<TuneMethod>"),
        TuneMethod::LoRA
    ));
    assert!(matches!(
        "qlora".parse::<TuneMethod>().expect("parse::<TuneMethod>"),
        TuneMethod::QLoRA
    ));
    assert!(matches!(
        "auto".parse::<TuneMethod>().expect("parse::<TuneMethod>"),
        TuneMethod::Auto
    ));
    assert!(matches!(
        "full".parse::<TuneMethod>().expect("parse::<TuneMethod>"),
        TuneMethod::Full
    ));
}

#[test]
fn test_tune_method_parse_case_insensitive() {
    assert!(matches!(
        "LORA".parse::<TuneMethod>().expect("parse::<TuneMethod>"),
        TuneMethod::LoRA
    ));
    assert!(matches!(
        "LoRa".parse::<TuneMethod>().expect("parse::<TuneMethod>"),
        TuneMethod::LoRA
    ));
    assert!(matches!(
        "QLORA".parse::<TuneMethod>().expect("parse::<TuneMethod>"),
        TuneMethod::QLoRA
    ));
}

#[test]
fn test_tune_method_parse_invalid() {
    let result: Result<TuneMethod, _> = "invalid".parse();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown method"));
}

#[test]
fn test_tune_method_default() {
    let method = TuneMethod::default();
    assert!(matches!(method, TuneMethod::Auto));
}

#[test]
fn test_tune_method_debug() {
    assert_eq!(format!("{:?}", TuneMethod::Auto), "Auto");
    assert_eq!(format!("{:?}", TuneMethod::Full), "Full");
    assert_eq!(format!("{:?}", TuneMethod::LoRA), "LoRA");
    assert_eq!(format!("{:?}", TuneMethod::QLoRA), "QLoRA");
}

#[test]
fn test_tune_method_clone() {
    let method = TuneMethod::LoRA;
    let cloned = method;
    assert!(matches!(cloned, TuneMethod::LoRA));
}

#[test]
fn test_tune_method_copy() {
    let method = TuneMethod::QLoRA;
    let copied: TuneMethod = method;
    assert!(matches!(method, TuneMethod::QLoRA));
    assert!(matches!(copied, TuneMethod::QLoRA));
}

#[test]
fn test_tune_method_into_entrenar_method() {
    let auto: Method = TuneMethod::Auto.into();
    assert!(matches!(auto, Method::Auto));

    let full: Method = TuneMethod::Full.into();
    assert!(matches!(full, Method::Full));

    let lora: Method = TuneMethod::LoRA.into();
    assert!(matches!(lora, Method::LoRA));

    let qlora: Method = TuneMethod::QLoRA.into();
    assert!(matches!(qlora, Method::QLoRA));
}

// =========================================================================
// parse_model_size tests
// =========================================================================

#[test]
fn test_parse_model_size() {
    assert_eq!(parse_model_size("7B").expect("value"), 7_000_000_000);
    assert_eq!(parse_model_size("1.5B").expect("5B'"), 1_500_000_000);
    assert_eq!(parse_model_size("70B").expect("value"), 70_000_000_000);
    assert_eq!(parse_model_size("500M").expect("value"), 500_000_000);
}

#[test]
fn test_parse_model_size_case_insensitive() {
    assert_eq!(parse_model_size("7b").expect("value"), 7_000_000_000);
    assert_eq!(parse_model_size("1.5b").expect("5b'"), 1_500_000_000);
}

#[test]
fn test_parse_model_size_invalid() {
    assert!(parse_model_size("7").is_err());
    assert!(parse_model_size("7GB").is_err());
    assert!(parse_model_size("abc").is_err());
}

#[test]
fn test_parse_model_size_decimal() {
    assert_eq!(parse_model_size("0.5B").expect("5B'"), 500_000_000);
    assert_eq!(parse_model_size("2.7B").expect("7B'"), 2_700_000_000);
    assert_eq!(parse_model_size("13.5B").expect("5B'"), 13_500_000_000);
}

#[test]
fn test_parse_model_size_millions() {
    assert_eq!(parse_model_size("125M").expect("value"), 125_000_000);
    assert_eq!(parse_model_size("350M").expect("value"), 350_000_000);
    assert_eq!(parse_model_size("1000M").expect("value"), 1_000_000_000);
}

#[test]
fn test_parse_model_size_large() {
    assert_eq!(parse_model_size("180B").expect("value"), 180_000_000_000);
    assert_eq!(parse_model_size("405B").expect("value"), 405_000_000_000);
}

#[test]
fn test_parse_model_size_invalid_number() {
    let result = parse_model_size("abcB");
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Invalid number"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

// =========================================================================
// format_params tests
// =========================================================================

#[test]
fn test_format_params() {
    assert_eq!(format_params(7_000_000_000), "7.0B");
    assert_eq!(format_params(1_500_000_000), "1.5B");
    assert_eq!(format_params(500_000_000), "500.0M");
}

#[test]
fn test_format_params_small() {
    assert_eq!(format_params(100_000), "100000");
    assert_eq!(format_params(999_999), "999999");
}

#[test]
fn test_format_params_millions() {
    assert_eq!(format_params(1_000_000), "1.0M");
    assert_eq!(format_params(125_000_000), "125.0M");
    assert_eq!(format_params(999_999_999), "1000.0M");
}

#[test]
fn test_format_params_billions() {
    assert_eq!(format_params(1_000_000_000), "1.0B");
    assert_eq!(format_params(70_000_000_000), "70.0B");
    assert_eq!(format_params(405_000_000_000), "405.0B");
}

// =========================================================================
// read_params_from_file — falsifiers for #2570
//
// These replace two tests that ASSERTED THE DEFECT. `test_estimate_params_from_file`
// wrote a megabyte of zeros to `test_model.gguf` and demanded the answer be
// exactly 2,000,000 parameters; `test_estimate_params_from_file_not_found` was
// the only case that could ever fail, and it only checked that a MISSING file
// errored. A file of a million zero bytes is not a model, and a green suite that
// requires it to be reported as a 2M-parameter one is how `apr tune` came to
// print "Model parameters: 982,800,128" for a model `apr inspect` reads as
// 630,167,424 out of the same bytes.
// =========================================================================

/// A GGUF whose true parameter count is deliberately nowhere near either of the
/// old size heuristics: 2048 F32 params in a file of ~8.3 KB.
fn tiny_gguf_bytes() -> Vec<u8> {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    let data: Vec<u8> = (0..2048u32)
        .flat_map(|i| (i as f32 + 1.0).to_le_bytes())
        .collect();
    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![64, 32],
        dtype: GgmlType::F32,
        data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &[tensor], &metadata).expect("export GGUF");
    bytes
}

#[test]
fn read_params_from_file_reads_the_shapes_not_the_file_size() {
    let dir = std::env::temp_dir().join("apr_tune_2570_real");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("model.gguf");
    let bytes = tiny_gguf_bytes();
    let file_len = bytes.len() as u64;
    fs::write(&path, &bytes).expect("write fixture");

    let params = read_params_from_file(&path).expect("a valid GGUF must yield its param count");

    // 64 x 32 tensor shape, read out of the header.
    assert_eq!(params, 2048, "must sum the declared tensor shapes");
    // The two constants the old estimator used. Naming them here is the point:
    // if someone reintroduces a size heuristic, this fails and says which one.
    assert_ne!(
        params,
        file_len * 2,
        "must not be the .gguf heuristic (file_len {file_len} x 2)"
    );
    assert_ne!(
        params,
        file_len / 2,
        "must not be the fp16 heuristic (file_len {file_len} / 2)"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// The headline case from #2570: a zero-byte file was reported as
/// "Model parameters: 0" that "fits in 16.0 GB VRAM", exit 0.
#[test]
fn read_params_from_file_refuses_an_empty_file() {
    let dir = std::env::temp_dir().join("apr_tune_2570_empty");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("empty.apr");
    fs::write(&path, b"").expect("write fixture");

    let err = read_params_from_file(&path)
        .expect_err("an empty file must not be planned as a model that fits");
    match err {
        CliError::ValidationFailed(msg) | CliError::InvalidFormat(msg) => {
            assert!(
                msg.contains("--model <SIZE>"),
                "the refusal must point at the flag that plans WITHOUT a model file, got: {msg}"
            );
        }
        other => panic!("expected a validation failure, got {other:?}"),
    }

    let _ = fs::remove_dir_all(&dir);
}

/// A megabyte of zeros named `.gguf` — the exact fixture the deleted test
/// demanded be reported as 2,000,000 parameters.
#[test]
fn read_params_from_file_refuses_a_million_zero_bytes_named_gguf() {
    let dir = std::env::temp_dir().join("apr_tune_2570_zeros");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("test_model.gguf");
    fs::write(&path, vec![0u8; 1_000_000]).expect("write fixture");

    let err = read_params_from_file(&path)
        .expect_err("a megabyte of zeros is not a 2,000,000-parameter model");
    assert!(
        !format!("{err:?}").contains("2000000"),
        "must not report the old heuristic's answer: {err:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// A truncated GGUF reaches `apr tune` through the same `list_tensors` bounds
/// check added for #2569 — the two defects share one correct implementation.
#[test]
fn read_params_from_file_refuses_a_truncated_gguf() {
    let dir = std::env::temp_dir().join("apr_tune_2570_truncated");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("truncated.gguf");
    let bytes = tiny_gguf_bytes();
    // Keep the header and tensor info, drop the 8192-byte data section.
    fs::write(&path, &bytes[..bytes.len() - 8192]).expect("write fixture");

    let err = read_params_from_file(&path)
        .expect_err("apr tune must not plan a fine-tune from a truncated model");
    assert!(
        format!("{err:?}").contains("Truncated"),
        "the refusal must carry the reason, got: {err:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// A structurally VALID GGUF that declares zero tensors. It parses cleanly, so
/// the listing succeeds and the sum is 0 — the one path where the explicit
/// `params == 0` refusal is the only thing standing between the user and a
/// "Model parameters: 0 … fits in 16.0 GB VRAM" verdict.
#[test]
fn read_params_from_file_refuses_a_valid_gguf_with_no_tensors() {
    let dir = std::env::temp_dir().join("apr_tune_2570_no_tensors");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("empty.gguf");

    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version
    bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count = 0
    fs::write(&path, &bytes).expect("write fixture");

    // Precondition: the listing itself succeeds — this is not the #2569 path.
    let listing = aprender::format::tensors::list_tensors(
        &path,
        aprender::format::tensors::TensorListOptions::new(),
    )
    .expect("a zero-tensor GGUF is structurally valid and must list");
    assert_eq!(listing.tensor_count, 0);

    let err = read_params_from_file(&path)
        .expect_err("zero parameters is not a model to plan a fine-tune for");
    assert!(
        format!("{err:?}").contains("no tensor parameters"),
        "got: {err:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn read_params_from_file_not_found() {
    let result = read_params_from_file(Path::new("/nonexistent/model.bin"));
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Cannot read tensor metadata"), "got: {msg}");
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

// =========================================================================
// run() error cases tests
// =========================================================================

#[test]
fn test_run_no_model_or_size() {
    let result = run(
        None, // No model path
        TuneMethod::Auto,
        None,
        16.0,
        true,
        None, // No model size
        false,
        None,
        false,
    );

    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Either --model or model path required"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_run_with_model_size() {
    let result = run(
        None,
        TuneMethod::LoRA,
        Some(8),
        24.0,
        true,
        Some("7B"),
        false,
        None,
        false,
    );

    assert!(result.is_ok());
}

#[test]
fn test_run_with_model_size_json_output() {
    let result = run(
        None,
        TuneMethod::QLoRA,
        Some(16),
        16.0,
        true,
        Some("1.5B"),
        false,
        None,
        true, // JSON output
    );

    assert!(result.is_ok());
}

#[test]
fn test_run_plan_only() {
    let result = run(
        None,
        TuneMethod::Auto,
        None,
        8.0,
        true, // plan_only
        Some("3B"),
        false,
        None,
        false,
    );

    assert!(result.is_ok());
}

#[test]
fn test_run_with_rank() {
    let result = run(
        None,
        TuneMethod::LoRA,
        Some(4), // rank
        16.0,
        true,
        Some("7B"),
        false,
        None,
        false,
    );

    assert!(result.is_ok());
}

#[test]
fn test_run_with_model_file() {
    let temp_dir = std::env::temp_dir().join("apr_tune_run_test");
    let _ = fs::create_dir_all(&temp_dir);

    // #2570: this fixture used to be 100 KB of ZEROS named `.gguf`, and the test
    // asserted `run(..).is_ok()` on it. That is precisely the defect — planning a
    // fine-tune for a file that contains no model — so the assertion locked it in.
    // A real (tiny) GGUF is what the success path is entitled to.
    let test_file = temp_dir.join("test_model.gguf");
    fs::write(&test_file, tiny_gguf_bytes()).expect("write fixture");

    let result = run(
        Some(&test_file),
        TuneMethod::QLoRA,
        None,
        8.0,
        true,
        None,
        false,
        None,
        false,
    );

    assert!(result.is_ok(), "a real GGUF must still plan: {result:?}");

    let _ = fs::remove_dir_all(&temp_dir);
}

/// The other half of the case above: the zero-filled file that used to pass.
#[test]
fn test_run_refuses_a_file_that_is_not_a_model() {
    let temp_dir = std::env::temp_dir().join("apr_tune_run_zeros");
    let _ = fs::create_dir_all(&temp_dir);

    let test_file = temp_dir.join("test_model.gguf");
    fs::write(&test_file, vec![0u8; 100_000]).expect("write fixture");

    let result = run(
        Some(&test_file),
        TuneMethod::QLoRA,
        None,
        8.0,
        true,
        None,
        false,
        None,
        false,
    );

    assert!(
        result.is_err(),
        "100 KB of zeros is not a model; apr tune must not plan for it"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_run_model_file_not_found() {
    let result = run(
        Some(Path::new("/nonexistent/model.gguf")),
        TuneMethod::Auto,
        None,
        16.0,
        true,
        None,
        false,
        None,
        false,
    );

    assert!(result.is_err());
}

#[test]
fn test_run_invalid_model_size() {
    let result = run(
        None,
        TuneMethod::Auto,
        None,
        16.0,
        true,
        Some("invalid"), // Invalid size format
        false,
        None,
        false,
    );

    assert!(result.is_err());
}

// =========================================================================
// classify tune tests (SPEC-TUNE-2026-001)
// =========================================================================

#[test]
fn test_classify_tune_json_output() {
    let result = run_classify_tune(
        None, 3,      // budget
        "tpe",  // strategy
        "asha", // scheduler
        true,   // scout
        None,   // no data (dry run)
        5,      // num_classes
        None,   // model_size
        None,   // from_scout
        20,     // max_epochs
        None,   // time_limit
        true,   // json output
    );
    assert!(result.is_ok(), "JSON classify tune should succeed");
}

#[test]
fn test_classify_tune_human_output() {
    let result = run_classify_tune(
        None, 5,        // budget
        "random", // strategy
        "none",   // scheduler
        false,    // full mode
        None,     // no data
        3,        // num_classes
        None,     // model_size
        None,     // from_scout
        10,       // max_epochs
        None,     // time_limit
        false,    // human output
    );
    assert!(result.is_ok(), "Human classify tune should succeed");
}

#[test]
fn test_classify_tune_invalid_strategy() {
    let result = run_classify_tune(
        None,
        5,
        "invalid_strategy",
        "asha",
        true,
        None,
        5,
        None,
        None,
        20,
        None,
        false,
    );
    assert!(result.is_err(), "Invalid strategy should fail");
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("Unknown strategy"),
                "Error should mention unknown strategy, got: {msg}"
            );
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_classify_tune_budget_zero() {
    let result = run_classify_tune(
        None, 0, // budget=0
        "tpe", "asha", true, None, 5, None, None, 20, None, false,
    );
    assert!(result.is_err(), "Budget=0 should fail");
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("FALSIFY-TUNE-001"),
                "Error should contain FALSIFY-TUNE-001, got: {msg}"
            );
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_classify_tune_missing_data() {
    let result = run_classify_tune(
        None,
        3,
        "tpe",
        "asha",
        true,
        Some(Path::new("/nonexistent/corpus.jsonl")),
        5,
        None,
        None,
        20,
        None,
        false,
    );
    assert!(result.is_err(), "Missing data file should fail");
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("FALSIFY-TUNE-003"),
                "Error should contain FALSIFY-TUNE-003, got: {msg}"
            );
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

// ── Additional falsification tests (SPEC-TUNE-2026-001 §7) ────

#[test]
fn test_classify_tune_grid_strategy_json() {
    let result = run_classify_tune(
        None, 5, "grid", "median", false, None, 3, None, None, 10, None, true, // JSON output
    );
    assert!(
        result.is_ok(),
        "Grid strategy with JSON output should succeed"
    );
}

#[test]
fn test_classify_tune_random_strategy() {
    let result = run_classify_tune(
        None, 3, "random", "none", true, None, 5, None, None, 1, None, false,
    );
    assert!(result.is_ok(), "Random strategy should succeed");
}

#[test]
fn test_classify_tune_invalid_scheduler() {
    let result = run_classify_tune(
        None,
        5,
        "tpe",
        "hyperband_v99", // invalid scheduler
        true,
        None,
        5,
        None,
        None,
        20,
        None,
        false,
    );
    assert!(result.is_err(), "Invalid scheduler should fail");
}

#[test]
fn test_classify_tune_num_classes_zero() {
    let result = run_classify_tune(
        None, 3, "tpe", "asha", true, None, 0, // num_classes=0
        None, None, 20, None, false,
    );
    assert!(result.is_err(), "num_classes=0 should fail");
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("FALSIFY-TUNE-004"),
                "Error should contain FALSIFY-TUNE-004, got: {msg}"
            );
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_classify_tune_scout_mode_caps_epochs() {
    // Scout mode should succeed with budget=1 (minimal)
    let result = run_classify_tune(
        None, 1, "tpe", "asha", true, // scout mode
        None, 5, None, None, 100, None, true, // JSON for easy verification
    );
    assert!(result.is_ok(), "Scout mode with budget=1 should succeed");
}

#[test]
fn test_classify_tune_large_budget_json() {
    let result = run_classify_tune(
        None, 100, "tpe", "asha", false, None, 10, None, None, 20, None, true, // JSON output
    );
    // Should succeed — budget=100 with no data just shows sample configs
    assert!(result.is_ok(), "Large budget with JSON should succeed");
}
