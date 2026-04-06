
// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod cli_tests;

// Additional inference coverage tests (Part 02)
#[cfg(test)]
#[path = "tests_temperature_nan.rs"]
mod cli_tests_part_02;

// CLI helper functions tests (Part 03)
#[cfg(test)]
#[path = "tests_03.rs"]
mod cli_tests_part_03;

// Coverage bridge tests (Part 04 - T-COV-95 B1)
#[cfg(test)]
#[path = "tests_04.rs"]
mod cli_tests_part_04;

// Deep CLI coverage tests (Part 05 - T-COV-95 Deep CLI)
#[cfg(test)]
#[path = "tests_05.rs"]
mod cli_tests_part_05;

// T-COV-95 Deep Coverage Bridge (Part 06 - handlers.rs: pull, push, list, serve, trace)
#[cfg(test)]
#[path = "tests_handle_pull.rs"]
mod cli_tests_part_06;

// T-COV-95 Deep Coverage Bridge (Part 07 - mod.rs: bench, viz, load, format, parse)
#[cfg(test)]
#[path = "tests_format_size.rs"]
mod cli_tests_format_size;

// T-COV-95 Extended Coverage (Part 08 - mod.rs: format_size, is_local_file_path, validate_suite_name, display_model_info)
#[cfg(test)]
#[path = "tests_format_size_02.rs"]
mod cli_tests_part_08;

// T-COV-95 Synthetic Falsification (Part 09 - inference.rs via Pygmy GGUF models)
#[cfg(test)]
#[path = "tests_pygmy_inference.rs"]
mod cli_tests_pygmy_inference;

// T-COV-95 CLI Inference Additional Coverage
#[cfg(test)]
#[path = "inference_tests_apr.rs"]
mod cli_inference_tests_part_02;

// T-COV-95 Active Pygmy CLI Inference (In-Memory)
#[cfg(test)]
#[path = "inference_tests_gguf_pygmy.rs"]
mod cli_inference_tests_part_03;

// T-COV-95 Artifact Falsification (Real Files, Real CLI)
#[cfg(test)]
#[path = "inference_tests_04.rs"]
mod cli_inference_tests_part_04;

// T-COV-95 Poisoned Pygmies: CLI Graceful Degradation Tests
#[cfg(test)]
#[path = "tests_10.rs"]
mod cli_tests_part_10;
