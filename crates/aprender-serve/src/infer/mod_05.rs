
// Additional coverage tests (tests_part_03.rs)
#[cfg(test)]
#[path = "tests_prefault_mmap.rs"]
mod infer_tests_part_03;

// Helper functions coverage tests (tests_part_04.rs)
#[cfg(test)]
#[path = "tests_04.rs"]
mod infer_tests_part_04;

// T-COV-95 Coverage Bridge tests (Part 05 - B5)
#[cfg(test)]
#[path = "tests_05.rs"]
mod infer_tests_part_05;

// Mock backend tests (PMAT-COV-95)
#[cfg(test)]
#[path = "tests_mock.rs"]
mod infer_tests_mock;

// T-COV-95 Deep Coverage Bridge (Part 06 - validate_model_path, qtype_to_dtype_str, mock paths)
#[cfg(test)]
#[path = "tests_validate_path.rs"]
mod infer_tests_part_06;

// T-COV-95 Synthetic Falsification (Part 07 - qtype all arms, InferenceConfig/Result fields)
#[cfg(test)]
#[path = "tests_qtype_f32_f16.rs"]
mod infer_tests_part_07;

// T-COV-95 Maimed Pygmy Campaign (Part 08 - Real inference paths with corrupted artifacts)
#[cfg(test)]
#[path = "tests_08.rs"]
mod infer_tests_part_08;

// T-COV-95 Phase 50: Deep coverage for infer/mod.rs pure functions
#[cfg(test)]
#[path = "tests_construction.rs"]
mod infer_tests_part_09;

// T-COV-95 Phase 55: Extended coverage for mock paths, builder, result types
#[cfg(test)]
#[path = "tests_10.rs"]
mod infer_tests_part_10;

// T-COV-95 Phase 60: Extended coverage for uncovered lines
#[cfg(test)]
#[path = "tests_11.rs"]
mod infer_tests_part_11;
