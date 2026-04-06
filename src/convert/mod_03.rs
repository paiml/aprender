
// T-COV-95 Deep Coverage Bridge (Part 03 - Q4K converter, rope_type, helpers)
#[cfg(test)]
#[path = "tests_apr_construction.rs"]
mod convert_tests_part_03;

// T-COV-95 Coverage Bridge (Part 04 - ConversionStats, to_apr_bytes, from_apr_bytes)
#[cfg(test)]
#[path = "tests_conversion_stats.rs"]
mod convert_tests_part_04;

// T-COV-95 Extended Coverage (Part 05 - RawTensor, dtypes, edge cases)
#[cfg(test)]
#[path = "tests_raw_tensor_coverage.rs"]
mod convert_tests_part_05;

// T-COV-95 Synthetic Falsification (Part 06 - Pygmy GGUF conversions)
#[cfg(test)]
#[path = "tests_convert_llama.rs"]
mod convert_tests_part_06;

// T-COV-95 Maimed Pygmy Campaign (Part 07 - Destructive APR/GGUF conversion tests)
#[cfg(test)]
#[path = "tests_maimed_apr.rs"]
mod convert_tests_part_07;

// T-COV-95 Semantic Divergence (Part 08 - Architecture Mismatch Tests)
#[cfg(test)]
#[path = "tests_semantic_divergence.rs"]
mod convert_tests_semantic_divergence;

// T-COV-95 Generative Falsification (Part 09 - Proptest Byte-Smasher)
#[cfg(test)]
#[path = "tests_byte_smasher.rs"]
mod convert_tests_byte_smasher;

// T-COV-95 Coverage Bridge (Part 10 - CRC32, checksum, metadata alignment, error paths)
#[cfg(test)]
#[path = "tests_checksum_and_q4k.rs"]
mod convert_tests_checksum_and_q4k;

// T-COV-95 Coverage Bridge (Part 11 - ConversionStats, RawTensor, Q4KConversionStats, CRC32)
#[cfg(test)]
#[path = "tests_conversion_stats_construction.rs"]
mod convert_tests_part_11;

// T-COV-95 Coverage Bridge (Part 12 - CRC32 vectors, uncovered rope_type archs, error paths)
#[cfg(test)]
#[path = "tests_crc32_single.rs"]
mod convert_tests_part_12;

// T-COV-95 Coverage Bridge (Part 13 - convert() pipeline, from_gguf_transformer edge cases, roundtrip)
#[cfg(test)]
#[path = "tests_convert.rs"]
mod convert_tests_part_13;

// T-COV-95 Coverage Bridge (Part 14 - GgufToAprQ4KConverter::convert full pipeline + roundtrip)
#[cfg(test)]
#[path = "tests_q4k_converter.rs"]
mod convert_tests_q4k_converter;
