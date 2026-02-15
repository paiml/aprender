
// ============================================================================
// Write/Import/Rosetta Function Coverage Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests_write_functions {
    use super::*;
    use crate::format::gguf::{GgufModelConfig, GgufTokenizer};
    use crate::format::test_factory::{build_pygmy_apr, build_pygmy_safetensors};
    use crate::format::v2::AprV2Reader;
    use std::collections::BTreeMap;
    use std::fs;
    use tempfile::TempDir;
    // GAP-UX-002: Import trueno_quant functions for Q5K/Q6K tests
    // GH-202: transpose functions no longer re-exported from converter (wrong assumption removed)
    // Import directly from trueno_quant for tests that validate the functions themselves.
    use trueno_quant::{
        dequantize_q6_k_to_f32, quantize_q5_k, quantize_q5_k_matrix, transpose_q4k_for_matmul,
        transpose_q5k_for_matmul, transpose_q6k_for_matmul,
    };
include!("coverage_functions_part_03_part_02.rs");
include!("coverage_functions_part_03_part_03.rs");
}
