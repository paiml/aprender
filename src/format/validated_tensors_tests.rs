
// =============================================================================
// POPPERIAN FALSIFICATION TESTS
// =============================================================================
//
// Per Popper (1959), these tests attempt to DISPROVE the contract works.
// If any test passes when it should fail, the contract has a logic error.

#[cfg(test)]
mod tests {
    use super::*;
include!("validated_tensors_tests_generators.rs");
include!("validated_tensors_tests_stats.rs");
}
