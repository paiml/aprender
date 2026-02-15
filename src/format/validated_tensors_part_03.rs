
// =============================================================================
// POPPERIAN FALSIFICATION TESTS
// =============================================================================
//
// Per Popper (1959), these tests attempt to DISPROVE the contract works.
// If any test passes when it should fail, the contract has a logic error.

#[cfg(test)]
mod tests {
    use super::*;
include!("validated_tensors_part_03_part_02.rs");
include!("validated_tensors_part_03_part_03.rs");
}
