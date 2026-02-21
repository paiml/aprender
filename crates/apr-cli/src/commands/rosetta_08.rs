
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};
include!("rosetta_format_type.rs");
include!("rosetta_rosetta.rs");
include!("rosetta_fingerprints_load.rs");
include!("rosetta_make_fingerprint.rs");
include!("rosetta_truncate_path_f16.rs");
include!("rosetta_tensor_fingerprint_statistical.rs");
include!("rosetta_format_type_02.rs");
include!("rosetta_compute_tensor_get.rs");
include!("rosetta_print_fingerprint.rs");
include!("rosetta_verification_report.rs");
}
