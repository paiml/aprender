
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::v2::AprV2Writer;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};
include!("construction.rs");
include!("builder.rs");
}
