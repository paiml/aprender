
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
include!("profile_output_format.rs");
include!("profile_file.rs");
include!("profile_real.rs");
include!("profile_filter_results.rs");
include!("profile_profile.rs");
include!("profile_print_flamegraph.rs");
include!("profile_filter_mlp.rs");
}
