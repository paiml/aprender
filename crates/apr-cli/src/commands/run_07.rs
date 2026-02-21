
#[cfg(test)]
mod tests {
    use super::*;
include!("fails.rs");
include!("run_tests_model_source.rs");
include!("run_tests_format_prediction.rs");
include!("run_tests_parse_token_ids.rs");
include!("run_tests_inference_output.rs");
}
