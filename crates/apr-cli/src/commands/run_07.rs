
#[cfg(test)]
mod tests {
    use super::*;
include!("fails.rs");
include!("run_tests_model_source.rs");
include!("run_tests_format_prediction.rs");
include!("run_tests_parse_token_ids.rs");
include!("run_tests_inference_output.rs");
include!("run_tests_chrome_trace.rs");
include!("run_tests_layer_trace.rs");
include!("run_tests_stream_output.rs");
}
