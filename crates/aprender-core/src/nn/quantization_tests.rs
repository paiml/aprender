#[cfg(test)]
#[path = "."]
mod tests {
    #[path = "quantization_tests_config.rs"]
    mod quantization_tests_config;
    #[path = "quantization_tests_linear.rs"]
    mod quantization_tests_linear;
}
