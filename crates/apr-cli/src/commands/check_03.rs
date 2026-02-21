
#[cfg(test)]
mod tests {
    use super::*;
include!("check_stage_print_results.rs");
include!("check_aggregation.rs");
include!("check_qkv_ffn_detection.rs");
include!("details.rs");
include!("check_embedding_validity_tokenizer.rs");
include!("check_full_model_has.rs");
}
