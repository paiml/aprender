
#[cfg(test)]
mod tests {
    use super::*;
include!("converter_types_tests_source_parse.rs");
include!("converter_types_tests_gpt2_split.rs");

    #[test]
    fn falsify_mt_004_is_llm_exhaustive() {
        for arch in [
            Architecture::Llama,
            Architecture::Qwen2,
            Architecture::Qwen3,
            Architecture::Qwen3_5,
            Architecture::Gpt2,
            Architecture::Phi,
            Architecture::GptNeoX,
            Architecture::Opt,
            Architecture::Auto,
        ] {
            assert!(arch.is_llm(), "{arch:?} should be LLM");
            assert_eq!(arch.category(), ModelCategory::Llm);
        }
    }

    #[test]
    fn falsify_mt_005_non_llm() {
        assert!(!Architecture::Whisper.is_llm());
        assert!(!Architecture::Bert.is_llm());
        assert_eq!(Architecture::Whisper.category(), ModelCategory::Audio);
        assert_eq!(Architecture::Bert.category(), ModelCategory::Embedding);
    }
}
