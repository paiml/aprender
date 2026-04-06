use super::*;
use proptest::prelude::*;

proptest! {
    /// cuda_error_string never panics for any i32
    #[test]
    fn prop_error_string_never_panics(code in any::<i32>()) {
        let _ = cuda_error_string(code);
    }

    /// cuda_error_string returns valid string for all inputs
    #[test]
    fn prop_error_string_valid(code in any::<i32>()) {
        let result = cuda_error_string(code);
        prop_assert!(!result.is_empty());
        prop_assert!(result.starts_with("CUDA_"));
    }

    /// Known error codes return their specific string
    #[test]
    fn prop_known_errors_have_specific_string(
        code in prop_oneof![
            Just(CUDA_SUCCESS),
            Just(CUDA_ERROR_INVALID_VALUE),
            Just(CUDA_ERROR_OUT_OF_MEMORY),
            Just(CUDA_ERROR_NO_DEVICE),
            Just(CUDA_ERROR_INVALID_PTX),
        ]
    ) {
        let result = cuda_error_string(code);
        prop_assert_ne!(result, "CUDA_ERROR_UNKNOWN");
    }
}
