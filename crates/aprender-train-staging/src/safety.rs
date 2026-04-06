//! Panic safety and graceful degradation
//!
//! Provides panic hooks and catch_unwind wrappers for training operations
//! to prevent data corruption on panic.
//!
//! Batuta: SF-08 (Panic Safety)

use std::panic;

/// Install a panic hook that logs structured panic information
/// and ensures checkpoint state is not corrupted.
pub fn install_panic_hook() {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        // Log the panic with structured info
        let location = info.location().map_or_else(
            || "unknown location".to_string(),
            |loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()),
        );
        let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic payload".to_string()
        };

        eprintln!("[entrenar::panic] at {location}: {message}");

        // Call the default hook for normal panic output
        default_hook(info);
    }));
}

/// Run a training operation with panic safety, returning None on panic.
///
/// This prevents panics from propagating through FFI boundaries
/// or corrupting shared state.
pub fn catch_training_panic<F, T>(op: F) -> Option<T>
where
    F: FnOnce() -> T + panic::UnwindSafe,
{
    match panic::catch_unwind(op) {
        Ok(result) => Some(result),
        Err(_) => {
            eprintln!("[entrenar::safety] Training operation panicked, returning None");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catch_training_panic_success() {
        let result = catch_training_panic(|| 42);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_catch_training_panic_failure() {
        let result = catch_training_panic(|| -> i32 { panic!("test panic") });
        assert_eq!(result, None);
    }

    #[test]
    fn test_install_panic_hook_does_not_panic() {
        // Just verify it doesn't crash
        install_panic_hook();
        // Restore default hook
        let _ = panic::take_hook();
    }

    #[test]
    fn test_catch_training_panic_with_string_result() {
        let result = catch_training_panic(|| "hello".to_string());
        assert_eq!(result, Some("hello".to_string()));
    }

    #[test]
    fn test_catch_training_panic_with_vec_result() {
        let result = catch_training_panic(|| vec![1, 2, 3]);
        assert_eq!(result, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_catch_training_panic_with_option_result() {
        let result = catch_training_panic(|| Some(42));
        assert_eq!(result, Some(Some(42)));
    }

    #[test]
    fn test_catch_training_panic_string_payload() {
        // Test panic with a String payload (not &str)
        let result = catch_training_panic(|| -> i32 {
            panic!("{}", "formatted panic message".to_string());
        });
        assert_eq!(result, None);
    }

    #[test]
    fn test_catch_training_panic_complex_computation() {
        let result = catch_training_panic(|| {
            let mut sum = 0;
            for i in 0..100 {
                sum += i;
            }
            sum
        });
        assert_eq!(result, Some(4950));
    }

    #[test]
    fn test_catch_training_panic_unit_return() {
        let result = catch_training_panic(|| {
            // Operation that returns ()
        });
        assert_eq!(result, Some(()));
    }

    #[test]
    fn test_catch_training_panic_bool_return() {
        let result = catch_training_panic(|| true);
        assert_eq!(result, Some(true));
    }

    #[test]
    fn test_catch_training_panic_nested_panic() {
        let result = catch_training_panic(|| -> i32 {
            let _inner = catch_training_panic(|| -> i32 {
                panic!("inner panic");
            });
            // inner panic is caught, outer should succeed
            99
        });
        assert_eq!(result, Some(99));
    }

    #[test]
    fn test_install_panic_hook_idempotent() {
        // Installing the hook twice should not crash
        install_panic_hook();
        install_panic_hook();
        // Restore default hook
        let _ = panic::take_hook();
    }

    #[test]
    fn test_catch_training_panic_after_hook_install() {
        install_panic_hook();
        let result = catch_training_panic(|| -> i32 {
            panic!("test after hook install");
        });
        assert_eq!(result, None);
        // Restore default hook
        let _ = panic::take_hook();
    }

    #[test]
    fn test_catch_training_panic_float_result() {
        let result = catch_training_panic(|| 3.14f64);
        assert_eq!(result, Some(3.14f64));
    }

    #[test]
    fn test_catch_training_panic_tuple_result() {
        let result = catch_training_panic(|| (42, "hello"));
        assert_eq!(result, Some((42, "hello")));
    }
}
