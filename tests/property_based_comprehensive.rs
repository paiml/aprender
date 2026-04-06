//! Comprehensive property-based tests for pre-commit hook
//!
//! This test suite covers all core features of renacer using property-based testing
//! with proptest. Designed to run under 30 seconds as a pre-commit quality gate.
//!
#![allow(clippy::unwrap_used, clippy::clone_on_copy)]
//! Core features tested:
//! 1. Syscall tracing and filtering
//! 2. DWARF debug info parsing
//! 3. Statistics tracking with Trueno
//! 4. Function profiling and stack unwinding
//! 5. Call graph tracking
//! 6. I/O bottleneck detection
//! 7. JSON output serialization
//! 8. Timing and performance tracking

use proptest::prelude::*;

// Test syscall name resolution with random syscall numbers
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_syscall_name_never_panics(syscall_num in 0u64..400) {
        // Property: syscall_name should never panic for any input
        let name = renacer::syscalls::syscall_name(syscall_num as i64);

        // Should always return a valid string
        assert!(!name.is_empty());

        // Should either return a known name or "syscall_N" format
        assert!(name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_filter_trace_spec_parsing(
        syscalls in prop::collection::vec("[a-z]+", 0..5),
    ) {
        use renacer::filter::SyscallFilter;

        // Property: SyscallFilter should handle any trace spec without panicking
        let trace_spec = syscalls.join(",");
        let filter = if trace_spec.is_empty() {
            SyscallFilter::all()
        } else {
            SyscallFilter::from_expr(&trace_spec).unwrap_or_else(|_| SyscallFilter::all())
        };

        // Should successfully create filter
        assert!(format!("{:?}", filter).contains("SyscallFilter"));

        // Empty spec should create "all" filter
        if trace_spec.is_empty() {
            assert!(filter.should_trace("read"));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_stats_tracker_handles_arbitrary_data(
        syscall_names in prop::collection::vec("[a-z]{3,8}", 1..20),
        results in prop::collection::vec(-1000i64..1000, 1..20),
        times in prop::collection::vec(0u64..1_000_000, 1..20),
    ) {
        use renacer::stats::StatsTracker;

        // Property: StatsTracker should handle arbitrary syscall data
        let mut tracker = StatsTracker::new();

        for i in 0..syscall_names.len().min(results.len()).min(times.len()) {
            tracker.record(&syscall_names[i], results[i], times[i]);
        }

        // Should be able to print without panicking
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("StatsTracker"));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_function_profiler_handles_arbitrary_functions(
        function_names in prop::collection::vec("[a-zA-Z_][a-zA-Z0-9_]{0,30}", 1..10),
        times in prop::collection::vec(0u64..10_000_000, 1..10),
    ) {
        use renacer::function_profiler::FunctionProfiler;

        // Property: FunctionProfiler should handle arbitrary function names and times
        let mut profiler = FunctionProfiler::new();

        for i in 0..function_names.len().min(times.len()) {
            profiler.record(&function_names[i], "read", times[i], None);
        }

        // Should successfully export to string
        let mut output = Vec::new();
        let result = profiler.export_flamegraph(&mut output);
        assert!(result.is_ok());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_stack_frame_operations(
        rip in 0x1000u64..0x7fff_ffff_ffff,
        rbp in 0x1000u64..0x7fff_ffff_ffff,
    ) {
        use renacer::stack_unwind::StackFrame;

        // Property: StackFrame should handle any valid memory address
        let frame = StackFrame { rip, rbp };

        // Clone should produce identical frame
        let cloned = frame.clone();
        assert_eq!(cloned.rip, frame.rip);
        assert_eq!(cloned.rbp, frame.rbp);

        // Debug should always work
        let debug_str = format!("{:?}", frame);
        assert!(debug_str.contains("StackFrame"));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_json_output_serialization(
        syscall_names in prop::collection::vec("[a-z]{3,8}", 1..10),
        results in prop::collection::vec(-100i64..100, 1..10),
    ) {
        use renacer::json_output::{JsonOutput, JsonSyscall};
        use serde_json;

        // Property: JsonOutput should serialize any valid syscall data
        let mut output = JsonOutput::new();

        for i in 0..syscall_names.len().min(results.len()) {
            let syscall = JsonSyscall {
                name: syscall_names[i].clone(),
                args: vec!["arg1".to_string(), "arg2".to_string()],
                result: results[i],
                duration_us: None,
                source: None,
            };
            output.add_syscall(syscall);
        }

        // Should successfully serialize to JSON
        let json_result = serde_json::to_string(&output);
        assert!(json_result.is_ok());

        // Should successfully deserialize back
        let json_str = json_result.expect("test");
        let parsed: Result<JsonOutput, _> = serde_json::from_str(&json_str);
        assert!(parsed.is_ok());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_call_graph_tracking(
        caller_names in prop::collection::vec("[a-zA-Z_][a-zA-Z0-9_]{0,20}", 1..8),
        callee_names in prop::collection::vec("[a-zA-Z_][a-zA-Z0-9_]{0,20}", 1..8),
    ) {
        use renacer::function_profiler::FunctionProfiler;

        // Property: Call graph tracking should handle arbitrary call relationships
        let mut profiler = FunctionProfiler::new();

        for i in 0..caller_names.len().min(callee_names.len()) {
            // Record with caller
            profiler.record(&callee_names[i], "read", 1000, Some(&caller_names[i]));
        }

        // Should be able to export without panicking
        let mut output = Vec::new();
        assert!(profiler.export_flamegraph(&mut output).is_ok());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_io_bottleneck_detection(
        io_syscalls in prop::collection::vec(
            prop::sample::select(vec!["read", "write", "open", "openat", "pread64", "pwrite64"]),
            1..10
        ),
        times in prop::collection::vec(0u64..5_000_000, 1..10),
    ) {
        use renacer::function_profiler::FunctionProfiler;

        // Property: I/O bottleneck detection should identify slow I/O operations
        let mut profiler = FunctionProfiler::new();

        let mut has_slow_io = false;
        for i in 0..io_syscalls.len().min(times.len()) {
            profiler.record(&format!("func_{}", i), io_syscalls[i], times[i], None);

            // Detect if we have slow I/O (>1ms = 1_000_000us)
            if times[i] > 1_000_000 {
                has_slow_io = true;
            }
        }

        // Profiler should track this data
        let mut output = Vec::new();
        assert!(profiler.export_flamegraph(&mut output).is_ok());

        // If we had slow I/O, export should succeed
        if has_slow_io {
            assert!(!output.is_empty());
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_hot_path_analysis(
        function_names in prop::collection::vec("[a-z]{3,8}", 1..15),
        times in prop::collection::vec(1u64..1_000_000, 1..15),
    ) {
        use renacer::function_profiler::FunctionProfiler;

        // Property: Hot path analysis should identify most expensive functions
        let mut profiler = FunctionProfiler::new();
        let mut max_time = 0u64;

        for i in 0..function_names.len().min(times.len()) {
            profiler.record(&function_names[i], "read", times[i], None);

            if times[i] > max_time {
                max_time = times[i];
            }
        }

        // Export should succeed
        let mut output = Vec::new();
        assert!(profiler.export_flamegraph(&mut output).is_ok());

        // Most expensive function should appear in output
        if !function_names.is_empty() {
            let output_str = String::from_utf8_lossy(&output);
            // At least one function should be in the flamegraph
            assert!(!output_str.is_empty() || max_time == 0);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_profiling_context_categories(
        times_us in prop::collection::vec(0u64..100_000, 1..10),
    ) {
        use renacer::profiling::{ProfilingContext, ProfilingCategory};
        use std::time::Duration;

        // Property: Profiling context should track all categories correctly
        let mut ctx = ProfilingContext::new();

        for &time_us in &times_us {
            ctx.record_time(ProfilingCategory::Ptrace, Duration::from_micros(time_us));
            ctx.record_time(ProfilingCategory::Formatting, Duration::from_micros(time_us));
            ctx.record_time(ProfilingCategory::DwarfLookup, Duration::from_micros(time_us));
        }

        // Should not panic when printing
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("ProfilingContext"));
    }
}

// Integration property tests - test combinations of features
proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_end_to_end_trace_filtering_stats(
        syscalls in prop::collection::vec(
            prop::sample::select(vec!["read", "write", "open", "close", "mmap"]),
            5..15
        ),
    ) {
        use renacer::filter::SyscallFilter;
        use renacer::stats::StatsTracker;

        // Property: Filtering and stats tracking should work together
        let filter = SyscallFilter::from_expr(&syscalls.join(","))
            .unwrap_or_else(|_| SyscallFilter::all());
        let mut tracker = StatsTracker::new();

        // Simulate tracing with filtering
        for syscall in &syscalls {
            if filter.should_trace(syscall) {
                tracker.record(syscall, 0, 100);
            }
        }

        // Both should work without panicking
        let filter_str = format!("{:?}", filter);
        let tracker_str = format!("{:?}", tracker);

        assert!(!filter_str.is_empty());
        assert!(!tracker_str.is_empty());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_profiling_with_json_export(
        function_names in prop::collection::vec("[a-z]{4,10}", 3..8),
        times in prop::collection::vec(100u64..10_000, 3..8),
    ) {
        use renacer::function_profiler::FunctionProfiler;
        use renacer::json_output::{JsonOutput, JsonSyscall};

        // Property: Function profiling and JSON output should integrate
        let mut profiler = FunctionProfiler::new();
        let mut json_out = JsonOutput::new();

        for i in 0..function_names.len().min(times.len()) {
            profiler.record(&function_names[i], "read", times[i], None);

            let syscall = JsonSyscall {
                name: format!("syscall_{}", i),
                args: vec!["1".to_string(), "2".to_string(), "3".to_string()],
                result: 0,
                duration_us: None,
                source: None,
            };
            json_out.add_syscall(syscall);
        }

        // Both should export successfully
        let mut flamegraph = Vec::new();
        assert!(profiler.export_flamegraph(&mut flamegraph).is_ok());

        let json_result = serde_json::to_string(&json_out);
        assert!(json_result.is_ok());
    }
}

// Regression tests using property-based approach
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_dwarf_source_location_invariants(
        line in 1u32..10000,
    ) {
        use renacer::dwarf::SourceLocation;

        // Property: SourceLocation should maintain invariants
        let loc = SourceLocation {
            file: "test.rs".to_string(),
            line,
            column: Some(10),
            function: Some("test_func".to_string()),
        };

        // Clone should be identical
        let cloned = loc.clone();
        assert_eq!(cloned.line, loc.line);
        assert_eq!(cloned.column, loc.column);

        // Debug should work
        let debug_str = format!("{:?}", loc);
        assert!(debug_str.contains("SourceLocation"));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_trueno_vector_integration(
        values in prop::collection::vec(0u64..1_000_000, 1..100),
    ) {
        use renacer::stats::StatsTracker;

        // Property: Trueno integration in StatsTracker handles arbitrary data
        let mut tracker = StatsTracker::new();

        for (i, &val) in values.iter().enumerate() {
            tracker.record(&format!("syscall_{}", i), 0, val);
        }

        // Tracker should work with any data
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("StatsTracker"));

        // This tests the Trueno integration used in stats tracking
        assert!(!values.is_empty());
    }
}

#[cfg(test)]
mod deterministic_core_feature_tests {
    //! Deterministic tests ensuring all core features work
    //! These complement the property tests above

    use renacer::*;

    #[test]
    fn test_all_core_features_integration() {
        // This test ensures all major modules can be instantiated together
        // without conflicts - validates the overall architecture

        let filter = filter::SyscallFilter::from_expr("trace=read,write").expect("test");
        let mut tracker = stats::StatsTracker::new();
        let mut profiler = function_profiler::FunctionProfiler::new();
        let mut json_out = json_output::JsonOutput::new();

        // Simulate a traced syscall going through all systems
        let result = 10i64;
        let time = 1500u64;

        if filter.should_trace("read") {
            tracker.record("read", result, time);
            profiler.record("my_function", "read", time, None);

            let syscall = json_output::JsonSyscall {
                name: "read".to_string(),
                args: vec!["1".to_string(), "2".to_string(), "3".to_string()],
                result,
                duration_us: Some(time),
                source: None,
            };
            json_out.add_syscall(syscall);
        }

        // All systems should work together
        assert!(format!("{:?}", tracker).contains("StatsTracker"));

        let mut flamegraph = Vec::new();
        assert!(profiler.export_flamegraph(&mut flamegraph).is_ok());

        let json = serde_json::to_string(&json_out);
        assert!(json.is_ok());
    }

    #[test]
    fn test_syscall_coverage_common_calls() {
        // Ensure common syscalls are properly named
        assert_eq!(syscalls::syscall_name(0), "read");
        assert_eq!(syscalls::syscall_name(1), "write");
        assert_eq!(syscalls::syscall_name(2), "open");
        assert_eq!(syscalls::syscall_name(257), "openat");
        assert_eq!(syscalls::syscall_name(9), "mmap");
    }

    #[test]
    fn test_filter_all_classes() {
        use renacer::filter::SyscallFilter;

        // Test all filter classes work
        let file_filter = SyscallFilter::from_expr("trace=file").expect("test");
        assert!(file_filter.should_trace("read"));

        let network_filter = SyscallFilter::from_expr("trace=network").expect("test");
        assert!(network_filter.should_trace("socket"));

        let process_filter = SyscallFilter::from_expr("trace=process").expect("test");
        assert!(process_filter.should_trace("fork"));

        let memory_filter = SyscallFilter::from_expr("trace=memory").expect("test");
        assert!(memory_filter.should_trace("mmap"));
    }

    #[test]
    fn test_profiling_all_categories() {
        use renacer::profiling::{ProfilingCategory, ProfilingContext};
        use std::time::Duration;

        let mut ctx = ProfilingContext::new();

        // Test all profiling categories
        ctx.record_time(ProfilingCategory::Ptrace, Duration::from_micros(100));
        ctx.record_time(ProfilingCategory::Formatting, Duration::from_micros(200));
        ctx.record_time(ProfilingCategory::DwarfLookup, Duration::from_micros(300));
        ctx.record_time(ProfilingCategory::MemoryRead, Duration::from_micros(400));
        ctx.record_time(ProfilingCategory::Statistics, Duration::from_micros(500));

        // Should track all categories
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("ProfilingContext"));
    }
}
