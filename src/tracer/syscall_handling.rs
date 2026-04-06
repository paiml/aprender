//! Syscall entry/exit handling, recording, and helper functions
//!
//! Contains the low-level syscall interception logic, memory reading,
//! DWARF function extraction, and per-format recording functions.

use anyhow::{Context, Result};
use nix::unistd::Pid;

use super::{SyscallEntry, Tracers, VisualizerEvent};

/// Check if a function name belongs to libc/system code (not user code)
fn is_system_function(name: &str) -> bool {
    name.starts_with("__")
        || name.contains("libc")
        || name.contains("@plt")
        || name.contains("@@GLIBC")
}

/// Extract user function name from a stack frame's DWARF info, if available.
fn extract_user_function(
    frame: &crate::stack_unwind::StackFrame,
    dwarf_ctx: &crate::dwarf::DwarfContext,
) -> Option<String> {
    let source_info = dwarf_ctx.lookup(frame.rip).ok().flatten()?;
    let func_name = source_info.function?;
    if is_system_function(&func_name) {
        return None;
    }
    Some(func_name.clone())
}

/// Find user function and its caller from stack unwinding
/// Returns (`current_function`, `caller_function`)
fn find_user_function_with_caller(
    child: Pid,
    dwarf_ctx: &crate::dwarf::DwarfContext,
) -> Option<(String, Option<String>)> {
    // Unwind the stack to get all frames
    let frames = crate::stack_unwind::unwind_stack(child).ok()?;

    let user_functions: Vec<String> =
        frames.iter().filter_map(|frame| extract_user_function(frame, dwarf_ctx)).collect();

    // Return the first user function and its caller (if available)
    match user_functions.len() {
        0 => None,
        1 => Some((user_functions[0].clone(), None)),
        _ => Some((user_functions[0].clone(), Some(user_functions[1].clone()))),
    }
}

/// Format syscall arguments for JSON output
fn format_syscall_args_for_json(
    child: Pid,
    name: &str,
    arg1: u64,
    arg2: u64,
    arg3: u64,
) -> Vec<String> {
    match name {
        "openat" => {
            let filename =
                read_string(child, arg2 as usize).unwrap_or_else(|_| format!("{arg2:#x}"));
            vec![format!("{:#x}", arg1), format!("\"{}\"", filename), format!("{:#x}", arg3)]
        }
        _ => vec![format!("{:#x}", arg1), format!("{:#x}", arg2), format!("{:#x}", arg3)],
    }
}

/// Extract function name and caller from DWARF context
fn extract_function_names(
    child: Pid,
    dwarf_ctx: Option<&crate::dwarf::DwarfContext>,
    source_info: &Option<crate::dwarf::SourceLocation>,
    function_profiling_enabled: bool,
) -> (Option<String>, Option<String>) {
    if function_profiling_enabled {
        if let Some(ctx) = dwarf_ctx {
            find_user_function_with_caller(child, ctx)
                .map_or((None, None), |(func, caller)| (Some(func), caller))
        } else {
            let func = source_info.as_ref().and_then(|src| src.function.clone());
            (func, None)
        }
    } else {
        let func = source_info.as_ref().and_then(|src| src.function.clone());
        (func, None)
    }
}

/// Handle syscall entry - record syscall number and arguments
/// Returns the syscall entry data if it should be traced, None otherwise
pub(super) fn handle_syscall_entry(
    child: Pid,
    dwarf_ctx: Option<&crate::dwarf::DwarfContext>,
    filter: &crate::filter::SyscallFilter,
    statistics_mode: bool,
    structured_output: bool,
    function_profiling_enabled: bool,
    transpiler_map: Option<&crate::transpiler_map::TranspilerMap>,
) -> Result<Option<SyscallEntry>> {
    let regs = crate::arch::PtraceRegs::get(child)?;

    let syscall_num = regs.syscall_number();

    // Get syscall name
    let name = crate::syscalls::syscall_name(syscall_num);

    // Sprint 9-10: Filter syscalls based on -e trace= expression
    if !filter.should_trace(name) {
        // Don't print or track this syscall
        return Ok(None);
    }

    let arg1 = regs.arg1();
    let arg2 = regs.arg2();
    let arg3 = regs.arg3();

    // Sprint 5-6: Look up source location using instruction pointer if DWARF is available
    let source_info = if let Some(ctx) = dwarf_ctx {
        let ip = regs.instruction_pointer();
        ctx.lookup(ip).ok().flatten()
    } else {
        None
    };

    // Format arguments for structured output modes (JSON, CSV, HTML) if needed
    let args = if structured_output {
        format_syscall_args_for_json(child, name, arg1, arg2, arg3)
    } else {
        Vec::new()
    };

    // Print syscall entry if not in statistics or structured output mode
    if !statistics_mode && !structured_output {
        super::output::print_syscall_entry(
            child,
            name,
            syscall_num,
            arg1,
            arg2,
            arg3,
            &source_info,
            transpiler_map,
        );
    }

    // Extract function names for profiling
    let (function_name, caller_name) =
        extract_function_names(child, dwarf_ctx, &source_info, function_profiling_enabled);

    let json_source = source_info.as_ref().map(|src| crate::json_output::JsonSourceLocation {
        file: src.file.clone(),
        line: src.line,
        function: src.function.clone(),
    });

    // Return syscall entry data
    Ok(Some(SyscallEntry {
        name: name.to_string(),
        args,
        source: json_source,
        function_name,
        caller_name,
        // Sprint 26: Store raw args for decision trace capture
        raw_arg1: Some(arg1),
        raw_arg2: Some(arg2),
        _raw_arg3: Some(arg3),
    }))
}

/// Map DWARF source location to transpiler source (Sprint 24-28)
/// Returns a formatted string with the original source location if mapping is available
pub(super) fn map_to_transpiler_source(
    dwarf_source: &crate::dwarf::SourceLocation,
    transpiler_map: Option<&crate::transpiler_map::TranspilerMap>,
) -> Option<String> {
    if let Some(map) = transpiler_map {
        // Extract line number from DWARF source location
        let rust_line = dwarf_source.line as usize;

        // Look up in transpiler map
        if let Some(mapping) = map.lookup_line(rust_line) {
            // Format as "python_file:line in python_function"
            return Some(format!(
                "{}:{} in {} [{}]",
                map.source_file().display(),
                mapping.python_line,
                mapping.python_function,
                map.source_language()
            ));
        }
    }
    None
}

/// Read a null-terminated string from the tracee's memory
pub(super) fn read_string(child: Pid, addr: usize) -> Result<String> {
    use nix::sys::uio::{process_vm_readv, RemoteIoVec};
    use std::io::IoSliceMut;

    // Read up to 4096 bytes (max path length)
    let mut buf = vec![0u8; 4096];
    let mut local_iov = [IoSliceMut::new(&mut buf)];
    let remote_iov = [RemoteIoVec { base: addr, len: 4096 }];

    let bytes_read = process_vm_readv(child, &mut local_iov, &remote_iov)
        .context("Failed to read string from tracee memory")?;

    if bytes_read == 0 {
        anyhow::bail!("Read 0 bytes from tracee");
    }

    // Find null terminator
    let null_pos = buf.iter().position(|&b| b == 0).unwrap_or(bytes_read);

    // Convert to UTF-8 string (lossy - invalid UTF-8 will be replaced with \u{fffd})
    Ok(String::from_utf8_lossy(&buf[..null_pos]).to_string())
}

/// Record statistics for a syscall
fn record_stats_for_syscall(
    syscall_entry: &Option<SyscallEntry>,
    stats_tracker: Option<&mut crate::stats::StatsTracker>,
    result: i64,
    duration_us: u64,
) {
    if let (Some(entry), Some(tracker)) = (syscall_entry, stats_tracker) {
        tracker.record(&entry.name, result, duration_us);
    }
}

/// Record JSON output for a syscall
fn record_json_for_syscall(
    syscall_entry: &Option<SyscallEntry>,
    json_output: Option<&mut crate::json_output::JsonOutput>,
    result: i64,
    timing_mode: bool,
    duration_us: u64,
) {
    if let (Some(entry), Some(output)) = (syscall_entry, json_output) {
        let duration = if timing_mode && duration_us > 0 { Some(duration_us) } else { None };

        output.add_syscall(crate::json_output::JsonSyscall {
            name: entry.name.clone(),
            args: entry.args.clone(),
            result,
            duration_us: duration,
            source: entry.source.clone(),
        });
    }
}

/// Record CSV output for a syscall
fn record_csv_for_syscall(
    syscall_entry: &Option<SyscallEntry>,
    csv_output: Option<&mut crate::csv_output::CsvOutput>,
    result: i64,
    timing_mode: bool,
    duration_us: u64,
) {
    if let (Some(entry), Some(output)) = (syscall_entry, csv_output) {
        let duration = if timing_mode && duration_us > 0 { Some(duration_us) } else { None };

        // Format source location as "file:line" string
        let source_location = entry.source.as_ref().map(|src| {
            if let Some(func) = &src.function {
                format!("{}:{} in {}", src.file, src.line, func)
            } else {
                format!("{}:{}", src.file, src.line)
            }
        });

        // Format arguments as comma-separated string
        let arguments = entry.args.join(", ");

        output.add_syscall(crate::csv_output::CsvSyscall {
            name: entry.name.clone(),
            arguments,
            result,
            duration_us: duration,
            source_location,
        });
    }
}

/// Record HTML output for a syscall
fn record_html_for_syscall(
    syscall_entry: &Option<SyscallEntry>,
    html_output: Option<&mut crate::html_output::HtmlOutput>,
    result: i64,
    timing_mode: bool,
    duration_us: u64,
) {
    if let (Some(entry), Some(output)) = (syscall_entry, html_output) {
        let duration = if timing_mode && duration_us > 0 { Some(duration_us) } else { None };

        // Format source location as "file:line" string
        let source_location = entry.source.as_ref().map(|src| {
            if let Some(func) = &src.function {
                format!("{}:{} in {}", src.file, src.line, func)
            } else {
                format!("{}:{}", src.file, src.line)
            }
        });

        // Format arguments as comma-separated string
        let arguments = entry.args.join(", ");

        output.add_syscall(crate::html_output::HtmlSyscall {
            name: entry.name.clone(),
            arguments,
            result,
            duration_us: duration,
            source_location,
        });
    }
}

/// Record function profiling data
fn record_function_profiling(
    syscall_entry: &Option<SyscallEntry>,
    function_profiler: Option<&mut crate::function_profiler::FunctionProfiler>,
    duration_us: u64,
) {
    if let (Some(entry), Some(profiler)) = (syscall_entry, function_profiler) {
        if let Some(function_name) = &entry.function_name {
            profiler.record(function_name, &entry.name, duration_us, entry.caller_name.as_deref());
        }
    }
}

/// Handle real-time anomaly detection and alerts
fn handle_anomaly_detection(
    syscall_entry: &Option<SyscallEntry>,
    anomaly_detector: Option<&mut crate::anomaly::AnomalyDetector>,
    duration_us: u64,
) {
    if let (Some(entry), Some(detector)) = (syscall_entry, anomaly_detector) {
        if let Some(anomaly) = detector.record_and_check(&entry.name, duration_us) {
            // Print real-time anomaly alert to stderr
            let severity_label = match anomaly.severity {
                crate::anomaly::AnomalySeverity::Low => "\u{1f7e2} Low",
                crate::anomaly::AnomalySeverity::Medium => "\u{1f7e1} Medium",
                crate::anomaly::AnomalySeverity::High => "\u{1f534} High",
            };
            eprintln!(
                "\u{26a0}\u{fe0f}  ANOMALY: {} took {} \u{03bc}s ({:.1}\u{03c3} from baseline {:.1} \u{03bc}s) - {}",
                anomaly.syscall_name,
                anomaly.duration_us,
                anomaly.z_score.abs(),
                anomaly.baseline_mean,
                severity_label
            );
        }
    }
}

/// Sprint 26: Capture decision traces from `write()` syscalls to stderr
///
/// Intercepts write(2, buffer, count) calls and parses [DECISION] and [RESULT] lines
fn capture_decision_trace(
    child: Pid,
    syscall_entry: &Option<SyscallEntry>,
    decision_tracer: Option<&mut crate::decision_trace::DecisionTracer>,
    bytes_written: i64,
) {
    // Only process if decision tracing is enabled
    let Some(tracer) = decision_tracer else {
        return;
    };

    // Only process if we have a syscall entry
    let Some(entry) = syscall_entry else {
        return;
    };

    // Only intercept write() syscalls
    if entry.name != "write" {
        return;
    }

    // Check if writing to stderr (fd = 2)
    if entry.raw_arg1 != Some(2) {
        return;
    }

    // Only process successful writes
    if bytes_written <= 0 {
        return;
    }

    // Get buffer address and size
    let buffer_addr = entry.raw_arg2.unwrap_or(0);
    let buffer_size = bytes_written as usize;

    // Read buffer from child process memory
    use nix::sys::uio::{process_vm_readv, RemoteIoVec};
    use std::io::IoSliceMut;

    let mut buffer = vec![0u8; buffer_size];
    let mut local_iov = [IoSliceMut::new(&mut buffer)];
    let remote_iov = [RemoteIoVec { base: buffer_addr as usize, len: buffer_size }];

    // Try to read; silently ignore errors (child may have exited, etc.)
    if process_vm_readv(child, &mut local_iov, &remote_iov).is_err() {
        return;
    }

    // Convert to string, replacing invalid UTF-8 with replacement character
    let content = String::from_utf8_lossy(&buffer);

    // Parse each line through DecisionTracer
    for line in content.lines() {
        // Ignore parse errors - not all stderr lines are decision traces
        let _ = tracer.parse_line(line);
    }
}

/// Handle syscall exit - print return value and record statistics
pub(super) fn handle_syscall_exit(
    child: Pid,
    syscall_entry: &Option<SyscallEntry>,
    tracers: &mut Tracers,
    timing_mode: bool,
    duration_us: u64,
) -> Result<()> {
    let regs = crate::arch::PtraceRegs::get(child)?;
    let result = regs.syscall_return();

    // Check modes before borrowing
    let in_stats_mode = tracers.stats_tracker.is_some();
    let in_json_mode = tracers.json_output.is_some();
    let in_csv_mode = tracers.csv_output.is_some() || tracers.csv_stats_output.is_some();
    let in_html_mode = tracers.html_output.is_some();

    // Record statistics
    record_stats_for_syscall(syscall_entry, tracers.stats_tracker.as_mut(), result, duration_us);

    // Record JSON output
    record_json_for_syscall(
        syscall_entry,
        tracers.json_output.as_mut(),
        result,
        timing_mode,
        duration_us,
    );

    // Record CSV output
    record_csv_for_syscall(
        syscall_entry,
        tracers.csv_output.as_mut(),
        result,
        timing_mode,
        duration_us,
    );

    // Record HTML output
    record_html_for_syscall(
        syscall_entry,
        tracers.html_output.as_mut(),
        result,
        timing_mode,
        duration_us,
    );

    // Sprint 26: Capture decision traces from stderr writes
    capture_decision_trace(child, syscall_entry, tracers.decision_tracer.as_mut(), result);

    // Record CSV stats (we'll handle this in print_summaries)
    if let (Some(entry), Some(stats)) = (syscall_entry, tracers.csv_stats_output.as_mut()) {
        // CSV stats are accumulated in stats_tracker and printed at the end
        let _ = (entry, stats); // Suppress unused warning for now
    }

    // Record function profiling
    record_function_profiling(syscall_entry, tracers.function_profiler.as_mut(), duration_us);

    // Sprint 20: Real-time anomaly detection
    handle_anomaly_detection(syscall_entry, tracers.anomaly_detector.as_mut(), duration_us);

    // Sprint 52-55: Send event to visualizer
    if let (Some(entry), Some(ref sink)) = (syscall_entry, &tracers.visualizer_sink) {
        let event =
            VisualizerEvent { name: entry.name.clone(), duration_us, result, pid: child.as_raw() };
        // Non-blocking send - if channel is full, drop the event
        let _ = sink.send(event);
    }

    // Sprint 30: Record syscall to OTLP exporter
    #[cfg(feature = "otlp")]
    if let (Some(entry), Some(exporter)) = (syscall_entry, tracers.otlp_exporter.as_ref()) {
        let source_file = entry.source.as_ref().map(|s| s.file.as_str());
        let source_line = entry.source.as_ref().map(|s| s.line);

        exporter.record_syscall(
            &entry.name,
            if duration_us > 0 { Some(duration_us) } else { None },
            result,
            source_file,
            source_line,
        );
    }

    // Print result if not in statistics, JSON, CSV, or HTML mode
    if super::output::should_print_result(
        syscall_entry,
        in_stats_mode,
        in_json_mode,
        in_csv_mode,
        in_html_mode,
    ) {
        super::output::print_syscall_result(result, timing_mode, duration_us);
    }

    Ok(())
}

// Test-accessible wrappers for private helper functions
#[cfg(test)]
pub(super) fn record_stats_for_syscall_test(
    syscall_entry: &Option<SyscallEntry>,
    stats_tracker: Option<&mut crate::stats::StatsTracker>,
    result: i64,
    duration_us: u64,
) {
    record_stats_for_syscall(syscall_entry, stats_tracker, result, duration_us);
}

#[cfg(test)]
pub(super) fn record_function_profiling_test(
    syscall_entry: &Option<SyscallEntry>,
    function_profiler: Option<&mut crate::function_profiler::FunctionProfiler>,
    duration_us: u64,
) {
    record_function_profiling(syscall_entry, function_profiler, duration_us);
}

#[cfg(test)]
pub(super) fn handle_anomaly_detection_test(
    syscall_entry: &Option<SyscallEntry>,
    anomaly_detector: Option<&mut crate::anomaly::AnomalyDetector>,
    duration_us: u64,
) {
    handle_anomaly_detection(syscall_entry, anomaly_detector, duration_us);
}
