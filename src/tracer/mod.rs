//! System call tracing using ptrace
//!
//! Sprint 3-4: Trace all syscalls with name resolution

mod ml_analysis;
mod output;
mod syscall_handling;

use anyhow::{Context, Result};
use nix::sys::ptrace;
use nix::sys::wait::{waitpid, WaitStatus};
use nix::unistd::{fork, ForkResult, Pid};
use std::os::unix::process::CommandExt;
use std::process::Command;
use tracing::{info, trace, warn};

/// Real-time syscall event for visualization (Sprint 52-55)
#[derive(Debug, Clone)]
pub struct VisualizerEvent {
    /// Syscall name
    pub name: String,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Return value (negative for errors)
    pub result: i64,
    /// Process ID
    pub pid: i32,
}

/// Configuration for tracer behavior
pub struct TracerConfig {
    pub enable_source: bool,
    pub filter: crate::filter::SyscallFilter,
    pub statistics_mode: bool,
    pub timing_mode: bool,
    pub output_format: crate::cli::OutputFormat,
    pub follow_forks: bool,
    pub profile_self: bool,
    pub function_time: bool,
    pub stats_extended: bool,   // Sprint 19: Extended statistics with Trueno
    pub anomaly_threshold: f32, // Sprint 19: Anomaly detection threshold (sigma)
    pub anomaly_realtime: bool, // Sprint 20: Real-time anomaly detection
    pub anomaly_window_size: usize, // Sprint 20: Sliding window size
    pub hpu_analysis: bool,     // Sprint 21: HPU-accelerated analysis (GPU if available)
    pub hpu_cpu_only: bool,     // Sprint 21: Force CPU backend (disable GPU)
    pub ml_anomaly: bool,       // Sprint 23: ML-based anomaly detection using Aprender
    pub ml_clusters: usize,     // Sprint 23: Number of clusters for KMeans
    pub ml_compare: bool,       // Sprint 23: Compare ML results with z-score
    pub ml_outliers: bool,      // Sprint 22: Isolation Forest outlier detection
    pub ml_outlier_threshold: f32, // Sprint 22: Contamination threshold
    pub ml_outlier_trees: usize, // Sprint 22: Number of trees
    pub explain: bool,          // Sprint 22: Enable explainability
    pub dl_anomaly: bool,       // Sprint 23: Deep Learning Autoencoder anomaly detection
    pub dl_threshold: f32,      // Sprint 23: Reconstruction error threshold (sigma multiplier)
    pub dl_hidden_size: usize,  // Sprint 23: Autoencoder hidden layer size
    pub dl_epochs: usize,       // Sprint 23: Training epochs
    pub trace_transpiler_decisions: bool, // Sprint 26: Trace transpiler compile-time decisions
    pub transpiler_map: Option<crate::transpiler_map::TranspilerMap>, // Sprint 24-28: Transpiler source mapping
    pub otlp_endpoint: Option<String>, // Sprint 30: OpenTelemetry OTLP endpoint
    pub otlp_service_name: String,     // Sprint 30: Service name for OTLP traces
    pub trace_parent: Option<String>,  // Sprint 33: W3C Trace Context for distributed tracing
    pub chaos_config: Option<crate::chaos::ChaosConfig>, // Sprint 47: Chaos engineering (Issue #17)
    /// Sprint 52-55: Event sink for real-time visualization
    pub visualizer_sink: Option<std::sync::mpsc::Sender<VisualizerEvent>>,
}

impl Default for TracerConfig {
    fn default() -> Self {
        Self {
            enable_source: false,
            filter: crate::filter::SyscallFilter::all(),
            statistics_mode: false,
            timing_mode: false,
            output_format: crate::cli::OutputFormat::default(),
            follow_forks: false,
            profile_self: false,
            function_time: false,
            stats_extended: false,
            anomaly_threshold: 2.0,
            anomaly_realtime: false,
            anomaly_window_size: 100,
            hpu_analysis: false,
            hpu_cpu_only: false,
            ml_anomaly: false,
            ml_clusters: 5,
            ml_compare: false,
            ml_outliers: false,
            ml_outlier_threshold: 0.1,
            ml_outlier_trees: 100,
            explain: false,
            dl_anomaly: false,
            dl_threshold: 2.0,
            dl_hidden_size: 8,
            dl_epochs: 50,
            trace_transpiler_decisions: false,
            transpiler_map: None,
            otlp_endpoint: None,
            otlp_service_name: "renacer".to_string(),
            trace_parent: None,
            chaos_config: None,
            visualizer_sink: None,
        }
    }
}

/// Sprint 9-10 Scope
/// - `-p PID` flag to attach to running processes
/// - Uses `PTRACE_ATTACH` instead of `fork()` + `PTRACE_TRACEME`
pub fn attach_to_pid(pid: i32, config: TracerConfig) -> Result<i32> {
    let pid = Pid::from_raw(pid);

    // Attach to the running process
    ptrace::attach(pid).context(format!("Failed to attach to PID {pid}"))?;

    // Wait for SIGSTOP from PTRACE_ATTACH
    waitpid(pid, None).context("Failed to wait for attach signal")?;

    eprintln!("[renacer: Attached to process {pid}]");

    // Use the same tracing logic as trace_command
    trace_child(pid, config)
}

/// Sprint 9-10 Scope
/// - Syscall filtering via -e trace= expressions
/// - Statistics mode via -c flag
/// - Timing per syscall via -T flag
/// - JSON output via --format json
/// - Fork following via -f flag
#[allow(unsafe_code)]
pub fn trace_command(command: &[String], config: TracerConfig) -> Result<i32> {
    if command.is_empty() {
        anyhow::bail!("Command array is empty");
    }

    let program = &command[0];
    let args = &command[1..];

    // Sprint 47: Clone chaos config for child process (Issue #17)
    let chaos_config = config.chaos_config.clone();

    // Fork: parent will trace, child will exec
    // SAFETY: fork() is safe to call; we handle both parent and child cases properly
    match unsafe { fork() }.context("Failed to fork")? {
        ForkResult::Parent { child } => trace_child(child, config),
        ForkResult::Child => {
            // Child: allow tracing and exec target program
            ptrace::traceme().context("Failed to PTRACE_TRACEME")?;

            // Sprint 47: Apply chaos resource limits before exec (Issue #17)
            if let Some(ref chaos) = chaos_config {
                if let Err(e) = chaos.apply_limits() {
                    eprintln!("[renacer: Warning: Failed to apply chaos limits: {e}]");
                }
            }

            // Use std::process::Command for exec
            let err = Command::new(program).args(args).exec();

            // If we get here, exec failed
            eprintln!("Failed to exec {program}: {err}");
            std::process::exit(1);
        }
    }
}

/// Tracers and profilers used during tracing
struct Tracers {
    profiling_ctx: Option<crate::profiling::ProfilingContext>,
    function_profiler: Option<crate::function_profiler::FunctionProfiler>,
    stats_tracker: Option<crate::stats::StatsTracker>,
    json_output: Option<crate::json_output::JsonOutput>,
    csv_output: Option<crate::csv_output::CsvOutput>,
    csv_stats_output: Option<crate::csv_output::CsvStatsOutput>,
    html_output: Option<crate::html_output::HtmlOutput>, // Sprint 22
    anomaly_detector: Option<crate::anomaly::AnomalyDetector>, // Sprint 20
    decision_tracer: Option<crate::decision_trace::DecisionTracer>, // Sprint 26
    #[cfg(feature = "otlp")]
    otlp_exporter: Option<crate::otlp_exporter::OtlpExporter>, // Sprint 30
    /// Sprint 52-55: Event sink for real-time visualization
    visualizer_sink: Option<std::sync::mpsc::Sender<VisualizerEvent>>,
}

/// Initialize profiling-related tracers
fn initialize_profiling_tracers(
    config: &TracerConfig,
) -> (
    Option<crate::profiling::ProfilingContext>,
    Option<crate::function_profiler::FunctionProfiler>,
    Option<crate::anomaly::AnomalyDetector>,
) {
    let profiling_ctx =
        if config.profile_self { Some(crate::profiling::ProfilingContext::new()) } else { None };

    let function_profiler = if config.function_time {
        Some(crate::function_profiler::FunctionProfiler::new())
    } else {
        None
    };

    let anomaly_detector = if config.anomaly_realtime {
        Some(crate::anomaly::AnomalyDetector::new(
            config.anomaly_window_size,
            config.anomaly_threshold,
        ))
    } else {
        None
    };

    (profiling_ctx, function_profiler, anomaly_detector)
}

/// Initialize output format tracers (JSON, CSV, HTML)
fn initialize_output_tracers(
    config: &TracerConfig,
) -> (
    Option<crate::json_output::JsonOutput>,
    Option<crate::csv_output::CsvOutput>,
    Option<crate::csv_output::CsvStatsOutput>,
    Option<crate::html_output::HtmlOutput>,
) {
    use crate::cli::OutputFormat;

    let json_output = if matches!(config.output_format, OutputFormat::Json) {
        Some(crate::json_output::JsonOutput::new())
    } else {
        None
    };

    let csv_output = if matches!(config.output_format, OutputFormat::Csv) && !config.statistics_mode
    {
        Some(crate::csv_output::CsvOutput::new(config.timing_mode, config.enable_source))
    } else {
        None
    };

    let csv_stats_output =
        if matches!(config.output_format, OutputFormat::Csv) && config.statistics_mode {
            Some(crate::csv_output::CsvStatsOutput::new())
        } else {
            None
        };

    let html_output = if matches!(config.output_format, OutputFormat::Html) {
        Some(crate::html_output::HtmlOutput::new(config.timing_mode, config.enable_source))
    } else {
        None
    };

    (json_output, csv_output, csv_stats_output, html_output)
}

/// Initialize all tracers and profilers based on config
fn initialize_tracers(config: &TracerConfig) -> Tracers {
    // Initialize profiling tracers
    let (profiling_ctx, function_profiler, anomaly_detector) = initialize_profiling_tracers(config);

    // Initialize output format tracers
    let (json_output, csv_output, csv_stats_output, html_output) =
        initialize_output_tracers(config);

    // Create stats_tracker if statistics mode is enabled OR if ML/DL anomaly analysis is enabled
    let stats_tracker =
        if config.statistics_mode || config.ml_anomaly || config.ml_outliers || config.dl_anomaly {
            Some(crate::stats::StatsTracker::new())
        } else {
            None
        };

    // Initialize decision tracer for transpiler decision tracking (Sprint 26)
    let decision_tracer = if config.trace_transpiler_decisions {
        Some(crate::decision_trace::DecisionTracer::new())
    } else {
        None
    };

    // Initialize OTLP exporter if endpoint is provided (Sprint 30)
    #[cfg(feature = "otlp")]
    let otlp_exporter = if let Some(ref endpoint) = config.otlp_endpoint {
        // Sprint 33: Extract trace context from CLI flag or environment
        use crate::trace_context::TraceContext;

        let trace_context = config
            .trace_parent
            .as_ref()
            .and_then(|s| TraceContext::parse(s).ok())
            .or_else(TraceContext::from_env);

        if trace_context.is_some() {
            eprintln!("[renacer: Distributed tracing enabled - joining parent trace]");
        }

        match crate::otlp_exporter::OtlpExporter::new(
            crate::otlp_exporter::OtlpConfig::new(
                endpoint.clone(),
                config.otlp_service_name.clone(),
            ),
            trace_context,
        ) {
            Ok(exporter) => {
                eprintln!("[renacer: OTLP export enabled to {endpoint}]");
                Some(exporter)
            }
            Err(e) => {
                eprintln!("[renacer: OTLP initialization failed: {e}]");
                None
            }
        }
    } else {
        None
    };

    Tracers {
        profiling_ctx,
        function_profiler,
        stats_tracker,
        json_output,
        csv_output,
        csv_stats_output,
        html_output,
        anomaly_detector,
        decision_tracer,
        #[cfg(feature = "otlp")]
        otlp_exporter,
        visualizer_sink: config.visualizer_sink.clone(),
    }
}

/// Initialize ptrace options for the child process
fn setup_ptrace_options(child: Pid, follow_forks: bool) -> Result<()> {
    setup_ptrace_options_internal(child, follow_forks, true)
}

/// Initialize ptrace options with optional initial wait
fn setup_ptrace_options_internal(child: Pid, follow_forks: bool, wait_first: bool) -> Result<()> {
    // Wait for initial SIGSTOP (from PTRACE_TRACEME or fork event)
    if wait_first {
        trace!(pid = %child, "waiting for initial SIGSTOP");
        let status = waitpid(child, None).context("Failed to wait for child")?;
        trace!(pid = %child, status = ?status, "initial wait completed");
    }

    // Set ptrace options to trace syscalls
    let mut options = ptrace::Options::PTRACE_O_TRACESYSGOOD | ptrace::Options::PTRACE_O_EXITKILL;

    // Add fork following options if enabled
    if follow_forks {
        options |= ptrace::Options::PTRACE_O_TRACEFORK
            | ptrace::Options::PTRACE_O_TRACEVFORK
            | ptrace::Options::PTRACE_O_TRACECLONE;
    }

    trace!(pid = %child, "setting ptrace options");
    ptrace::setoptions(child, options).context("Failed to set ptrace options")?;
    trace!(pid = %child, "ptrace options set");

    // Continue the child to start syscall tracing
    trace!(pid = %child, "sending initial PTRACE_SYSCALL");
    ptrace::syscall(child, None).context("Failed to continue child with PTRACE_SYSCALL")?;
    trace!(pid = %child, "initial PTRACE_SYSCALL sent");

    Ok(())
}

/// Load DWARF debug info for source correlation
fn load_dwarf_context(child: Pid) -> Option<crate::dwarf::DwarfContext> {
    if let Ok(exe_path) = std::fs::read_link(format!("/proc/{child}/exe")) {
        match crate::dwarf::DwarfContext::load(&exe_path) {
            Ok(ctx) => {
                eprintln!("[renacer: DWARF debug info loaded from {}]", exe_path.display());
                Some(ctx)
            }
            Err(e) => {
                eprintln!("[renacer: Warning - failed to load DWARF: {e}]");
                eprintln!("[renacer: Continuing without source correlation]");
                None
            }
        }
    } else {
        None
    }
}

/// Handle ptrace fork/vfork/clone events (Sprint 18: Multi-process tracing)
fn handle_ptrace_event(
    pid: Pid,
    event: i32,
    processes: &mut std::collections::HashMap<Pid, ProcessState>,
    config: &TracerConfig,
) -> Result<()> {
    use nix::libc;

    // Check if this is a fork/vfork/clone event
    match event {
        libc::PTRACE_EVENT_FORK | libc::PTRACE_EVENT_VFORK | libc::PTRACE_EVENT_CLONE => {
            // Extract the new child PID
            let new_pid_raw = ptrace::getevent(pid)
                .context("Failed to get event message for fork/vfork/clone")?;
            let new_pid = Pid::from_raw(new_pid_raw as i32);

            // Wait for the new child to stop
            let wait_status = waitpid(new_pid, None).context("Failed to wait for new child")?;

            // Check if child is still alive and can be continued
            match wait_status {
                WaitStatus::Exited(_, _) | WaitStatus::Signaled(_, _, _) => {
                    // Child already exited, nothing to continue
                    eprintln!("[renacer: Process {pid} forked child {new_pid} (already exited)]");
                }
                _ => {
                    // Setup ptrace options for the new child (already waited)
                    if let Err(e) =
                        setup_ptrace_options_internal(new_pid, config.follow_forks, false)
                    {
                        // Child may have exited between waitpid and setoptions
                        warn!("Failed to setup ptrace options for child {}: {}", new_pid, e);
                        return Ok(());
                    }

                    // Add to tracking
                    processes.insert(new_pid, ProcessState::new());

                    // Continue the new child process
                    // Handle ESRCH gracefully - child may have exited between waitpid and syscall
                    match ptrace::syscall(new_pid, None) {
                        Ok(()) => {
                            eprintln!("[renacer: Process {pid} forked child {new_pid}]");
                        }
                        Err(nix::errno::Errno::ESRCH) => {
                            // Child already exited, remove from tracking
                            processes.remove(&new_pid);
                            eprintln!(
                                "[renacer: Process {pid} forked child {new_pid} (exited immediately)]"
                            );
                        }
                        Err(e) => {
                            return Err(anyhow::anyhow!("Failed to continue new child: {e}"));
                        }
                    }
                }
            }
        }
        _ => {
            // Unknown ptrace event, ignore
        }
    }

    Ok(())
}

/// Handle syscall event (entry or exit)
fn handle_syscall_event(
    child: Pid,
    in_syscall: &mut bool,
    current_syscall_entry: &mut Option<SyscallEntry>,
    syscall_entry_time: &mut Option<std::time::Instant>,
    dwarf_ctx: Option<&crate::dwarf::DwarfContext>,
    config: &TracerConfig,
    tracers: &mut Tracers,
) -> Result<()> {
    // Check if we're in a structured output mode (JSON, CSV, HTML) to suppress text output
    let in_json_mode = tracers.json_output.is_some();
    let in_csv_mode = tracers.csv_output.is_some() || tracers.csv_stats_output.is_some();
    let in_html_mode = tracers.html_output.is_some();
    let structured_output = in_json_mode || in_csv_mode || in_html_mode;

    if *in_syscall {
        // Syscall exit - calculate duration
        let duration_us =
            syscall_entry_time.map(|start| start.elapsed().as_micros() as u64).unwrap_or(0);

        process_syscall_exit(
            child,
            current_syscall_entry,
            tracers,
            config.timing_mode,
            duration_us,
        )?;

        *current_syscall_entry = None;
        *syscall_entry_time = None;
        *in_syscall = false;
    } else {
        // Syscall entry - record start time if timing enabled
        if config.timing_mode || config.statistics_mode || structured_output {
            *syscall_entry_time = Some(std::time::Instant::now());
        }

        *current_syscall_entry = process_syscall_entry(
            child,
            dwarf_ctx,
            config,
            tracers.profiling_ctx.as_mut(),
            structured_output,
        )?;
        *in_syscall = true;
    }
    Ok(())
}

/// Process syscall entry event
fn process_syscall_entry(
    child: Pid,
    dwarf_ctx: Option<&crate::dwarf::DwarfContext>,
    config: &TracerConfig,
    profiling_ctx: Option<&mut crate::profiling::ProfilingContext>,
    structured_output: bool,
) -> Result<Option<SyscallEntry>> {
    if let Some(prof) = profiling_ctx {
        prof.measure(crate::profiling::ProfilingCategory::Other, || {
            syscall_handling::handle_syscall_entry(
                child,
                dwarf_ctx,
                &config.filter,
                config.statistics_mode,
                structured_output,
                config.function_time,
                config.transpiler_map.as_ref(),
            )
        })
    } else {
        syscall_handling::handle_syscall_entry(
            child,
            dwarf_ctx,
            &config.filter,
            config.statistics_mode,
            structured_output,
            config.function_time,
            config.transpiler_map.as_ref(),
        )
    }
}

/// Process syscall exit event
fn process_syscall_exit(
    child: Pid,
    current_syscall_entry: &Option<SyscallEntry>,
    tracers: &mut Tracers,
    timing_mode: bool,
    duration_us: u64,
) -> Result<()> {
    // Check if profiling is enabled and handle accordingly
    if let Some(mut prof) = tracers.profiling_ctx.take() {
        // Temporarily take profiling_ctx out to avoid borrow conflict
        let result = prof.measure(crate::profiling::ProfilingCategory::Other, || {
            syscall_handling::handle_syscall_exit(
                child,
                current_syscall_entry,
                tracers,
                timing_mode,
                duration_us,
            )
        });
        prof.record_syscall();
        // Put profiling_ctx back
        tracers.profiling_ctx = Some(prof);
        result
    } else {
        syscall_handling::handle_syscall_exit(
            child,
            current_syscall_entry,
            tracers,
            timing_mode,
            duration_us,
        )
    }
}

/// Analysis configuration for `print_summaries`
struct AnalysisConfig {
    stats_extended: bool,
    anomaly_threshold: f32,
    hpu_analysis: bool,
    hpu_cpu_only: bool,
    ml_anomaly: bool,
    ml_clusters: usize,
    ml_compare: bool,
    ml_outliers: bool,         // Sprint 22: Isolation Forest outlier detection
    ml_outlier_threshold: f32, // Sprint 22: Contamination threshold
    ml_outlier_trees: usize,   // Sprint 22: Number of trees
    dl_anomaly: bool,          // Sprint 23: Deep Learning Autoencoder anomaly detection
    dl_threshold: f32,         // Sprint 23: Reconstruction error threshold (sigma multiplier)
    dl_hidden_size: usize,     // Sprint 23: Hidden layer size
    dl_epochs: usize,          // Sprint 23: Training epochs
    explain: bool,             // Sprint 22/23: Enable explainability
}

/// Per-process state for multi-process tracing
#[derive(Debug)]
struct ProcessState {
    in_syscall: bool,
    current_syscall_entry: Option<SyscallEntry>,
    syscall_entry_time: Option<std::time::Instant>,
    dwarf_ctx: Option<crate::dwarf::DwarfContext>,
    dwarf_loaded: bool,
}

impl ProcessState {
    fn new() -> Self {
        Self {
            in_syscall: false,
            current_syscall_entry: None,
            syscall_entry_time: None,
            dwarf_ctx: None,
            dwarf_loaded: false,
        }
    }
}

/// Handle wait status and update process tracking
fn handle_traced_process_status(
    status: WaitStatus,
    processes: &mut std::collections::HashMap<Pid, ProcessState>,
    main_pid: Pid,
    main_exit_code: &mut i32,
    config: &TracerConfig,
) -> Result<Option<Pid>> {
    match status {
        WaitStatus::Exited(p, code) => {
            processes.remove(&p);
            if p == main_pid {
                *main_exit_code = code;
            }
            Ok(None)
        }
        WaitStatus::Signaled(p, sig, _) => {
            eprintln!("Process {p} killed by signal: {sig:?}");
            processes.remove(&p);
            if p == main_pid {
                *main_exit_code = 128 + sig as i32;
            }
            Ok(None)
        }
        WaitStatus::PtraceSyscall(p) => Ok(Some(p)),
        WaitStatus::PtraceEvent(p, _sig, event) => {
            handle_ptrace_event(p, event, processes, config)?;
            ptrace::syscall(p, None).context("Failed to PTRACE_SYSCALL after event")?;
            Ok(None)
        }
        _ => {
            if let Some(p) = status.pid() {
                ptrace::syscall(p, None).ok();
            }
            Ok(None)
        }
    }
}

/// Process a single syscall event for a traced PID
fn process_syscall_for_pid(
    pid: Pid,
    processes: &mut std::collections::HashMap<Pid, ProcessState>,
    config: &TracerConfig,
    tracers: &mut Tracers,
) -> Result<()> {
    let state = if let Some(s) = processes.get_mut(&pid) {
        s
    } else {
        ptrace::syscall(pid, None).ok();
        return Ok(());
    };

    // Load DWARF context on first syscall if needed
    if config.enable_source && !state.dwarf_loaded {
        state.dwarf_loaded = true;
        state.dwarf_ctx = load_dwarf_context(pid);
    }

    // Handle syscall entry/exit
    handle_syscall_event(
        pid,
        &mut state.in_syscall,
        &mut state.current_syscall_entry,
        &mut state.syscall_entry_time,
        state.dwarf_ctx.as_ref(),
        config,
        tracers,
    )?;

    ptrace::syscall(pid, None).context("Failed to PTRACE_SYSCALL")
}

/// Start the OTLP root span if the feature is enabled and exporter is configured.
#[cfg(feature = "otlp")]
fn start_otlp_root_span(tracers: &mut Tracers, child: Pid) {
    if let Some(ref mut exporter) = tracers.otlp_exporter {
        let program_name = std::fs::read_to_string(format!("/proc/{child}/cmdline"))
            .ok()
            .and_then(|s| s.split('\0').next().map(std::string::ToString::to_string))
            .unwrap_or_else(|| format!("pid:{child}"));
        exporter.start_root_span(&program_name, child.as_raw());
    }
}

/// Build analysis configuration from tracer configuration.
fn build_analysis_config(config: &TracerConfig) -> AnalysisConfig {
    AnalysisConfig {
        stats_extended: config.stats_extended,
        anomaly_threshold: config.anomaly_threshold,
        hpu_analysis: config.hpu_analysis,
        hpu_cpu_only: config.hpu_cpu_only,
        ml_anomaly: config.ml_anomaly,
        ml_clusters: config.ml_clusters,
        ml_compare: config.ml_compare,
        ml_outliers: config.ml_outliers,
        ml_outlier_threshold: config.ml_outlier_threshold,
        ml_outlier_trees: config.ml_outlier_trees,
        dl_anomaly: config.dl_anomaly,
        dl_threshold: config.dl_threshold,
        dl_hidden_size: config.dl_hidden_size,
        dl_epochs: config.dl_epochs,
        explain: config.explain,
    }
}

/// Wait for next process event, returning the wait status.
/// Returns `Ok(None)` when all processes have exited.
fn wait_for_event(
    config: &TracerConfig,
    child: Pid,
    processes: &std::collections::HashMap<Pid, ProcessState>,
) -> Result<Option<nix::sys::wait::WaitStatus>> {
    let wait_result =
        if config.follow_forks { waitpid(Pid::from_raw(-1), None) } else { waitpid(child, None) };

    match wait_result {
        Ok(s) => {
            trace!(status = ?s, "waitpid returned");
            Ok(Some(s))
        }
        Err(_) if processes.is_empty() => {
            trace!("waitpid error but processes empty, breaking");
            Ok(None)
        }
        Err(e) => {
            warn!(error = %e, "waitpid failed");
            Err(e).context("Failed to waitpid")
        }
    }
}

/// Trace a child process, filtering syscalls based on filter
fn trace_child(child: Pid, config: TracerConfig) -> Result<i32> {
    info!(pid = %child, "starting trace_child");

    let mut tracers = initialize_tracers(&config);
    trace!("tracers initialized");

    #[cfg(feature = "otlp")]
    start_otlp_root_span(&mut tracers, child);

    trace!("calling setup_ptrace_options");
    setup_ptrace_options(child, config.follow_forks)?;
    trace!("ptrace options set successfully");

    use std::collections::HashMap;
    let mut processes: HashMap<Pid, ProcessState> = HashMap::new();
    processes.insert(child, ProcessState::new());

    let main_pid = child;
    let mut main_exit_code = 0;

    info!("entering main wait loop");
    while !processes.is_empty() {
        trace!(num_processes = processes.len(), "calling waitpid");
        let Some(status) = wait_for_event(&config, child, &processes)? else {
            break;
        };

        let pid = if let Some(p) = handle_traced_process_status(
            status,
            &mut processes,
            main_pid,
            &mut main_exit_code,
            &config,
        )? {
            trace!(pid = %p, "handle_traced_process_status returned pid");
            p
        } else {
            trace!("handle_traced_process_status returned None, continuing");
            continue;
        };

        trace!(pid = %pid, "calling process_syscall_for_pid");
        process_syscall_for_pid(pid, &mut processes, &config, &mut tracers)?;
        trace!(pid = %pid, "process_syscall_for_pid completed");
    }

    info!("exited main wait loop");

    output::print_summaries(
        tracers,
        config.timing_mode,
        main_exit_code,
        &build_analysis_config(&config),
    );
    Ok(main_exit_code)
}

/// Syscall entry data for JSON output
#[derive(Debug)]
struct SyscallEntry {
    name: String,
    args: Vec<String>,
    source: Option<crate::json_output::JsonSourceLocation>,
    function_name: Option<String>,
    caller_name: Option<String>,
    // Sprint 26: Raw args for decision trace capture (write syscall interception)
    raw_arg1: Option<u64>,
    raw_arg2: Option<u64>,
    _raw_arg3: Option<u64>,
}

#[cfg(test)]
#[path = "core_tests.rs"]
mod core_tests;
