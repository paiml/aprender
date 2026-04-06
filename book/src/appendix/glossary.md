# Glossary

Technical terms and concepts used throughout Renacer documentation.

## Syscall Tracing

**System Call (Syscall)**
: A mechanism that allows user-space programs to request services from the operating system kernel. Examples: `read()`, `write()`, `open()`, `close()`.

**ptrace**
: Linux system call (`ptrace(2)`) used for process tracing and debugging. Renacer uses ptrace to intercept and monitor syscalls made by target processes.

**Tracee**
: The process being traced by Renacer (the target process).

**Tracer**
: The Renacer process that attaches to and monitors the tracee.

**Attach**
: The operation of connecting Renacer to an already-running process using `ptrace(PTRACE_ATTACH)`. See `renacer -p PID`.

**Fork Following**
: Automatically tracing child processes created by the target process. Enabled with `-f` flag.

## Debug Information

**DWARF**
: Debugging With Attributed Record Formats - a standardized debugging data format used by compilers (gcc, clang, rustc) to embed source-level information in binaries.

**Debug Symbols**
: Metadata embedded in binaries that map machine code back to source code (file names, line numbers, function names). Generated with `-g` flag during compilation.

**Frame Pointer**
: A CPU register (rbp on x86_64) that points to the current stack frame. Used for stack unwinding. Enable with `-fno-omit-frame-pointer`.

**Stack Unwinding**
: The process of walking up the call stack to reconstruct the sequence of function calls. Renacer uses frame pointer chain walking (max 64 frames).

**Source Correlation**
: Mapping syscalls back to specific source code locations using DWARF debug info. Enabled with `--source` flag.

## Filtering

**Syscall Filter**
: Rules for selecting which syscalls to trace. Specified with `-e trace=...` syntax.

**Syscall Class**
: Predefined groups of related syscalls (e.g., `file`, `network`, `ipc`, `desc`). See [Syscall Classes](../core-concepts/filtering-classes.md).

**Negation Operator**
: The `!` prefix to exclude specific syscalls from tracing. Example: `-e trace=file,!openat`.

**Regex Pattern**
: Regular expression for matching syscall names, enclosed in slashes `/pattern/`. Example: `-e trace=/^open.*/`.

## Performance Analysis

**Function Profiling**
: Attributing syscall execution time to specific functions using DWARF correlation. Enabled with `--function-time` flag.

**I/O Bottleneck**
: Slow I/O operations (>1ms threshold) that degrade performance. Tracked syscalls: `read`, `write`, `fsync`, `openat`, etc.

**Percentile (p50, p95, p99)**
: Statistical measure indicating the value below which a percentage of observations fall. p99 = 99% of syscalls complete within this time.

**Tail Latency**
: Performance outliers at the high end of the latency distribution (p99, p99.9). Often indicate systemic issues.

**Anomaly Detection**
: Identifying unusual syscall patterns via statistical methods (Z-score, IQR) or real-time monitoring.

**Z-score**
: Number of standard deviations a value is from the mean. Values >3σ are typically considered outliers.

**IQR (Interquartile Range)**
: Q3 - Q1, used for robust outlier detection. Outliers: values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR].

## Statistics

**SIMD (Single Instruction, Multiple Data)**
: CPU instructions that process multiple data elements in parallel (4-8× speedup). Used for percentile calculations via NumPy/AVX2.

**HPU (Hardware Processing Unit)**
: Generic term for GPU/TPU acceleration. Renacer uses HPU for matrix operations in statistical analysis (Sprint 21).

**Correlation Matrix**
: Matrix showing pairwise correlation coefficients between syscall durations. Identifies related operations.

**K-means Clustering**
: Unsupervised learning algorithm that groups syscalls into K clusters based on features (duration, frequency). Used for pattern discovery.

## Output Formats

**Text Format**
: Human-readable strace-like output (default). Example: `openat(AT_FDCWD, "file", O_RDONLY) = 3`.

**JSON Format**
: Machine-parsable structured output (`--format json`). Ideal for post-processing with jq, Python pandas.

**CSV Format**
: Comma-separated values (`--format csv`). Compatible with spreadsheets (Excel, LibreOffice) and R.

**HTML Format**
: Interactive visual reports (`--format html`). Includes charts, tables, color-coded statistics (Sprint 22).

## Quality Engineering

**EXTREME TDD**
: Test-Driven Development methodology emphasizing RED-GREEN-REFACTOR cycle, 85%+ coverage, mutation testing.

**Property-Based Testing**
: Testing approach using randomly generated inputs to verify invariants. Implemented with `proptest` crate (18 comprehensive tests).

**Mutation Testing**
: Testing technique that modifies code to verify tests catch defects. Tool: `cargo-mutants`.

**Fuzz Testing**
: Automated testing using malformed/random inputs to find edge cases. Applied to filter parser (Sprint 29).

**Chaos Engineering**
: Injecting failures (file not found, permission denied) to verify error handling robustness (Sprint 29).

**Quality Gates**
: Automated pre-commit checks: format, clippy, bashrs, property tests, security audit (completes in ~2s).

## Sprint Milestones

**Sprint 13** - Function Profiling with DWARF correlation
**Sprint 15** - Negation operator for advanced filtering
**Sprint 16** - Regex pattern matching for syscall filtering
**Sprint 18** - Multi-process tracing with fork following
**Sprint 19** - Enhanced statistics with percentiles
**Sprint 20** - Anomaly detection (post-hoc and real-time)
**Sprint 21** - HPU acceleration for statistical analysis
**Sprint 22** - HTML output format with visual reports
**Sprint 23** - ML-based anomaly detection via Aprender
**Sprint 29** - Chaos engineering and fuzz testing

See [CHANGELOG](./changelog.md) for detailed sprint history.
