/// Format a cell result as a status string
fn format_cell_status(result: &CellResult) -> String {
    if result.passed() {
        format!(
            "{}✓ {}/{}{}  ",
            GREEN, result.total_points, result.max_points, NC
        )
    } else {
        format!(
            "{}✗ {}/{}{}  ",
            RED, result.total_points, result.max_points, NC
        )
    }
}

/// Print a single row (CPU or GPU) of the QA matrix
fn print_matrix_row(label: &str, backend: Backend, results: &[CellResult]) {
    print!("{}║{} {:^10} │", MAGENTA, NC, label);
    for fmt in [Format::Gguf, Format::SafeTensors, Format::Apr] {
        if let Some(r) = results
            .iter()
            .find(|r| r.cell.backend == backend && r.cell.format == fmt)
        {
            print!(" {:^12} │", format_cell_status(r));
        } else {
            print!(" {:^12} │", "—");
        }
    }
    println!("{}║{}", MAGENTA, NC);
}

/// Compute letter grade from points ratio
fn compute_grade(total_points: u32, max_points: u32) -> &'static str {
    if total_points == max_points {
        "A+"
    } else {
        let ratio = total_points as f64 / max_points as f64;
        if ratio >= 0.9 {
            "A"
        } else if ratio >= 0.8 {
            "B"
        } else if ratio >= 0.7 {
            "C"
        } else {
            "F"
        }
    }
}

fn print_matrix_summary(results: &[CellResult]) {
    println!();
    println!(
        "{}╔═════════════════════════════════════════════════════════════╗{}",
        MAGENTA, NC
    );
    println!(
        "{}║             QA MATRIX SUMMARY (PMAT-QA-MATRIX-001)          ║{}",
        MAGENTA, NC
    );
    println!(
        "{}╠═════════════════════════════════════════════════════════════╣{}",
        MAGENTA, NC
    );
    println!(
        "{}║{} {:^10} │ {:^12} │ {:^12} │ {:^12} {}║{}",
        MAGENTA, NC, "", "GGUF", "SafeTensors", "APR", MAGENTA, NC
    );
    println!(
        "{}╟───────────┼──────────────┼──────────────┼──────────────╢{}",
        MAGENTA, NC
    );

    print_matrix_row("CPU", Backend::Cpu, results);
    print_matrix_row("GPU", Backend::Gpu, results);

    println!(
        "{}╠═════════════════════════════════════════════════════════════╣{}",
        MAGENTA, NC
    );

    let total_points: u32 = results.iter().map(|r| r.total_points).sum();
    let max_points: u32 = results.iter().map(|r| r.max_points).sum();
    let passed = results.iter().filter(|r| r.passed()).count();
    let total = results.len();
    let grade = compute_grade(total_points, max_points);

    println!(
        "{}║{} Cells: {}/{} passed    Points: {}/{}    Grade: {:>14}{}║{}",
        MAGENTA, NC, passed, total, total_points, max_points, grade, MAGENTA, NC
    );
    println!(
        "{}╚═════════════════════════════════════════════════════════════╝{}",
        MAGENTA, NC
    );

    if passed == total {
        println!();
        println!("{}Hypothesis \"apr run produces correct output across all formats/backends\" SURVIVED.{}", GREEN, NC);
    } else {
        println!();
        println!(
            "{}Hypothesis FALSIFIED. {} cell(s) failed.{}",
            RED,
            total - passed,
            NC
        );
    }
}

fn print_help() {
    println!("{}QA Matrix Runner (PMAT-QA-PROTOCOL-001){}", BOLD, NC);
    println!();
    println!("{}CRITICAL: Same-Model Comparison Protocol{}", YELLOW, NC);
    println!("  Class A (Quantized): GGUF Q4_K vs APR Q4_K (SAME weights)");
    println!("  Class B (Full Prec): SafeTensors F32 vs APR F32 (SAME weights)");
    println!();
    println!("USAGE:");
    println!("    cargo run --example qa_run -- [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --matrix              Run backend × format matrix (apr run only)");
    println!(
        "    --full-matrix         {}Run FULL 21-cell matrix (modality × format × trace){}",
        CYAN, NC
    );
    println!("    --modality <MODE>     Modality: run (default), chat, serve");
    println!("    --backend <cpu|gpu>   Force specific backend");
    println!("    --format <gguf|safetensors|apr>  Force specific format");
    println!("    --trace               Enable tracing (shows [TRACE-CACHE] messages)");
    println!("    --trace-level <layer|profile>  Detailed trace level");
    println!("    --gguf <PATH>         Path to GGUF model");
    println!("    --safetensors <PATH>  Path to SafeTensors model");
    println!("    --apr <PATH>          Path to APR model");
    println!("    --model <PATH>        Legacy: single model path");
    println!("    --min-cpu-tps <N>     Minimum CPU tok/s (default: 5.0)");
    println!("    --min-gpu-tps <N>     Minimum GPU tok/s for quantized (default: 5.0)");
    println!("    --class <CLASS>       Test class: quantized (default), full-precision, all");
    println!("    --with-ollama         Compare against Ollama as groundtruth");
    println!("    --verbose, -v         Verbose output");
    println!("    --help, -h            Show this help");
    println!();
    println!("MODALITIES (PMAT-QA-PROTOCOL-001 §7.4):");
    println!("    run     `apr run` - single prompt inference");
    println!("    chat    `apr chat` - interactive mode (stdin piped)");
    println!("    serve   `apr serve` - HTTP server (curl tested)");
    println!();
    println!("TEST CLASSES:");
    println!("    quantized      Class A: GGUF Q4_K vs APR Q4_K (faster)");
    println!("    full-precision Class B: SafeTensors F32 vs APR F32 (slower)");
    println!("    all            Both Class A and B");
    println!();
    println!("CANONICAL MODEL:");
    println!("    {}", CANONICAL_GGUF);
    println!();
    println!("EXAMPLES:");
    println!("    # Quick backend × format matrix (apr run only)");
    println!("    cargo run --example qa_run -- --matrix");
    println!();
    println!(
        "    {}# FULL 21-cell matrix (all modalities × formats × trace){}",
        CYAN, NC
    );
    println!("    cargo run --example qa_run -- --full-matrix");
    println!();
    println!("    # Single modality test");
    println!("    cargo run --example qa_run -- --modality chat --backend cpu --format gguf");
    println!();
    println!("    # Compare against Ollama groundtruth");
    println!("    cargo run --example qa_run -- --with-ollama");
}

/// Parsed CLI arguments for the QA run matrix
struct ParsedArgs {
    config: Config,
    run_matrix: bool,
    run_full_matrix: bool,
    single_backend: Option<Backend>,
    single_format: Option<Format>,
    single_modality: Option<Modality>,
    legacy_model: Option<PathBuf>,
    show_help: bool,
}

fn parse_args(args: &[String]) -> ParsedArgs {
    let mut parsed = ParsedArgs {
        config: Config::default(),
        run_matrix: false,
        run_full_matrix: false,
        single_backend: None,
        single_format: None,
        single_modality: None,
        legacy_model: None,
        show_help: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--matrix" => parsed.run_matrix = true,
            "--full-matrix" => parsed.run_full_matrix = true,
            "--with-ollama" => parsed.config.with_ollama = true,
            "--verbose" | "-v" => parsed.config.verbose = true,
            "--help" | "-h" => parsed.show_help = true,
            flag => {
                if let Some(val) = args.get(i + 1) {
                    parse_flag_with_value(flag, val, &mut parsed);
                    i += 1; // extra increment for value
                }
            }
        }
        i += 1;
    }

    parsed
}

fn parse_modality(val: &str) -> Option<Modality> {
    match val {
        "run" => Some(Modality::Run),
        "chat" => Some(Modality::Chat),
        "serve" => Some(Modality::Serve),
        _ => None,
    }
}

fn parse_backend(val: &str) -> Option<Backend> {
    match val {
        "cpu" => Some(Backend::Cpu),
        "gpu" => Some(Backend::Gpu),
        _ => None,
    }
}

fn parse_format(val: &str) -> Option<Format> {
    match val {
        "gguf" => Some(Format::Gguf),
        "safetensors" => Some(Format::SafeTensors),
        "apr" => Some(Format::Apr),
        _ => None,
    }
}

fn parse_trace_level(val: &str) -> TraceLevel {
    match val {
        "brick" => TraceLevel::Brick,
        "step" => TraceLevel::Step,
        "layer" => TraceLevel::Layer,
        "profile" => TraceLevel::Profile,
        _ => TraceLevel::None,
    }
}

fn parse_test_class(val: &str) -> TestClass {
    match val {
        "quantized" | "a" | "A" => TestClass::Quantized,
        "full-precision" | "fp" | "b" | "B" => TestClass::FullPrecision,
        "all" | "both" => TestClass::All,
        _ => TestClass::Quantized,
    }
}

fn parse_flag_with_value(flag: &str, val: &str, parsed: &mut ParsedArgs) {
    match flag {
        "--modality" => parsed.single_modality = parse_modality(val),
        "--backend" => parsed.single_backend = parse_backend(val),
        "--format" => parsed.single_format = parse_format(val),
        "--trace-level" => parsed.config.trace_level = parse_trace_level(val),
        "--class" => parsed.config.test_class = parse_test_class(val),
        "--gguf" => parsed.config.gguf_model = val.to_string(),
        "--safetensors" => parsed.config.safetensors_model = val.to_string(),
        "--apr" => parsed.config.apr_model = val.to_string(),
        "--model" => parsed.legacy_model = Some(PathBuf::from(val)),
        "--min-cpu-tps" => parsed.config.min_cpu_tps = val.parse().unwrap_or(8.0),
        "--min-gpu-tps" => parsed.config.min_gpu_tps = val.parse().unwrap_or(100.0),
        "--min-gpu-tps-f32" => parsed.config.min_gpu_tps_float32 = val.parse().unwrap_or(40.0),
        _ => {}
    }
}

/// Resolve model URI for a given format from config
fn model_for_format(config: &Config, format: Format) -> String {
    match format {
        Format::Gguf => config.gguf_model.clone(),
        Format::SafeTensors => config.safetensors_model.clone(),
        Format::Apr => config.apr_model.clone(),
    }
}

/// Build matrix cells based on parsed CLI arguments (PMAT-SHOWCASE-METHODOLOGY-001)
fn build_cells(config: &Config, parsed: &ParsedArgs) -> Vec<MatrixCell> {
    if parsed.run_full_matrix {
        return build_full_matrix_cells(config);
    }
    if parsed.run_matrix {
        return build_standard_matrix_cells(config);
    }
    if let (Some(modality), Some(backend), Some(format)) = (
        parsed.single_modality,
        parsed.single_backend,
        parsed.single_format,
    ) {
        let model = model_for_format(config, format);
        return vec![MatrixCell::new("S1", backend, format, model).with_modality(modality)];
    }
    if let (Some(backend), Some(format)) = (parsed.single_backend, parsed.single_format) {
        let model = model_for_format(config, format);
        return vec![MatrixCell::new("S1", backend, format, model)];
    }
    if let Some(ref model_path) = parsed.legacy_model {
        return build_legacy_cells(model_path);
    }
    println!(
        "{}No mode specified. Use --matrix, --backend + --format, or --model{}",
        YELLOW, NC
    );
    println!();
    print_help();
    std::process::exit(2);
}

/// Build full 21-cell matrix (3 modalities × 3 formats × trace variants)
fn build_full_matrix_cells(config: &Config) -> Vec<MatrixCell> {
    let mut cells = Vec::new();
    let mut id = 1;
    for modality in [Modality::Run, Modality::Chat, Modality::Serve] {
        for format in [Format::Gguf, Format::SafeTensors, Format::Apr] {
            let model = model_for_format(config, format);
            cells.push(
                MatrixCell::new(&format!("F{id:02}"), Backend::Cpu, format, model.clone())
                    .with_modality(modality),
            );
            id += 1;
            cells.push(
                MatrixCell::new(&format!("F{id:02}"), Backend::Cpu, format, model.clone())
                    .with_modality(modality)
                    .with_trace(true),
            );
            id += 1;
            if format == Format::Gguf {
                cells.push(
                    MatrixCell::new(&format!("F{id:02}"), Backend::Gpu, format, model)
                        .with_modality(modality),
                );
                id += 1;
            }
        }
    }
    println!(
        "{}FULL MATRIX: {} cells (modality × format × trace){}\n",
        MAGENTA,
        cells.len(),
        NC
    );
    cells
}

/// Build standard matrix cells (Class A quantized + Class B full precision)
fn build_standard_matrix_cells(config: &Config) -> Vec<MatrixCell> {
    let mut cells = Vec::new();
    if config.test_class.includes_quantized() {
        cells.push(MatrixCell::new(
            "A1",
            Backend::Cpu,
            Format::Gguf,
            config.gguf_model.clone(),
        ));
        cells.push(MatrixCell::new(
            "A2",
            Backend::Cpu,
            Format::Apr,
            config.apr_model.clone(),
        ));
        cells.push(MatrixCell::new(
            "A3",
            Backend::Gpu,
            Format::Gguf,
            config.gguf_model.clone(),
        ));
        cells.push(MatrixCell::new(
            "A4",
            Backend::Gpu,
            Format::Apr,
            config.apr_model.clone(),
        ));
    }
    if config.test_class.includes_full_precision() {
        cells.push(MatrixCell::new(
            "B1",
            Backend::Cpu,
            Format::SafeTensors,
            config.safetensors_model.clone(),
        ));
        cells.push(MatrixCell::new(
            "B2",
            Backend::Cpu,
            Format::Apr,
            config.apr_model.clone(),
        ));
        cells.push(MatrixCell::new(
            "B3",
            Backend::Gpu,
            Format::SafeTensors,
            config.safetensors_model.clone(),
        ));
        cells.push(MatrixCell::new(
            "B4",
            Backend::Gpu,
            Format::Apr,
            config.apr_model.clone(),
        ));
    }
    cells
}

/// Build cells for legacy --model flag
fn build_legacy_cells(model_path: &PathBuf) -> Vec<MatrixCell> {
    let model = model_path.to_string_lossy().to_string();
    let format = if model_path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        Format::Gguf
    } else if model_path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
    {
        Format::SafeTensors
    } else {
        Format::Apr
    };
    vec![
        MatrixCell::new("L1", Backend::Cpu, format, model.clone()),
        MatrixCell::new("L2", Backend::Gpu, format, model),
    ]
}

