//! Unit proofs for `apr setfit bench run`'s adapter mechanics.
//!
//! # What is proven here and what deliberately is not
//!
//! The PARSERS, the median, the cell-key refusals, the filename grammar and the transport
//! ingest are proven here, because each is a pure function over bytes a test can author.
//!
//! The full SetFit execution path is NOT faked here. Its cost is a real training run against a
//! pinned 86.7 MB encoder checkout, and a stub row labelled "real" would be worse than no test:
//! it would put a green tick beside a claim nothing measured. Plan 05-12 executes it for real.

use super::resource::{
    median, parse_child_max_rss, parse_cold_latency_ms, parse_vm_hwm_bytes, MECHANISM_CHILD_TIME_L,
    MECHANISM_CHILD_VM_HWM, MECHANISM_SAMPLED_PREFIX, MECHANISM_VM_HWM,
};
use super::*;

/// This module's parent source, for the source assertions.
const SETFIT_BENCH_SOURCE: &str = include_str!("setfit_bench.rs");

/// Assemble a search needle from fragments at RUNTIME.
///
/// The scans below read the file this module is attached to, so a whole literal written here
/// would NOT appear in it — but a literal written in `setfit_bench.rs`'s own doc comments
/// would, which is the self-match hazard `setfit_train.rs` records. Fragmenting keeps the
/// discipline uniform across both files.
fn needle(fragments: &[&str]) -> String {
    fragments.concat()
}

// ==========================================================================================
// The `/usr/bin/time` max-RSS parser — the two-platform case table (CLAUDE.md rule 7)
// ==========================================================================================

/// A canned macOS `/usr/bin/time -l` block, values in BYTES.
///
/// Transcribed from a real run's shape, including the neighbouring lines that are one word
/// away from matching. `average shared memory size` is the trap: it ends in `size`, sits in
/// the same column layout, and a looser suffix match would take it.
const MACOS_TIME_L: &str = "        0.31 real         0.19 user         0.06 sys
             1441792  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                 366  page reclaims
                   0  page faults
             1441792  peak memory footprint
";

/// A canned GNU `/usr/bin/time -v` block, values in KILOBYTES.
///
/// `Average resident set size (kbytes): 0` is this side's trap: it differs from the wanted
/// line by ONE WORD and carries the same unit suffix.
const LINUX_TIME_V: &str = "\tCommand being timed: \"apr setfit bench run --cold-probe m.apr\"
\tUser time (seconds): 0.19
\tMaximum resident set size (kbytes): 3456
\tAverage resident set size (kbytes): 0
\tAverage total size (kbytes): 0
\tExit status: 0
";

#[test]
fn setfit_bench_max_rss_parser_case_table_must_match() {
    // macOS reports BYTES. The parsed value is the numeral ITSELF.
    let (bytes, mechanism) =
        parse_child_max_rss(MACOS_TIME_L).expect("the macOS block carries a maximum RSS line");
    assert_eq!(
        bytes, 1_441_792,
        "macOS `/usr/bin/time -l` reports BYTES, so the numeral passes through unscaled"
    );
    assert_eq!(
        mechanism, MECHANISM_CHILD_TIME_L,
        "the mechanism is DERIVED FROM THE FORM THAT PARSED, not from cfg!(target_os) — the \
         two hosts are deliberately different (D-09), so a parent that assumed its own \
         platform would mislabel every transported number"
    );

    // Linux reports KILOBYTES. The parsed value is the numeral TIMES 1024.
    let (bytes, mechanism) =
        parse_child_max_rss(LINUX_TIME_V).expect("the GNU block carries a maximum RSS line");
    assert_eq!(
        bytes,
        3456 * 1024,
        "GNU `/usr/bin/time -v` reports KILOBYTES, so the numeral is scaled by 1024"
    );
    assert_eq!(
        bytes, 3_538_944,
        "stated as a literal so the scaling is visible"
    );
    assert_eq!(mechanism, MECHANISM_CHILD_VM_HWM);
}

#[test]
fn setfit_bench_max_rss_parser_pins_the_unit_conversion_in_both_directions() {
    // THE SAME NUMERAL in the two blocks means two DIFFERENT quantities. This is the whole
    // hazard the case table exists for: a parser that ignored the unit would report the two as
    // equal, and every cross-host memory comparison in the phase would be wrong by 1024x.
    let macos = "             4096  maximum resident set size\n";
    let linux = "\tMaximum resident set size (kbytes): 4096\n";

    let (macos_bytes, _) = parse_child_max_rss(macos).expect("macOS form parses");
    let (linux_bytes, _) = parse_child_max_rss(linux).expect("GNU form parses");

    assert_eq!(macos_bytes, 4096, "macOS: 4096 bytes");
    assert_eq!(
        linux_bytes, 4_194_304,
        "GNU: 4096 kilobytes = 4194304 bytes"
    );
    assert_eq!(
        linux_bytes,
        macos_bytes * 1024,
        "and the ratio is exactly 1024, in that direction — a parser that had the conversion \
         backwards would satisfy an equality test written only one way round"
    );
}

#[test]
fn setfit_bench_max_rss_parser_case_table_must_not_match() {
    // Each of these is a line the parser MUST refuse. They are the neighbours of the wanted
    // lines, which is where a loose pattern actually goes wrong — not on obviously unrelated
    // text.
    let must_not_match: &[(&str, &str)] = &[
        ("", "empty output"),
        (
            "                   0  average shared memory size\n",
            "macOS's neighbouring line, same column layout",
        ),
        (
            "             1441792  peak memory footprint\n",
            "macOS's peak footprint is a DIFFERENT metric",
        ),
        (
            "\tAverage resident set size (kbytes): 0\n",
            "GNU's average, one word from the maximum",
        ),
        (
            "\tMaximum resident set size (bytes): 3456\n",
            "a unit this parser has never seen must be refused, not silently read as kB",
        ),
        (
            "             notanumber  maximum resident set size\n",
            "a non-numeric macOS value",
        ),
        (
            "maximum resident set size\n",
            "the macOS label with no value at all",
        ),
        ("\tExit status: 0\n", "a line that mentions neither metric"),
    ];
    for (input, why) in must_not_match {
        assert!(
            parse_child_max_rss(input).is_none(),
            "must NOT match ({why}); input: {input:?}"
        );
    }
}

// ==========================================================================================
// The VmHWM parser
// ==========================================================================================

/// A canned `/proc/self/status` excerpt. `VmHWM` sits between two lines that share its prefix
/// shape, which is why the parser anchors rather than searching.
const PROC_STATUS: &str = "Name:\tapr
VmPeak:\t  912344 kB
VmSize:\t  845112 kB
VmHWM:\t  123456 kB
VmRSS:\t   99000 kB
Threads:\t8
";

#[test]
fn setfit_bench_vm_hwm_parser_case_table() {
    assert_eq!(
        parse_vm_hwm_bytes(PROC_STATUS),
        Some(123_456 * 1024),
        "VmHWM is reported in kB and the row records bytes"
    );

    let must_not_match: &[(&str, &str)] = &[
        ("", "empty"),
        (
            "VmRSS:\t 99000 kB\n",
            "the current RSS is not the high-water mark",
        ),
        (
            "VmHWMX:\t 1 kB\n",
            "a longer field name that shares the prefix",
        ),
        (
            "the VmHWM: field is documented in proc(5)\n",
            "prose mentioning the field — `contains` would take this",
        ),
        (
            "VmHWM:\t 123456 MB\n",
            "a unit this parser has never seen must be refused, not misread by 1024x",
        ),
        ("VmHWM:\t notanumber kB\n", "a non-numeric value"),
        ("VmHWM:\n", "the field with no value"),
    ];
    for (input, why) in must_not_match {
        assert!(
            parse_vm_hwm_bytes(input).is_none(),
            "must NOT match ({why}); input: {input:?}"
        );
    }
}

// ==========================================================================================
// The median and the cold-latency line
// ==========================================================================================

#[test]
fn setfit_bench_median_handles_even_and_odd() {
    assert_eq!(
        median(vec![3.0, 1.0, 2.0]),
        Some(2.0),
        "odd: the middle element after sorting, NOT the middle of the input order"
    );
    assert_eq!(
        median(vec![4.0, 1.0, 3.0, 2.0]),
        Some(2.5),
        "even: the mean of the two middles"
    );
    assert_eq!(
        median(vec![7.5]),
        Some(7.5),
        "a single sample is its own median"
    );
    assert_eq!(
        median(Vec::new()),
        None,
        "an empty sample has no median; returning 0.0 would publish a latency of zero"
    );
}

#[test]
fn setfit_bench_cold_latency_line_round_trips() {
    let rendered = format!("{COLD_LATENCY_PREFIX}12.5");
    assert_eq!(
        parse_cold_latency_ms(&rendered),
        Some(12.5),
        "the child writes this line and the parent parses it — one constant, two processes"
    );
    assert_eq!(
        parse_cold_latency_ms("noise\nCOLD_LATENCY_MS=0.25\nmore noise\n"),
        Some(0.25),
        "the line is found among other output"
    );
    assert_eq!(
        parse_cold_latency_ms("COLD_LATENCY=1.0\n"),
        None,
        "a near-miss prefix is not the contract"
    );
    assert_eq!(parse_cold_latency_ms(""), None, "no line, no measurement");
}

// ==========================================================================================
// The four mechanism strings
// ==========================================================================================

#[test]
fn setfit_bench_names_all_four_mechanism_strings() {
    // Asserted against LITERALS rather than against the constants they are bound to: a test
    // quoting the constants would pass no matter what they became, which is the shape 05-06
    // records for `resolve_training_config`.
    assert_eq!(MECHANISM_VM_HWM, "vm_hwm");
    assert_eq!(MECHANISM_SAMPLED_PREFIX, "sysinfo_sampled_");
    assert_eq!(MECHANISM_CHILD_VM_HWM, "child_max_rss_vm_hwm");
    assert_eq!(MECHANISM_CHILD_TIME_L, "child_max_rss_time_l");

    // And the two exact ones are the LIBRARY's, not a second spelling of them. The contract
    // declares them mutually comparable; two copies could drift and the comparison would
    // silently stop meaning anything.
    assert_eq!(
        MECHANISM_CHILD_VM_HWM,
        entrenar::train::setfit::bench_row::MECHANISM_CHILD_MAX_RSS_VM_HWM
    );
    assert_eq!(
        MECHANISM_CHILD_TIME_L,
        entrenar::train::setfit::bench_row::MECHANISM_CHILD_MAX_RSS_TIME_L
    );
}

#[test]
fn setfit_bench_sampled_mechanism_always_carries_a_nonzero_interval() {
    // A sampled mechanism whose interval is absent or zero is unreadable: the reader cannot
    // tell how much of the peak the sampler could possibly have seen. The sampler's own
    // `finish` clamps the achieved rate to at least 1, and this pins that the STRING and the
    // FIELD agree.
    let peak = super::resource::TrainRssSampler::start().finish();
    match peak.sample_interval_hz {
        Some(hz) => {
            assert!(
                hz >= 1,
                "a sampled interval of zero is not a measurement boundary"
            );
            assert_eq!(
                peak.mechanism,
                format!("{MECHANISM_SAMPLED_PREFIX}{hz}"),
                "the interval in the mechanism string must be the interval in the field"
            );
        }
        None => assert_eq!(
            peak.mechanism, MECHANISM_VM_HWM,
            "the only mechanism permitted to omit an interval is the exact kernel one"
        ),
    }
}

// ==========================================================================================
// Cell identity: a non-contracted cell is a TYPED REFUSAL, never a run
// ==========================================================================================

#[test]
fn setfit_bench_refuses_a_non_contracted_seed_naming_the_contract() {
    let error = resolve_cell(Some("setfit"), Some(8), Some(42))
        .expect_err("42 is deliberately NOT a contracted seed");
    let rendered = error.to_string();
    assert!(
        rendered.contains(CLAIMS_CONTRACT_ID),
        "the refusal must name the contract that decides this; got: {rendered}"
    );
    assert!(
        rendered.contains("42"),
        "and the offending value; got: {rendered}"
    );
}

#[test]
fn setfit_bench_refuses_an_empty_or_degenerate_shot_count() {
    // 0 is the DEGENERATE cell (a "few-shot" run with no examples), 12 is a plausible-looking
    // count that is simply not contracted, and 128 is outside the top of the range. All three
    // are refusals rather than runs: a cell outside the matrix produces a row the report has
    // no slot for, which is exactly the omission the expectation set exists to make visible.
    for bad in [0_u32, 12, 128] {
        let error = resolve_cell(Some("setfit"), Some(bad), Some(13))
            .expect_err("a non-contracted shot count must be refused");
        let rendered = error.to_string();
        assert!(
            rendered.contains(CLAIMS_CONTRACT_ID),
            "the refusal must name the contract; s{bad} gave: {rendered}"
        );
        assert!(
            rendered.contains(&bad.to_string()),
            "and the offending value; s{bad} gave: {rendered}"
        );
    }
}

#[test]
fn setfit_bench_refuses_a_method_outside_the_two_compared() {
    let error = resolve_cell(Some("qlora"), Some(8), Some(13))
        .expect_err("a third method is not part of the comparison");
    assert!(error.to_string().contains("setfit"), "got: {error}");
    assert!(error.to_string().contains("lora"), "got: {error}");
}

#[test]
fn setfit_bench_accepts_every_contracted_cell() {
    // The positive half. Without it the refusals above would be satisfied by a function that
    // refuses everything.
    let mut accepted = 0_usize;
    for method in ["setfit", "lora"] {
        for shots in BENCH_SHOTS {
            for seed in BENCH_SEEDS {
                let cell = resolve_cell(Some(method), Some(shots), Some(seed))
                    .expect("every contracted cell resolves");
                assert!(cell.is_contracted());
                accepted += 1;
            }
        }
    }
    // THE ROW-VALIDITY DOMAIN, NOT THE EXPECTATION SET — the two questions this plan split
    // apart. `resolve_cell` answers "is this a representable cell", which still admits both
    // methods; EXPECTED_CELLS answers "what must a complete run contain", which since the 2.0.0
    // narrowing is the SetFit half alone. Asserting one against the other is exactly the
    // collapse ACTIVE_METHODS exists to prevent, and it is what this assertion used to do.
    assert_eq!(
        accepted,
        entrenar::train::setfit::bench_row::BENCH_METHODS.len()
            * BENCH_SHOTS.len()
            * BENCH_SEEDS.len(),
        "every REPRESENTABLE cell resolves — both methods, because BENCH_METHODS is the \
         row-validity domain and keeps both"
    );
    assert_eq!(
        accepted,
        2 * entrenar::train::setfit::bench_row::EXPECTED_CELLS,
        "and the ACTIVE expectation set is exactly half of it: a second method's cell is VALID \
         as a cell while being OUT OF SCOPE for a complete run"
    );
}

// ==========================================================================================
// The filename grammar
// ==========================================================================================

#[test]
fn setfit_bench_row_file_name_grammar() {
    assert_eq!(
        row_file_name(CellKey::new(Method::Setfit, 8, 13)),
        "setfit-s8-seed13.json"
    );
    assert_eq!(
        row_file_name(CellKey::new(Method::Lora, 64, 53)),
        "lora-s64-seed53.json"
    );
    assert_eq!(
        lock_relative_path(CellKey::new(Method::Setfit, 16, 29)),
        "locks/setfit-s16-seed29.lock.json"
    );
    assert_eq!(
        ledger_relative_path(CellKey::new(Method::Lora, 32, 41)),
        "ledger/lora-s32-seed41.jsonl"
    );

    // ONE function produces the name, so a resume path comparing a recorded digest against
    // "the file this cell would have written" cannot be looking at a different file.
    let mut names = std::collections::BTreeSet::new();
    for cell in RunManifest::expectation() {
        assert!(
            names.insert(row_file_name(cell)),
            "the grammar must be injective over the expectation set: {cell} collided"
        );
    }
    assert_eq!(
        names.len(),
        entrenar::train::setfit::bench_row::EXPECTED_CELLS
    );
}

// ==========================================================================================
// The clap surface
// ==========================================================================================

/// `apr`'s clap tree is 103 subcommands deep in places, and BUILDING it recurses far enough
/// to blow libtest's 2 MiB default thread stack. Measured, not guessed: the first version of
/// these tests aborted with `has overflowed its stack` inside clap's own construction, before
/// any assertion ran. So every clap-tree test runs on a thread with room.
fn on_roomy_stack<T, F>(body: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .expect("the roomy-stack thread spawns")
        .join()
        .expect("the roomy-stack thread does not panic")
}

/// Parse an `apr setfit bench run …` argv through the REAL top-level parser.
///
/// Through `Cli`, not through a hand-built `Command`: the conflicts this asserts are declared
/// on the subcommand and only take effect once it is reachable from the binary's own tree, so
/// a test that constructed the subcommand in isolation would prove the attributes exist and
/// not that they apply.
///
/// Returns the `ErrorKind` rather than the `Error`, so the outcome crosses the thread boundary
/// as a `Copy` discriminant — and so an assertion names a KIND rather than matching rendered
/// prose.
fn parse_bench_run(extra: &[&str]) -> std::result::Result<(), clap::error::ErrorKind> {
    let owned: Vec<String> = std::iter::once("apr".to_string())
        .chain(["setfit", "bench", "run"].iter().map(|s| (*s).to_string()))
        .chain(extra.iter().map(|s| (*s).to_string()))
        .collect();
    on_roomy_stack(move || {
        use clap::Parser as _;
        crate::Cli::try_parse_from(owned)
            .map(|_| ())
            .map_err(|error| error.kind())
    })
}

#[test]
fn setfit_bench_run_accepts_the_execution_flag_set() {
    parse_bench_run(&[
        "--method",
        "setfit",
        "--shots",
        "8",
        "--seed",
        "13",
        "--data",
        "d",
        "--selection",
        "s.json",
        "--bench-dir",
        "b",
        "--model-dir",
        "m",
        "--config",
        "c.toml",
        "--force",
    ])
    .expect("the execution flag set parses");
}

#[test]
fn setfit_bench_run_record_conflicts_with_every_execution_flag() {
    // Each pairing is asserted SEPARATELY. One combined argv would go red on the first
    // conflict and prove nothing about the rest — the failure mode where four of five
    // `conflicts_with` entries are missing and the test still passes.
    for (flag, value) in [
        ("--method", Some("setfit")),
        ("--shots", Some("8")),
        ("--seed", Some("13")),
        ("--data", Some("d")),
        ("--selection", Some("s.json")),
        ("--model-dir", Some("m")),
        ("--config", Some("c.toml")),
    ] {
        let mut argv = vec![
            "--record",
            "rows/setfit-s8-seed13.json",
            "--bench-dir",
            "b",
            flag,
        ];
        if let Some(value) = value {
            argv.push(value);
        }
        let kind = parse_bench_run(&argv).expect_err(
            "--record must conflict with the execution flags: a record mode that also accepted \
             them would look like it had RUN the cell it merely ingested",
        );
        assert_eq!(
            kind,
            clap::error::ErrorKind::ArgumentConflict,
            "{flag} must be a CONFLICT, not some other parse failure; got: {kind:?}"
        );
    }
}

#[test]
fn setfit_bench_run_cold_probe_conflicts_with_the_bench_dir_and_the_execution_flags() {
    for (flag, value) in [
        ("--bench-dir", Some("b")),
        ("--method", Some("setfit")),
        ("--shots", Some("8")),
        ("--record", Some("r.json")),
    ] {
        let mut argv = vec!["--cold-probe", "m.apr", "--probe-text", "t.txt", flag];
        if let Some(value) = value {
            argv.push(value);
        }
        let kind =
            parse_bench_run(&argv).expect_err("--cold-probe must conflict with the other modes");
        assert_eq!(
            kind,
            clap::error::ErrorKind::ArgumentConflict,
            "{flag}; got: {kind:?}"
        );
    }
}

#[test]
fn setfit_bench_run_probe_flags_require_the_cold_probe() {
    // `--probe-text` without `--cold-probe` is a request to measure nothing. `requires` makes
    // that a parse error rather than a silently ignored flag, which is the distinction
    // `eval/setfit.rs`'s `check_split_flags` states: a flag silently ignored is worse than a
    // flag refused.
    let kind = parse_bench_run(&["--probe-text", "t.txt"])
        .expect_err("--probe-text alone must be refused");
    assert_eq!(kind, clap::error::ErrorKind::MissingRequiredArgument);

    let kind = parse_bench_run(&["--cold-probe-base", "base.apr"])
        .expect_err("--cold-probe-base alone must be refused");
    assert_eq!(kind, clap::error::ErrorKind::MissingRequiredArgument);

    // And the positive half, so the two refusals above are not satisfied by a parser that
    // refuses everything.
    parse_bench_run(&["--cold-probe", "m.apr", "--probe-text", "t.txt"])
        .expect("the probe's own flag set parses");
}

#[test]
fn setfit_bench_run_long_help_lists_the_machinery_flags() {
    let (long_help, short_help) = on_roomy_stack(|| {
        use clap::CommandFactory as _;

        let mut root = crate::Cli::command();
        let run = root
            .find_subcommand_mut("setfit")
            .expect("`apr setfit` is reachable")
            .find_subcommand_mut("bench")
            .expect("`apr setfit bench` is reachable")
            .find_subcommand_mut("run")
            .expect("`apr setfit bench run` is reachable");
        (
            run.render_long_help().to_string(),
            run.render_help().to_string(),
        )
    });

    for flag in [
        "--method",
        "--shots",
        "--seed",
        "--data",
        "--selection",
        "--bench-dir",
        "--model-dir",
        "--config",
        "--force",
        "--record",
        // `--cold-probe` is `hide_short_help`, not `hide`: it is machinery rather than a user
        // surface, so it stays out of `-h` — but a reader auditing the resource protocol must
        // be able to find it without reading the source, so it IS in `--help`.
        "--cold-probe",
        "--probe-text",
    ] {
        assert!(
            long_help.contains(flag),
            "`apr setfit bench run --help` must list {flag}; got:\n{long_help}"
        );
    }

    assert!(
        !short_help.contains("--cold-probe"),
        "and -h must NOT: the cold probe is spawned by `bench run` itself, and offering it as \
         a user surface would invite someone to type it and read its number as a cell's"
    );
    assert!(
        short_help.contains("--method") && short_help.contains("--record"),
        "while the two modes an operator does drive stay in the short help"
    );
}

// ==========================================================================================
// A synthetic row, for the transport-ingest proofs
// ==========================================================================================

mod synthetic {
    use entrenar::train::setfit::bench_row::{
        sha256_hex, BenchLockRef, BenchRow, BenchRowPayload, CellKey, HostIdentity, MethodEvidence,
        QualityBlock, ResourceBlock, SetfitEvidence, BENCH_ROW_SCHEMA_VERSION, CALIBRATION_SPLIT,
        CLAIMS_CONTRACT_ID, MECHANISM_CHILD_MAX_RSS_TIME_L, WARMUP_COUNT,
    };

    /// The canonical three labels, in the pinned dataset's own order.
    pub(super) fn labels() -> Vec<String> {
        ["none", "against", "favor"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }

    /// The bytes a committed lock record would have. Arbitrary, and that is the point: the
    /// property under test is that the ROW's `lock_hash` is the SHA-256 of whatever is on
    /// disk, so the content must not be something the row could re-derive.
    pub(super) const LOCK_BYTES: &[u8] = br#"{"chosen_artifact_hash":"ab","rule":"max_metric"}"#;

    /// A structurally valid row for `cell`, whose lock hash is the digest of [`LOCK_BYTES`].
    pub(super) fn payload(cell: CellKey, artifact_bytes: u64) -> BenchRowPayload {
        let f_avg = 0.625_f64;
        let macro_f1 = 0.5_f64;
        let mcc = 0.25_f64;
        let ece = 0.1_f64;
        let brier = 0.4_f64;
        BenchRowPayload {
            schema_version: BENCH_ROW_SCHEMA_VERSION,
            contract_id: CLAIMS_CONTRACT_ID.to_string(),
            method: cell.method,
            shots: cell.shots,
            seed: cell.seed,
            dataset_revision: "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66".to_string(),
            dataset_fingerprint: "aa".repeat(32),
            model_revision: "bb".repeat(20),
            selection_manifest_hash: "cc".repeat(32),
            backend_identity: "cpu:setfit-core:autograd-trueno-matmul".to_string(),
            host: HostIdentity {
                hostname: "probe".to_string(),
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
            },
            quality: QualityBlock {
                f_avg,
                f_avg_bits: f_avg.to_bits(),
                macro_f1,
                macro_f1_bits: macro_f1.to_bits(),
                per_class_precision: vec![0.5, 0.5, 0.5],
                per_class_recall: vec![0.5, 0.5, 0.5],
                per_class_f1: vec![0.25, 0.5, 0.75],
                mcc,
                mcc_bits: mcc.to_bits(),
                confusion_matrix: vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]],
                n_test_rows: 3,
                ordered_labels: labels(),
                ece_top_label_validation: ece,
                ece_top_label_validation_bits: ece.to_bits(),
                brier_multiclass_validation: brier,
                brier_multiclass_validation_bits: brier.to_bits(),
                calibration_split: CALIBRATION_SPLIT.to_string(),
            },
            resource: ResourceBlock {
                train_wall_ms: 1234,
                cold_latency_ms: 42.0,
                cold_measured_in_child_process: true,
                warm_latency_ms_median: 4.0,
                throughput_rows_per_sec: 100.0,
                throughput_batch_size: 32,
                warmup_count: WARMUP_COUNT,
                train_peak_rss_bytes: 1_000_000,
                train_peak_rss_mechanism: "vm_hwm".to_string(),
                inference_peak_rss_bytes: 500_000,
                inference_peak_rss_mechanism: MECHANISM_CHILD_MAX_RSS_TIME_L.to_string(),
                peak_rss_sample_interval_hz: None,
                artifact_bytes,
                // A SetFit row: one standalone file, so the deployable size IS the artifact
                // size. The LoRA half of this equality is asserted separately, because there
                // the two must DIFFER.
                deployable_total_bytes: artifact_bytes,
            },
            evidence: MethodEvidence::Setfit(SetfitEvidence {
                evidence_table_hash: "dd".repeat(32),
                apr_artifact_sha256: "ee".repeat(32),
                lock: BenchLockRef {
                    lock_hash: sha256_hex(LOCK_BYTES),
                    role: "written".to_string(),
                    rule: "max_metric_lowest_index_tie_break".to_string(),
                    lock_record_path: super::lock_relative_path(cell),
                },
            }),
        }
    }

    /// The row file's bytes for `cell`.
    pub(super) fn row_bytes(cell: CellKey, artifact_bytes: u64) -> Vec<u8> {
        BenchRow::new(payload(cell, artifact_bytes))
            .to_file_bytes()
            .expect("the synthetic row serializes")
    }
}

// ==========================================================================================
// `--record`: the transport ingest
// ==========================================================================================

/// Write `bytes` into a temp directory under the name `cell` demands, and return both paths.
fn staged_row(
    temp: &tempfile::TempDir,
    cell: CellKey,
    bytes: &[u8],
) -> (std::path::PathBuf, std::path::PathBuf) {
    let incoming = temp.path().join("incoming");
    std::fs::create_dir_all(&incoming).expect("the incoming directory is creatable");
    let row_file = incoming.join(row_file_name(cell));
    std::fs::write(&row_file, bytes).expect("the staged row is writable");
    (temp.path().join("bench"), row_file)
}

#[test]
fn setfit_bench_record_ingests_a_valid_transported_row() {
    let temp = tempfile::TempDir::new().expect("tempdir");
    let cell = CellKey::new(Method::Setfit, 16, 29);
    let bytes = synthetic::row_bytes(cell, 793_416);
    let (bench_dir, row_file) = staged_row(&temp, cell, &bytes);

    super::record_mode(&bench_dir, &row_file, false, true).expect("a valid row records");

    let landed = bench_dir.join(ROWS_DIR).join(row_file_name(cell));
    assert_eq!(
        std::fs::read(&landed).expect("the row landed"),
        bytes,
        "the recorded row must be the transported BYTES, not a re-serialization of them — a \
         re-encode would produce a file whose digest the sender never computed"
    );

    // And the manifest now carries it. Completeness is defined by THAT file, so a row on disk
    // that the manifest never saw would be invisible to the 05-10 gate.
    let manifest = load_or_declare_manifest(&bench_dir).expect("the manifest reloads");
    assert_eq!(manifest.completed(), 1);
    assert!(manifest.row_sha256(cell).is_some());
}

#[test]
fn setfit_bench_record_is_idempotent_on_an_identical_digest() {
    // The resume-after-a-dropped-ssh case. Refusing this would get manifests deleted and
    // re-created, which erases the pre-declared expectation set that makes omission visible.
    let temp = tempfile::TempDir::new().expect("tempdir");
    let cell = CellKey::new(Method::Setfit, 8, 13);
    let bytes = synthetic::row_bytes(cell, 1_024);
    let (bench_dir, row_file) = staged_row(&temp, cell, &bytes);

    super::record_mode(&bench_dir, &row_file, false, true).expect("the first record lands");
    super::record_mode(&bench_dir, &row_file, false, true)
        .expect("re-recording the IDENTICAL digest is idempotent, and needs no --force");

    let manifest = load_or_declare_manifest(&bench_dir).expect("the manifest reloads");
    assert_eq!(
        manifest.completed(),
        1,
        "an idempotent re-record must not duplicate the cell"
    );
}

#[test]
fn setfit_bench_record_refuses_a_differing_digest_for_a_recorded_cell() {
    let temp = tempfile::TempDir::new().expect("tempdir");
    let cell = CellKey::new(Method::Setfit, 8, 13);
    let (bench_dir, row_file) = staged_row(&temp, cell, &synthetic::row_bytes(cell, 1_024));
    super::record_mode(&bench_dir, &row_file, false, true).expect("the first record lands");

    // A row for the SAME cell with a DIFFERENT measurement — the collision.
    let differing = synthetic::row_bytes(cell, 2_048);
    std::fs::write(&row_file, &differing).expect("the second staged row is writable");
    let error = super::record_mode(&bench_dir, &row_file, false, true)
        .expect_err("a differing re-record must be refused, not silently overwritten");
    let rendered = error.to_string();
    assert!(
        rendered.contains("already recorded"),
        "the refusal must say the cell is already recorded; got: {rendered}"
    );

    // And the refusal did not partially apply: the ORIGINAL row is still on disk.
    let landed = bench_dir.join(ROWS_DIR).join(row_file_name(cell));
    assert_eq!(
        std::fs::read(&landed).expect("the original row survives"),
        synthetic::row_bytes(cell, 1_024),
        "the run that produced the published number must stay the run on disk"
    );
}

#[test]
fn setfit_bench_record_refuses_a_doctored_digest() {
    let temp = tempfile::TempDir::new().expect("tempdir");
    let cell = CellKey::new(Method::Setfit, 32, 41);
    let mut bytes = synthetic::row_bytes(cell, 4_096);
    // Flip one byte INSIDE the payload, leaving the envelope's digest untouched. This is the
    // shape a transport tamper actually has: the file still parses.
    let text = String::from_utf8(bytes.clone()).expect("the row is UTF-8");
    let doctored = text.replace("\"train_wall_ms\": 1234", "\"train_wall_ms\": 9999");
    assert_ne!(
        doctored, text,
        "the fixture must actually have been altered"
    );
    bytes = doctored.into_bytes();
    let (bench_dir, row_file) = staged_row(&temp, cell, &bytes);

    let error = super::record_mode(&bench_dir, &row_file, false, true)
        .expect_err("a payload whose digest no longer matches must be refused");
    assert!(
        error.to_string().contains("digest mismatch"),
        "got: {error}"
    );
    assert!(
        !bench_dir.join(ROWS_DIR).join(row_file_name(cell)).exists(),
        "and nothing was written"
    );
}

#[test]
fn setfit_bench_record_refuses_a_filename_that_disagrees_with_the_payload() {
    // The one check the library CANNOT make: it verifies the payload, not the name the
    // operator filed it under. A row named for another cell would be counted as that cell by
    // every later reader.
    let temp = tempfile::TempDir::new().expect("tempdir");
    let cell = CellKey::new(Method::Setfit, 8, 13);
    let bytes = synthetic::row_bytes(cell, 4_096);

    let incoming = temp.path().join("incoming");
    std::fs::create_dir_all(&incoming).expect("creatable");
    let wrong_name = incoming.join(row_file_name(CellKey::new(Method::Setfit, 8, 17)));
    std::fs::write(&wrong_name, &bytes).expect("writable");

    let error = super::record_mode(&temp.path().join("bench"), &wrong_name, false, true)
        .expect_err("a name/content disagreement must be refused");
    let rendered = error.to_string();
    assert!(
        rendered.contains("setfit-s8-seed13.json"),
        "got: {rendered}"
    );
    assert!(
        rendered.contains("setfit-s8-seed17.json"),
        "got: {rendered}"
    );
}

#[test]
fn setfit_bench_record_refusals_are_distinct_typed_errors() {
    // Three refusals that must not collapse into one. A caller who cannot tell a tampered row
    // from a misfiled one cannot act on either.
    let temp = tempfile::TempDir::new().expect("tempdir");
    let cell = CellKey::new(Method::Setfit, 8, 13);

    // (a) An UNCONTRACTED cell — refused by the library before the filename is even consulted.
    let uncontracted = CellKey::new(Method::Setfit, 8, 42);
    let mut payload = synthetic::payload(cell, 1);
    payload.seed = 42;
    let row = entrenar::train::setfit::bench_row::BenchRow::new(payload);
    let bytes = row.to_file_bytes().expect("serializes");
    let (bench_dir, _) = staged_row(&temp, cell, b"placeholder");
    let incoming = temp.path().join("incoming");
    let path = incoming.join(row_file_name(uncontracted));
    std::fs::write(&path, &bytes).expect("writable");
    let uncontracted_error = super::record_mode(&bench_dir, &path, false, true)
        .expect_err("seed 42 is outside the contracted matrix")
        .to_string();
    assert!(
        uncontracted_error.contains("outside the contracted matrix"),
        "got: {uncontracted_error}"
    );

    // (b) A DOCTORED digest and (c) a MISFILED name are covered by their own tests above; what
    //     this asserts is that (a) reads differently from both.
    assert!(!uncontracted_error.contains("digest mismatch"));
    assert!(!uncontracted_error.contains("the file handed over is named"));
}

// ==========================================================================================
// The lock-hash relationship the 05-10 gate recomputes
// ==========================================================================================

#[test]
fn setfit_bench_row_lock_hash_is_the_digest_of_the_committed_lock_file() {
    use entrenar::train::setfit::bench_row::MethodEvidence;

    // A SetFit row claims `lock.lock_hash`; the FILE at `lock.lock_record_path` is the
    // evidence. 05-10 recomputes the digest from those bytes rather than trusting the field,
    // so this pins the relationship the gate will check.
    let temp = tempfile::TempDir::new().expect("tempdir");
    let bench_dir = temp.path().join("bench");
    let cell = CellKey::new(Method::Setfit, 16, 23);
    let payload = synthetic::payload(cell, 90_777_156);

    let lock_path = bench_dir.join(lock_relative_path(cell));
    std::fs::create_dir_all(lock_path.parent().expect("has a parent")).expect("creatable");
    std::fs::write(&lock_path, synthetic::LOCK_BYTES).expect("the committed lock is writable");

    let MethodEvidence::Setfit(evidence) = &payload.evidence else {
        panic!("the synthetic row is a setfit row");
    };
    let recomputed = sha256_hex(&std::fs::read(&lock_path).expect("the lock file reads"));
    assert_eq!(
        evidence.lock.lock_hash, recomputed,
        "the row's lock_hash must be recomputable from the committed file's bytes — a field \
         that only agreed with itself would attest nothing"
    );
    assert_eq!(
        bench_dir.join(&evidence.lock.lock_record_path),
        lock_path,
        "and the recorded path must be RELATIVE to the bench directory, so a transported row \
         resolves against the receiving host's tree"
    );
}

#[test]
fn setfit_bench_setfit_rows_have_equal_artifact_and_deployable_bytes() {
    // SetFit ships ONE standalone file, so its deployable size IS its artifact size. Asserted
    // rather than assumed because the LoRA side deliberately differs: an adapter-only figure
    // standing in for a deployable size is the size claim PF-008 exists to forbid.
    let payload = synthetic::payload(CellKey::new(Method::Setfit, 64, 53), 90_777_156);
    assert_eq!(payload.resource.artifact_bytes, 90_777_156);
    assert_eq!(
        payload.resource.deployable_total_bytes, payload.resource.artifact_bytes,
        "SetFit: one file, one size"
    );
}

// ==========================================================================================
// The row file is write-once
// ==========================================================================================

#[test]
fn setfit_bench_emit_row_refuses_an_existing_row_without_force() {
    let temp = tempfile::TempDir::new().expect("tempdir");
    let bench_dir = temp.path().join("bench");
    let cell = CellKey::new(Method::Setfit, 8, 13);

    emit_row(&bench_dir, synthetic::payload(cell, 1_000), false).expect("the first emit lands");
    let first = std::fs::read(bench_dir.join(ROWS_DIR).join(row_file_name(cell)))
        .expect("the row is readable");

    let error = emit_row(&bench_dir, synthetic::payload(cell, 2_000), false)
        .expect_err("re-running a completed cell must be refused, never silently duplicated");
    assert!(
        error.to_string().contains("--force"),
        "the refusal must name the flag; got: {error}"
    );
    assert_eq!(
        std::fs::read(bench_dir.join(ROWS_DIR).join(row_file_name(cell))).expect("still readable"),
        first,
        "and a refused re-run must not have touched the row it refused to replace"
    );
}

#[test]
fn setfit_bench_emit_row_records_the_digest_the_row_file_carries() {
    let temp = tempfile::TempDir::new().expect("tempdir");
    let bench_dir = temp.path().join("bench");
    let cell = CellKey::new(Method::Setfit, 8, 13);

    let (path, hash) =
        emit_row(&bench_dir, synthetic::payload(cell, 1_000), false).expect("the emit lands");

    // The manifest's recorded digest and the file's own envelope must be ONE value. If they
    // could differ, the resume path would compare a digest against a file that never had it.
    let on_disk = BenchRow::from_bytes(&std::fs::read(&path).expect("readable"))
        .expect("the emitted row verifies through the library's own door");
    assert_eq!(on_disk.semantic_hash, hash);
    let manifest = load_or_declare_manifest(&bench_dir).expect("the manifest reloads");
    assert_eq!(manifest.row_sha256(cell), Some(hash.as_str()));
}

// ==========================================================================================
// The LoRA cell: the candidate ledger and the size split
// ==========================================================================================

/// A synthetic LoRA row, for the size-field and attestation proofs.
fn lora_payload(
    cell: CellKey,
    base_bytes: u64,
    adapter_bytes: u64,
) -> entrenar::train::setfit::bench_row::BenchRowPayload {
    use entrenar::train::setfit::bench_row::{LoraEvidence, MethodEvidence};

    let mut payload = synthetic::payload(cell, adapter_bytes);
    payload.resource.artifact_bytes = adapter_bytes;
    payload.resource.deployable_total_bytes = base_bytes + adapter_bytes;
    payload.evidence = MethodEvidence::Lora(LoraEvidence {
        base_model_sha256: "ff".repeat(32),
        base_model_bytes: base_bytes,
        adapter_sha256: "ab".repeat(32),
        epochs_requested: 3,
        epochs_completed: 3,
        early_stopping_disabled: true,
        val_split: 0.0,
        no_selection_attestation: true,
        candidate_ledger_sha256: "cd".repeat(32),
        candidates_trained: 1,
        candidate_ledger_path: ledger_relative_path(cell),
    });
    payload
}

#[test]
fn setfit_bench_lora_rows_split_adapter_bytes_from_deployable_bytes() {
    // The measured 05-06 preflight numbers, so the arithmetic is checked against a real
    // artifact pair rather than round ones: base.apr 783236, model.adapter.apr 10180,
    // deployable 793416.
    let cell = CellKey::new(Method::Lora, 8, 13);
    let payload = lora_payload(cell, 783_236, 10_180);

    assert_eq!(
        payload.resource.artifact_bytes, 10_180,
        "artifact_bytes is the ADAPTER ALONE — the honest answer to what this method produced"
    );
    assert_eq!(
        payload.resource.deployable_total_bytes, 793_416,
        "and deployable_total_bytes is base + adapter, the ONLY field a cross-method size \
         claim may be built on"
    );
    assert_ne!(
        payload.resource.artifact_bytes, payload.resource.deployable_total_bytes,
        "the two must DIFFER for LoRA: an adapter-only figure standing in for a deployable \
         size understates the method by orders of magnitude (PF-008)"
    );

    let entrenar::train::setfit::bench_row::MethodEvidence::Lora(evidence) = &payload.evidence
    else {
        panic!("the synthetic row is a lora row");
    };
    assert_eq!(
        evidence.base_model_bytes + payload.resource.artifact_bytes,
        payload.resource.deployable_total_bytes,
        "and the three fields must be arithmetically consistent, so a reader can check the \
         claim without owning the files"
    );
    assert_eq!(
        evidence.candidates_trained, 1,
        "exactly one candidate: a second is the uncontracted model selection the attestation \
         denies"
    );
    assert!(evidence.no_selection_attestation);
    assert!(evidence.early_stopping_disabled);
    assert_eq!(evidence.val_split, 0.0);
    assert_eq!(
        evidence.epochs_completed, evidence.epochs_requested,
        "equality is what makes `no epoch was selected on a metric` checkable from the row"
    );

    // And it survives the library's own door, which independently requires the method tag and
    // the evidence block to agree.
    let row = BenchRow::new(payload);
    let bytes = row.to_file_bytes().expect("serializes");
    BenchRow::from_bytes(&bytes).expect("a well-formed lora row verifies");
}

#[test]
fn setfit_bench_lora_ledger_refuses_a_second_candidate_without_force() {
    let temp = tempfile::TempDir::new().expect("tempdir");
    let cell = CellKey::new(Method::Lora, 32, 41);
    let ledger = temp.path().join(ledger_relative_path(cell));

    super::lora::append_candidate(&ledger, cell, false, "manifesthash", "confighash")
        .expect("the first candidate is recorded");
    let after_first = std::fs::read_to_string(&ledger).expect("the ledger is readable");
    assert_eq!(
        after_first.lines().filter(|l| !l.is_empty()).count(),
        1,
        "exactly one line"
    );
    assert!(
        after_first.contains("manifesthash") && after_first.contains("confighash"),
        "and it records what the run was: {after_first}"
    );

    let error = super::lora::append_candidate(&ledger, cell, false, "manifesthash", "confighash")
        .expect_err("a SECOND candidate for one cell must be refused");
    let rendered = error.to_string();
    assert!(
        rendered.contains("lora-s32-seed41.jsonl"),
        "the refusal must name the ledger path so an operator can inspect it; got: {rendered}"
    );
    assert_eq!(
        std::fs::read_to_string(&ledger).expect("still readable"),
        after_first,
        "and a refused append must not have written anything — a ledger that grew on the \
         refusal path would itself become the second candidate it refused"
    );
}

#[test]
fn setfit_bench_lora_ledger_force_starts_a_fresh_ledger_rather_than_appending() {
    // `candidates_trained == 1` must stay a true statement about the RUN being recorded, not
    // a count of every run this directory has ever seen. Appending under --force would make
    // the count grow forever and the contract's "exactly 1" unreachable after a single retry.
    let temp = tempfile::TempDir::new().expect("tempdir");
    let cell = CellKey::new(Method::Lora, 8, 13);
    let ledger = temp.path().join(ledger_relative_path(cell));

    super::lora::append_candidate(&ledger, cell, false, "first", "cfg").expect("first");
    super::lora::append_candidate(&ledger, cell, true, "second", "cfg").expect("forced replace");

    let text = std::fs::read_to_string(&ledger).expect("readable");
    assert_eq!(
        text.lines().filter(|l| !l.is_empty()).count(),
        1,
        "--force replaces rather than appends"
    );
    assert!(
        text.contains("second") && !text.contains("first"),
        "got: {text}"
    );
}

#[test]
fn setfit_bench_lora_ledger_is_appended_before_the_training_call_in_source_order() {
    // The ledger's whole value is that it is written BEFORE the run it describes and before
    // any test access. A ledger appended afterwards records only the candidate that survived,
    // which is precisely the selection it exists to reveal.
    let append_site = SETFIT_BENCH_SOURCE
        .find(&needle(&["append_candi", "date(\n"]))
        .expect("the ledger append site is in this file");
    let train_site = SETFIT_BENCH_SOURCE
        .find(&needle(&["run_classify_", "core(&run"]))
        .expect("the training call is in this file");
    assert!(
        append_site < train_site,
        "the ledger append (byte {append_site}) must precede the training call (byte \
         {train_site}) in source order"
    );
}

#[test]
fn setfit_bench_lora_path_drives_the_named_library_doors_and_no_second_implementation() {
    // The reload route is the one 05-06 Task 3 PROVED, called by those exact names. Forty
    // remote 9B cells against an improvised reload is the failure this ordering prevents.
    for (fragments, why) in [
        (vec!["run_classify_", "core"], "the shared training entry"),
        (
            vec!["load_", "adapter("],
            "05-06's strict, never-partial adapter reload",
        ),
        (
            vec!["predict_proba_", "tokenized("],
            "05-06's ordered probability-vector entry",
        ),
        (
            vec!["pre_", "tokenize("],
            "the pipeline's OWN tokenizer — not a reproduction of its byte-level fallback",
        ),
        (
            vec!["assemble_quality_", "block("],
            "the SAME method-agnostic assembly the setfit cell uses",
        ),
        (
            vec!["outcome.", "gpu"],
            "the GPU identity, carried out of the run rather than re-derived",
        ),
    ] {
        let needle = needle(&fragments);
        assert!(
            SETFIT_BENCH_SOURCE.contains(&needle),
            "the LoRA path must drive {why}"
        );
    }

    // And `outcome.gpu` is itself the PIPELINE's accessors, read one call away in the module
    // that owns the run. The scan crosses the file boundary deliberately: asserting only on
    // this file would prove the field is used and say nothing about where its value came
    // from, which is the whole D-12 question.
    const FINETUNE_SOURCE: &str = include_str!("finetune.rs");
    assert!(
        FINETUNE_SOURCE.contains(&needle(&["pipeline.gpu_", "name()"])),
        "ClassifyOutcome.gpu must come from ClassifyPipeline::gpu_name"
    );
    assert!(
        FINETUNE_SOURCE.contains(&needle(&["gpu_total_", "memory()"])),
        "and from ClassifyPipeline::gpu_total_memory"
    );

    // And no second implementation of the things those doors already own.
    for (fragments, why) in [
        (vec!["fn ", "softmax"], "no second softmax"),
        (vec!["exp", "()  //"], "no hand-rolled normalisation"),
    ] {
        let needle = needle(&fragments);
        assert_eq!(SETFIT_BENCH_SOURCE.matches(&needle).count(), 0, "{why}");
    }
}

#[test]
fn setfit_bench_lora_path_contains_no_default_training_literal() {
    // The three values 05-06 replaced: seed 42, val_split 0.2, patience 10. A benchmark cell
    // whose configuration fell back to a default table would be reproducible only by someone
    // who knew which build wrote it.
    for forbidden in ["seed: 42", "val_split: 0.2", "early_stopping_patience: 10"] {
        assert!(
            !SETFIT_BENCH_SOURCE.contains(forbidden),
            "the bench path must carry no default training literal; found `{forbidden}`"
        );
    }
}

#[test]
fn setfit_bench_lora_backend_identity_cannot_fabricate_a_gpu() {
    // Verification Discipline rule 2: if the row says GPU, a pipeline-read GPU name must
    // exist. The cpu arm is reached exactly when the pipeline reported none, and it names no
    // device it did not observe.
    assert!(
        SETFIT_BENCH_SOURCE.contains("\"cpu:entrenar-classify:lora\""),
        "the no-GPU arm must be a fixed cpu identity"
    );
    assert!(
        SETFIT_BENCH_SOURCE.contains("gpu({name})"),
        "and the GPU arm must interpolate the name the PIPELINE reported"
    );
    // There is no path from a flag to the identity: `--gpu-backend` is passed to the trainer,
    // never read back into the row.
    assert_eq!(
        SETFIT_BENCH_SOURCE
            .matches(&needle(&["gpu_backend", ":"]))
            .count(),
        1,
        "gpu_backend appears exactly once, as the ClassifyRun field it hands to the trainer — \
         never as a source for backend_identity"
    );
}

#[test]
fn setfit_bench_lora_adapter_path_is_the_completed_epoch_not_the_newest_directory() {
    // Resolved BY NAME from the epoch the run reported. `ClassifyTrainer::save_checkpoint`'s
    // result is discarded (`let _ = ...`), so a silently-failed final write must surface as a
    // missing file rather than as an earlier epoch's adapter being attested as the trained
    // model.
    let dir = std::path::Path::new("/bench/lora/lora-s8-seed13");
    assert_eq!(
        super::lora::adapter_path_for(dir, 3),
        dir.join("epoch-2").join("model.adapter.apr"),
        "three completed epochs are 0,1,2 — the final adapter is epoch-2"
    );
    assert_eq!(
        super::lora::adapter_path_for(dir, 1),
        dir.join("epoch-0").join("model.adapter.apr")
    );
    assert_eq!(
        super::lora::adapter_path_for(dir, 0),
        dir.join("epoch-0").join("model.adapter.apr"),
        "a zero-epoch run has no adapter; the path is still well-formed so the caller's \
         is_file() check is what refuses, with a message naming the file"
    );
}

#[test]
fn setfit_bench_lora_row_predictions_door_checks_its_shapes() {
    use entrenar::train::setfit::apr_evaluate::row_predictions_from_lora;

    let labels = synthetic::labels();

    let rows = row_predictions_from_lora(
        &[vec![0.7, 0.2, 0.1], vec![0.1, 0.1, 0.8]],
        &[0, 2],
        &labels,
        "adapterhash",
        "test",
    )
    .expect("well-formed vectors are accepted");
    assert_eq!(
        rows.predicted(),
        &[0, 2],
        "argmax with the lowest index winning a tie — the same deterministic rule the SetFit \
         head's reduction uses, so a tie does not resolve differently per method"
    );
    assert_eq!(rows.split_tag(), "test");
    assert_eq!(rows.ordered_labels(), labels.as_slice());

    // A tie must go to the LOWEST index, asserted rather than assumed.
    let tied = row_predictions_from_lora(
        &[vec![0.5, 0.5, 0.0]],
        &[0],
        &labels,
        "adapterhash",
        "validation",
    )
    .expect("a tie is not an error");
    assert_eq!(tied.predicted(), &[0]);

    // And the refusals.
    assert!(
        row_predictions_from_lora(&[], &[], &labels, "h", "test").is_err(),
        "an empty split is a refusal, not a row of NaNs"
    );
    assert!(
        row_predictions_from_lora(&[vec![0.5, 0.5, 0.0]], &[0, 1], &labels, "h", "test").is_err(),
        "probability rows and truth rows must agree in count"
    );
    assert!(
        row_predictions_from_lora(&[vec![0.5, 0.5]], &[0], &labels, "h", "test").is_err(),
        "a probability row narrower than the label map would publish per-class numbers under \
         another class's name"
    );
    assert!(
        row_predictions_from_lora(&[vec![0.5, 0.5, 0.0]], &[7], &labels, "h", "test").is_err(),
        "a truth index outside the map would index a metric vector's wrong slot"
    );
}

// ==========================================================================================
// Source assertions: the shape the resource protocol requires
// ==========================================================================================

#[test]
fn setfit_bench_resolves_the_cold_probe_child_through_current_exe() {
    let door = needle(&["std::env::current_", "exe()"]);
    assert!(
        SETFIT_BENCH_SOURCE.contains(&door),
        "the cold-probe child must be THIS binary resolved through current_exe — never a bare \
         `apr`, which once resolved to a 26-day-old build on this very dev box"
    );
}

#[test]
fn setfit_bench_creates_files_only_through_the_shared_atomic_writer() {
    // The proof that an interrupted cell leaves either NO row or one complete digest-valid
    // row is `atomic_write`'s — temp file in the destination directory, sync, ONE rename,
    // cleanup on every error path — and it is proven once, in
    // `setfit_train::tests::setfit_train_a_failed_write_leaves_no_partial_file`. What THIS
    // file has to guarantee is that it never writes around that writer, which is a property
    // of its own source.
    for (fragments, why) in [
        (
            vec!["fs::", "rename("],
            "no rename site of its own: the atomicity is the shared writer's",
        ),
        (vec!["File::", "create("], "no unconditional file creation"),
        (
            vec!["fs::", "write("],
            "and no unsynced convenience write, which would leave a partial row on a crash",
        ),
    ] {
        let needle = needle(&fragments);
        assert_eq!(SETFIT_BENCH_SOURCE.matches(&needle).count(), 0, "{why}");
    }
    assert!(
        SETFIT_BENCH_SOURCE.contains(&needle(&["atomic_", "write("])),
        "and it does go through the shared writer"
    );

    // THE ONE DELIBERATE EXCEPTION, pinned so it stays one. The candidate ledger is
    // APPEND-ONLY: a rename-based writer replaces its destination, which would erase the
    // accumulated line the ledger exists to reveal. So it opens with `OpenOptions` — exactly
    // once, and with `append`.
    let open_options = needle(&["OpenOptions", "::new()"]);
    assert_eq!(
        SETFIT_BENCH_SOURCE.matches(&open_options).count(),
        1,
        "exactly ONE hand-opened file in this module, and it is the append-only ledger"
    );
    assert!(
        SETFIT_BENCH_SOURCE.contains(&needle(&[".append", "(!force)"])),
        "and it opens in APPEND mode: a truncating ledger could not record a second candidate"
    );
}

// ==========================================================================================
// The 40-cell driver script
// ==========================================================================================

/// The driver, embedded at compile time.
///
/// `include_str!` rather than a runtime read, on `bench_row`'s precedent: a gate that silently
/// skips when its input is absent proves nothing, and a path that resolves differently under
/// `cargo test` and in a packaged crate is a defect waiting for a release.
const DRIVER_SOURCE: &str = include_str!("../../../../scripts/run_bench_cells.sh");

/// The driver with every comment line removed.
///
/// The gates below are about what the script DOES. Its own explanatory comments say the words
/// `--jobs` and `rm` deliberately — the header explains at length why there is no parallel
/// dispatch — and a scan that counted those would be a gate the documentation could turn red.
fn driver_without_comments() -> String {
    DRIVER_SOURCE
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn driver_has_no_parallel_dispatch_construct() {
    // Concurrent cell processes would race the run-manifest AND invalidate every EVAL-05
    // resource number by contending for CPU and memory. There is deliberately no `--jobs`,
    // and the absence is a gate rather than a convention.
    let body = driver_without_comments();
    for construct in ["--jobs", "xargs -P", "parallel "] {
        assert!(
            !body.contains(construct),
            "the driver must contain no parallel-dispatch construct; found `{construct}`"
        );
    }
    for line in body.lines() {
        assert!(
            !line.trim_end().ends_with('&') || line.trim_end().ends_with("&&"),
            "a background-job `&` would be a parallel dispatch with no flag to grep for: {line}"
        );
    }
}

#[test]
fn driver_pins_the_binary_and_sets_the_shell_flags() {
    assert!(
        DRIVER_SOURCE.contains("set -euo pipefail"),
        "an executable script sets the flags; only a SOURCED library stays option-neutral"
    );
    assert!(
        DRIVER_SOURCE.contains(". scripts/apr_bin.sh || exit 1"),
        "the pin is SOURCED and fails by return status — `set` in a sourced file mutates the \
         caller's shell, which once killed the nightly six lines in"
    );
    assert!(
        !DRIVER_SOURCE.contains("\"apr\" ") && !DRIVER_SOURCE.contains("\napr "),
        "and never a bare `apr`: four binaries once coexisted on this dev box and a bare `apr` \
         resolved to a 26-day-old one"
    );
}

#[test]
fn driver_never_reads_a_status_through_a_pipe() {
    // Every `rc=$?` must sit on the line immediately after the command it describes, and no
    // line may pipe into something whose status is then read. This defect shipped twice here
    // (#2336 captured tee's status, #2360 captured grep's) and both times produced a gate
    // that could not fail while printing success.
    let body = driver_without_comments();
    let lines: Vec<&str> = body.lines().map(str::trim).collect();

    // FORM 1 — a bare `rc=$?` on its own line. The line BEFORE it must not be a pipeline.
    for (index, line) in lines.iter().enumerate() {
        if !line.starts_with("rc=$?") {
            continue;
        }
        let previous = lines
            .get(index.wrapping_sub(1))
            .copied()
            .unwrap_or_default();
        assert!(
            !previous.contains('|'),
            "`rc=$?` at line {index} follows a pipeline (`{previous}`), so it captures the LAST \
             command's status rather than the one it appears to describe"
        );
    }

    // FORM 2 — `... || rc=$?` inline, which is the form THIS driver deliberately uses: it runs
    // under `set -e`, so a bare `rc=$?` on the next line is unreachable after a non-zero exit.
    // The same defect is reachable here — `cmd | tee log || rc=$?` captures tee's status — so
    // the inline form is scanned too. Scanning only form 1 left this half unguarded.
    for (index, line) in lines.iter().enumerate() {
        let Some(before) = line
            .split("|| rc=$?")
            .next()
            .filter(|_| line.contains("|| rc=$?"))
        else {
            continue;
        };
        // Strip the `||` that belongs to the capture itself before looking for a pipe.
        let command = before.trim_end().trim_end_matches('|');
        assert!(
            !command.contains('|'),
            "`|| rc=$?` at line {index} captures the status of a PIPELINE (`{command}`), which \
             is the last command's status, not the one it appears to describe"
        );
    }

    // NON-VACUITY. The scan must have something to scan — a driver with no rc capture in EITHER
    // form would satisfy both loops above trivially.
    //
    // This assertion used to look only for form 1 and had been unsatisfiable since the driver
    // moved to the inline form under `set -e` (commit 9920feae8): every remaining bare `rc=$?`
    // in the file sits in a COMMENT, and comments are stripped before the scan. Repaired rather
    // than deleted — a non-vacuity guard that cannot pass is the same defect it exists to catch.
    let captures = lines
        .iter()
        .filter(|line| line.starts_with("rc=$?") || line.contains("|| rc=$?"))
        .count();
    assert!(
        captures > 0,
        "the driver captures no exit status in either form, so both scans above were vacuous"
    );
}

#[test]
fn driver_holds_a_single_writer_lock_with_distinct_failure_exit_codes() {
    let body = driver_without_comments();
    assert!(
        body.contains("noclobber"),
        "the lock must be taken with an ATOMIC primitive; a test-then-create has a window two \
         drivers can both pass"
    );
    assert!(
        body.contains("EXIT_LOCKED"),
        "and a held lock has its own exit code"
    );

    // The evidence class and the transient class must be DISTINGUISHABLE. Conflating them
    // would make "just re-run it" the advice for a finding no re-run can fix.
    for code in ["EXIT_EVIDENCE=3", "EXIT_TRANSIENT=4", "EXIT_LOCKED=5"] {
        assert!(body.contains(code), "missing distinct exit code `{code}`");
    }
    assert!(
        body.contains("is_evidence_failure"),
        "and the classifier that chooses between them"
    );
    assert!(
        body.contains("UncalibratedRegime"),
        "whose vocabulary names the refusals 05-03's coverage claim depends on halting for"
    );
}

#[test]
fn driver_resume_is_hash_based_not_a_bare_existence_check() {
    let body = driver_without_comments();
    assert!(
        body.contains("semantic_hash"),
        "a row file that EXISTS is not evidence the cell completed; resume compares the row's \
         own envelope digest against the digest the run manifest recorded"
    );
    assert!(
        body.contains("run-manifest.json"),
        "and the manifest is what completeness is defined by — a directory listing can only \
         report what is present, never what is missing"
    );
    assert!(
        body.contains("cell_is_complete"),
        "through one predicate, so the skip decision has one definition"
    );
}

#[test]
fn driver_writes_and_reads_the_selection_at_one_path() {
    // The generator and the consumer build this path in two places. They must agree, and the
    // agreement is worth a gate rather than a convention: `apr data select --output` takes a
    // DIRECTORY and writes `selection-manifest.json` inside it, so an earlier version of this
    // script that passed a FILE path would have written
    // `.../s8-seed13.json/selection-manifest.json` and pointed every `--selection` at nothing
    // — a failure that surfaces forty cells later as "file not found".
    let body = driver_without_comments();
    let stem = "$BENCH_DIR/selections/s${shots}-seed${seed}";
    let write_site = format!("target_dir=\"{stem}\"");
    let read_site = format!("selection=\"{stem}/selection-manifest.json\"");
    assert!(
        body.contains(&write_site),
        "the generator must write into the per-cell directory; got:\n{body}"
    );
    assert!(
        body.contains(&read_site),
        "and the consumer must read selection-manifest.json from that SAME directory"
    );
    assert!(
        body.contains("--output \"$target_dir\""),
        "`--output` takes a directory — passing a file path silently nests the manifest one \
         level deeper"
    );
    assert!(
        body.contains("if [[ ! -f \"$target\" ]]"),
        "and the generator must prove the file appeared: `apr data select` exiting 0 is not \
         evidence that THIS path now holds a manifest"
    );
}

#[test]
fn driver_covers_exactly_the_contracted_matrix() {
    let body = driver_without_comments();
    // The matrix literals in the script must be the contract's, checked against the LIBRARY's
    // constants rather than against a second copy of the four numbers.
    for shots in BENCH_SHOTS {
        assert!(
            body.contains(&shots.to_string()),
            "the driver's shot list must contain {shots}"
        );
    }
    for seed in BENCH_SEEDS {
        assert!(
            body.contains(&seed.to_string()),
            "the driver's seed list must contain {seed}"
        );
    }
    assert!(
        !body.contains("SEEDS=(13 17 23 29 31 37 41 42"),
        "and 42 is NOT among them"
    );
    // The vacuity floor: a loop that covered nothing must not exit 0 with a tally of zeroes.
    assert!(
        body.contains("-ne 40"),
        "the driver must assert its own coverage count — CR-02's precedent is a zero-match \
         filter that printed `test result: ok`"
    );
}

#[test]
fn setfit_bench_never_reads_a_status_through_a_pipe() {
    // The comment-filtered scan the plan's acceptance criterion specifies, run in-process so a
    // future edit turns a test red rather than waiting for someone to run the grep.
    let offending = SETFIT_BENCH_SOURCE
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .filter(|line| {
            let has_pipe_into_grep = line.contains('|') && line.contains("grep");
            has_pipe_into_grep && line.contains("$?")
        })
        .count();
    assert_eq!(
        offending, 0,
        "a status read through a pipe is the LAST command's status; that defect has shipped \
         twice in this repository (#2336, #2360)"
    );
}

// ==========================================================================================
// `apr setfit bench report` — the renderer (05-10 task 2, D-13/D-15)
// ==========================================================================================
//
// THE RENDERER IS TESTED AGAINST A SYNTHETIC AGGREGATE, NOT A BUILT DIRECTORY.
//
// `verify_run` and `aggregate` have their own eighty-row suite in
// `entrenar::train::setfit::bench_gate`, including the six doctored negatives. What is left to
// get wrong HERE is the presentation — a technically-true table that reads as a like-for-like
// benchmark, an adapter's bytes standing in for a deployable size, a verdict word, a blank cell
// where a degenerate interval belongs. Those are properties of a `RunAggregate` -> `String`
// function, and constructing the input directly is what lets each be varied ONE AT A TIME. The
// two-sided incomparability control below is the clearest case: the same fixture with one
// mechanism string changed is what separates a real label from unconditional boilerplate.

mod report_render {
    use entrenar::train::setfit::bench_gate::{
        Ci95, MechanismClass, MethodShotQuality, MethodShotResource, RunAggregate, SeedDelta,
        SeedValue, SeriesSummary, ShotDelta, ZERO_VARIANCE_NULL_REASON,
    };
    use entrenar::train::setfit::bench_row::{
        Method, BENCH_METHODS, BENCH_SEEDS, BENCH_SHOTS, CLAIMS_CONTRACT_ID,
    };

    use crate::commands::setfit_bench::report::{
        render_deltas, render_human, render_quality, render_resource_comparison,
        render_resource_detail, render_sizes, verified_aggregate, ReportPayload,
        ADAPTER_ONLY_LABEL, ARTIFACT_BYTES_LABEL, ARTIFACT_BYTES_LABEL_TWO_METHOD, CI_UNAVAILABLE,
        COMPARISON_ROW_MARKER, DELTA_TABLE_HEADER, ESTIMATION_FIRST_NOTE,
        ESTIMATION_FIRST_NOTE_PAIRED, INCOMPARABLE_NOTE, PER_HOST_FRAMING,
        PER_HOST_FRAMING_TWO_HOST, PROVENANCE_SOURCES, PROVENANCE_SOURCES_TWO_METHOD,
        QUALITY_TABLE_HEADER, REPORT_PAYLOAD_SCHEMA, RESIDUAL_DISCLOSURE,
        RESOURCE_COMPARISON_HEADER, SAMPLED_LOWER_BOUND_LABEL, SEED_CI_LABEL, SINGLE_METHOD_NOTE,
        SINGLE_METHOD_TITLE, SIZE_TABLE_FOOTNOTE, SIZE_TABLE_FOOTNOTE_TWO_METHOD,
        SIZE_TABLE_HEADER, TWO_METHOD_TITLE,
    };

    /// LoRA's ADAPTER-ONLY byte count.
    ///
    /// Chosen so its decimal rendering is not a substring of either deployable figure —
    /// otherwise "the adapter figure is absent from the size table" would be decided by digit
    /// coincidence rather than by the property under test. (`40_000_000` was the obvious choice
    /// and is a substring of `18_040_000_000`.)
    const LORA_ADAPTER_BYTES: f64 = 41_943_040.0;
    /// LoRA's deployable total: base + adapter.
    const LORA_DEPLOYABLE_BYTES: f64 = 18_253_611_008.0;
    /// SetFit ships ONE standalone file, so its two size figures are equal by construction.
    const SETFIT_ARTIFACT_BYTES: f64 = 94_371_840.0;

    fn summary(mean: f64) -> SeriesSummary {
        SeriesSummary {
            n: 10,
            mean,
            std: 0.0025,
            min: mean - 0.01,
            max: mean + 0.01,
        }
    }

    fn resource_group(
        method: Method,
        shots: u32,
        train_mechanism: &str,
        train_class: MechanismClass,
    ) -> MethodShotResource {
        let (artifact, deployable, host, backend) = match method {
            Method::Setfit => (
                SETFIT_ARTIFACT_BYTES,
                SETFIT_ARTIFACT_BYTES,
                "local-cpu (macos/aarch64)",
                "cpu:trueno:simd",
            ),
            Method::Lora => (
                LORA_ADAPTER_BYTES,
                LORA_DEPLOYABLE_BYTES,
                "lambda-vector (linux/x86_64)",
                "gpu:cuda:cublas",
            ),
        };
        MethodShotResource {
            method,
            shots,
            hosts: vec![host.to_string()],
            backends: vec![backend.to_string()],
            train_wall_ms: summary(1_000.0),
            cold_latency_ms: summary(12.0),
            warm_latency_ms_median: summary(4.0),
            throughput_rows_per_sec: summary(200.0),
            throughput_batch_sizes: vec![32],
            train_peak_rss_bytes: summary(500_000_000.0),
            train_peak_rss_mechanisms: vec![train_mechanism.to_string()],
            train_peak_rss_mechanism_classes: vec![train_class],
            inference_peak_rss_bytes: summary(200_000_000.0),
            inference_peak_rss_mechanisms: vec!["child_max_rss_vm_hwm".to_string()],
            inference_peak_rss_mechanism_classes: vec![MechanismClass::ExactKernelHighWaterMark],
            artifact_bytes: summary(artifact),
            deployable_total_bytes: summary(deployable),
        }
    }

    /// A synthetic aggregate.
    ///
    /// `lora_train_mechanism` is the knob the two-sided incomparability control turns: a sampled
    /// lower bound beside an exact high-water mark must be labelled, and two figures of the SAME
    /// class must not be — a note that always fires is boilerplate a reader learns to skip,
    /// which is the same as no note at all.
    fn synthetic_aggregate(
        zero_variance_shots: Option<u32>,
        lora_train_mechanism: &str,
        lora_train_class: MechanismClass,
    ) -> RunAggregate {
        let mut quality = Vec::new();
        let mut resource = Vec::new();
        let mut key_sequence = Vec::new();
        for method in BENCH_METHODS {
            for shots in BENCH_SHOTS {
                let base = if method == Method::Setfit { 0.55 } else { 0.45 };
                quality.push(MethodShotQuality {
                    method,
                    shots,
                    f_avg: summary(base),
                    macro_f1: summary(base - 0.05),
                    mcc: summary(base - 0.10),
                    f_avg_seed_ci95: Ci95 {
                        low: Some(base - 0.02),
                        high: Some(base + 0.02),
                        half_width: Some(0.02),
                        std_err: Some(0.008),
                        null_reason: None,
                    },
                    per_seed: BENCH_SEEDS
                        .iter()
                        .map(|seed| SeedValue {
                            seed: *seed,
                            f_avg: base,
                            f_avg_bits: base.to_bits(),
                            macro_f1: base - 0.05,
                            mcc: base - 0.10,
                        })
                        .collect(),
                });
                let (mechanism, class) = if method == Method::Setfit {
                    ("vm_hwm", MechanismClass::ExactKernelHighWaterMark)
                } else {
                    (lora_train_mechanism, lora_train_class)
                };
                resource.push(resource_group(method, shots, mechanism, class));
                for seed in BENCH_SEEDS {
                    key_sequence.push(format!("{}/s{shots}/seed{seed}", method.tag()));
                }
            }
        }

        let deltas = BENCH_SHOTS
            .iter()
            .map(|shots| {
                let degenerate = zero_variance_shots == Some(*shots);
                ShotDelta {
                    shots: *shots,
                    per_seed_deltas: BENCH_SEEDS
                        .iter()
                        .map(|seed| SeedDelta {
                            seed: *seed,
                            delta: 0.10,
                            delta_bits: 0.10_f64.to_bits(),
                        })
                        .collect(),
                    mean_delta: 0.10,
                    std_delta: if degenerate { None } else { Some(0.01) },
                    ci95: if degenerate {
                        Ci95 {
                            low: None,
                            high: None,
                            half_width: None,
                            std_err: None,
                            null_reason: Some(ZERO_VARIANCE_NULL_REASON.to_string()),
                        }
                    } else {
                        Ci95 {
                            low: Some(0.09),
                            high: Some(0.11),
                            half_width: Some(0.01),
                            std_err: Some(0.004),
                            null_reason: None,
                        }
                    },
                    t_statistic: if degenerate { None } else { Some(25.0) },
                    p_value: if degenerate { None } else { Some(0.000_001) },
                }
            })
            .collect();

        RunAggregate {
            contract_id: CLAIMS_CONTRACT_ID.to_string(),
            n_seeds: 10,
            degrees_of_freedom: 9,
            t_crit_975_df9: 2.262_157_162_798_205,
            key_sequence,
            methods: BENCH_METHODS.iter().map(|m| m.tag().to_string()).collect(),
            quality,
            deltas,
            resource,
        }
    }

    /// The everyday report: two hosts, two train mechanisms, no degenerate level.
    ///
    /// DEFERRED SCOPE. Retained so the two-method renderer keeps a running proof and
    /// D-ITEM-05-15 restores a renderer that still works.
    fn ordinary() -> RunAggregate {
        synthetic_aggregate(
            None,
            "sysinfo_sampled_10hz",
            MechanismClass::SampledLowerBound,
        )
    }

    /// The ACTIVE-scope report: ONE method, no deltas, one host.
    ///
    /// Built by NARROWING `ordinary()` exactly as `aggregate` narrows a verified set — the
    /// second method's groups are dropped and the delta list is emptied — so the fixture cannot
    /// describe a shape the library cannot produce.
    fn active_scope() -> RunAggregate {
        let mut report = ordinary();
        report.methods = vec![Method::Setfit.tag().to_string()];
        report.quality.retain(|g| g.method == Method::Setfit);
        report.resource.retain(|g| g.method == Method::Setfit);
        report.key_sequence.retain(|k| k.starts_with("setfit/"));
        report.deltas.clear();
        report
    }

    /// The literals the TWO-METHOD renderer actually emitted, and which the ACTIVE report must
    /// not contain anywhere.
    ///
    /// EVERY ROW IS A REAL PRIOR OUTPUT, NOT AN INVENTED NEAR-MISS. Each is either a shipped
    /// constant of the two-method renderer or a fragment copied out of its own format string,
    /// so the table asserts against text this program genuinely used to print. A must-not-match
    /// list of plausible-looking strings nobody ever emitted would pass forever while the real
    /// literal sat in the report.
    ///
    /// Provenance, recorded here rather than left to a reader to reconstruct:
    /// - `DELTA_TABLE_HEADER`      — `render_deltas`, first line of its output
    /// - `RESOURCE_COMPARISON_HEADER` — `render_resource_comparison`, first line
    /// - `COMPARISON_ROW_MARKER`   — `comparison_row`'s own format string, the `"  |  lora "`
    ///                               separator between the two sides of one row
    /// - `TWO_METHOD_TITLE`        — `render_header`'s title line
    /// - `ESTIMATION_FIRST_NOTE_PAIRED` — the note that advertised paired intervals
    /// - `PER_HOST_FRAMING_TWO_HOST`    — the framing line that described a two-host design
    /// - `ARTIFACT_BYTES_LABEL_TWO_METHOD` — `render_resource_detail`'s per-group artifact-bytes
    ///                               ROW LABEL, copied out of its own format string. Added by
    ///                               plan 05-13: it survived the 05-11 retarget because the
    ///                               table only carried section-level literals, and it printed
    ///                               the deferred method's name on all four active groups.
    /// - `SIZE_TABLE_FOOTNOTE_TWO_METHOD` — `render_sizes`'s trailing paragraph. Same origin,
    ///                               same omission: four sentences of cross-method claim
    ///                               language under a scope with one method.
    /// - `PROVENANCE_SOURCES_TWO_METHOD` — the fragment `render_header`'s `verified:` block used
    ///                               to name; it credited a candidate-ledger recomputation the
    ///                               active scope has no ledger to perform.
    fn must_not_match_literals() -> Vec<(&'static str, &'static str)> {
        vec![
            ("delta table header", DELTA_TABLE_HEADER),
            ("cross-method comparison header", RESOURCE_COMPARISON_HEADER),
            ("per-row comparison format", COMPARISON_ROW_MARKER),
            ("two-method title line", TWO_METHOD_TITLE),
            ("paired estimation-first note", ESTIMATION_FIRST_NOTE_PAIRED),
            ("two-host resource framing", PER_HOST_FRAMING_TWO_HOST),
            (
                "two-method artifact-bytes row label",
                ARTIFACT_BYTES_LABEL_TWO_METHOD,
            ),
            ("cross-method size footnote", SIZE_TABLE_FOOTNOTE_TWO_METHOD),
            (
                "two-method provenance sources",
                PROVENANCE_SOURCES_TWO_METHOD,
            ),
        ]
    }

    #[test]
    fn setfit_bench_report_active_scope_matches_the_case_table() {
        let rendered = render_human(&active_scope());

        // MUST MATCH — what the active report has to contain to be worth publishing.
        for (label, needle) in [
            ("the single-method title", SINGLE_METHOD_TITLE),
            ("the quality table", QUALITY_TABLE_HEADER),
            ("the seed-dispersion label", SEED_CI_LABEL),
            ("the estimation-first note", ESTIMATION_FIRST_NOTE),
            ("the single-method scope note", SINGLE_METHOD_NOTE),
            ("the single-host resource framing", PER_HOST_FRAMING),
            ("the size table", SIZE_TABLE_HEADER),
            (
                "the single-method artifact-bytes label",
                ARTIFACT_BYTES_LABEL,
            ),
            ("the single-method size footnote", SIZE_TABLE_FOOTNOTE),
            ("the active provenance sources", PROVENANCE_SOURCES),
            // 05-17. THE DISCLOSURE IS GATED, not merely edited. A residual line is a claim
            // about the report's own strength, and it drifts the way any other claim does.
            ("the corrected residual disclosure", RESIDUAL_DISCLOSURE),
        ] {
            assert!(
                rendered.contains(needle),
                "{label} is MISSING from the active-scope report"
            );
        }

        // MUST NOT MATCH — the real literals the two-method renderer emitted.
        for (label, needle) in must_not_match_literals() {
            assert!(
                !rendered.contains(needle),
                "the active-scope report emitted the two-method literal `{label}` \
                 ({needle:?}); the cross-method sections are REMOVED under this scope, not \
                 emitted empty"
            );
        }

        // NON-VACUITY, BOTH WAYS. The must-not-match rows must be capable of matching
        // SOMETHING, or the loop above proves nothing: every one of them appears in the
        // TWO-METHOD render of the same fixture.
        let two_method = render_human(&ordinary());
        for (label, needle) in must_not_match_literals() {
            assert!(
                two_method.contains(needle),
                "must-not-match row `{label}` never appears in ANY render, so asserting its \
                 absence from the active report is vacuous"
            );
        }
    }

    #[test]
    fn setfit_bench_report_residual_concedes_exactly_what_the_gate_still_cannot_refuse() {
        // 05-17. THE RESIDUAL LINE IS A CLAIM ABOUT THE REPORT'S OWN STRENGTH, and it is held
        // to the same standard as the numbers: it must not overstate what the gate enforces,
        // and it must not keep conceding something the gate now refuses. Both directions are
        // defects — an understated disclosure teaches a reader to trust real evidence less than
        // it warrants, which is over-claiming with the sign flipped.
        let rendered = render_human(&active_scope());

        // GONE — the sentence this program actually used to print, copied verbatim out of its
        // own format string rather than invented as a near-miss. It conceded a forgery that
        // three separate checks now refuse: a doctored quality figure (the closed-form
        // cross-check, 05-17), an escaping evidence path (containment, 05-15), and a doctored
        // pairing key (the selection-manifest recomputation, 05-16).
        for (label, needle) in [
            (
                "the retired unqualified-forgery concession",
                "a producer holding both the rows and those files could still emit a",
            ),
            (
                "the retired consistency-not-truth sentence",
                "proves consistency, not truth",
            ),
        ] {
            assert!(
                !rendered.contains(needle),
                "the report still carries `{label}` ({needle:?}); the gate now refuses what \
                 that sentence concedes"
            );
        }

        // STILL CONCEDED — the three residuals that remain TRUE. Retiring the sentence without
        // replacing it would have been the opposite defect, and a bigger one.
        for (label, needle) in [
            ("the doctored-matrix residual", "the matrix itself is"),
            ("the calibration residual", "not recomputable at"),
            ("the uncommitted-artifact residual", "apr_artifact_sha256"),
            ("the evidence-table residual", "evidence_table_hash"),
        ] {
            assert!(
                rendered.contains(needle),
                "the corrected residual must still concede `{label}` ({needle:?})"
            );
        }

        // AND IT MUST NOT OVERSTATE. The cross-check proves consistency with the RECORDED
        // matrix; a residual claiming the matrix itself was verified would be the forgery this
        // disclosure exists to keep visible.
        let lower = rendered.to_lowercase();
        for overclaim in [
            "proves the matrix",
            "verified against the model",
            "cannot be forged",
        ] {
            assert!(
                !lower.contains(overclaim),
                "the residual overstates the gate with `{overclaim}`"
            );
        }

        // NON-VACUITY: the disclosure is actually present, so the absence assertions above are
        // not holding over a report that dropped the line entirely.
        assert!(rendered.contains(RESIDUAL_DISCLOSURE));
        assert!(rendered.contains("residual:"));
    }

    #[test]
    fn setfit_bench_verify_cell_scope_note_enumerates_every_recomputation_the_door_performs() {
        // FINDING T-05-16-05, carried into this plan by 05-16's own threat model. The door's
        // printed `scope:` line named provenance alone while step 6 had grown the selection
        // binding (05-16) and the closed-form quality cross-check (05-17). Under-claiming is
        // the SAFE direction and is still a defect: a door whose own account of its coverage is
        // incomplete is exactly the class of artifact this phase exists to prevent.
        use crate::commands::setfit_bench::verify_cell::DOOR_SCOPE_NOTE;

        for (label, needle) in [
            ("the step list", "steps 1+4+6"),
            ("provenance", "provenance"),
            ("the selection manifest", "selection manifest"),
            ("the quality cross-check", "confusion matrix"),
            ("the excluded set-level steps", "NOT the set-level steps"),
        ] {
            assert!(
                DOOR_SCOPE_NOTE.contains(needle),
                "the door's scope note does not name `{label}` ({needle:?}): {DOOR_SCOPE_NOTE}"
            );
        }

        // AND IT MUST NOT CLAIM THE SET-LEVEL STEPS IT DELIBERATELY SKIPS. The under-claim is
        // fixed; the opposite error would be worse, because a reader would take a single-cell
        // pass for a run-level one.
        let lower = DOOR_SCOPE_NOTE.to_lowercase();
        for overclaim in [
            "every cell",
            "the whole run",
            "expectation-set equality holds",
        ] {
            assert!(
                !lower.contains(overclaim),
                "the door's scope note claims the set-level coverage it skips: `{overclaim}`"
            );
        }
    }

    /// The active report must not name the deferred METHOD anywhere, by any spelling.
    ///
    /// WHY A TOKEN CHECK AND NOT ONLY THE CASE TABLE. The case table asserts against literals
    /// that were once shipped, which is what makes it non-vacuous — but it can only refuse the
    /// literals someone thought to add. Plan 05-13 found the gap empirically: the retarget
    /// removed the delta and comparison SECTIONS and left the deferred method's name in a row
    /// label printed on all four active groups and in a four-sentence size footnote, neither of
    /// which the section-level table covered. This assertion is over the rendered TEXT, so it
    /// catches the next such label without anyone having to predict it.
    #[test]
    fn setfit_bench_report_active_scope_never_names_the_deferred_method() {
        let rendered = render_human(&active_scope()).to_lowercase();
        assert!(
            !rendered.contains("lora"),
            "the active-scope report named the deferred method; absence must read as absence              (D-19), and a reader who sees the name beside a number reads a comparison:              {rendered}"
        );
        for connective in ["versus", " vs ", "compared to", "paired delta"] {
            assert!(
                !rendered.contains(connective),
                "the active-scope report used the comparative connective `{connective}`"
            );
        }
        // NON-VACUITY. The token IS present in the two-method render of the same fixture, so
        // asserting its absence above is a real constraint rather than a check over a
        // vocabulary this program never had.
        let two_method = render_human(&ordinary()).to_lowercase();
        assert!(
            two_method.contains("lora"),
            "the deferred method's name never appears in ANY render, so asserting its absence              from the active report is vacuous"
        );
        assert!(rendered.len() > 1_000, "the report must have rendered");
    }

    #[test]
    fn setfit_bench_report_active_scope_emits_no_delta_or_comparison_section() {
        let rendered = render_human(&active_scope());
        // Not "an empty section": the headers are ABSENT. A header with no rows still tells a
        // reader a comparison was attempted.
        assert!(!rendered.contains("PAIRED DELTA"));
        assert!(!rendered.contains("RESOURCE COMPARISON"));
        // ... and the interval that DOES belong to this scope is present and labelled.
        assert!(rendered.contains(SEED_CI_LABEL));
        assert!(rendered.contains("95% CI (seeds)"));
    }

    #[test]
    fn setfit_bench_report_single_method_note_cannot_trip_the_gate_it_is_mandated_beside() {
        // THE TENSION RESOLVED IN THE SUITE RATHER THAN DISCOVERED IN 05-13. The note is
        // mandated by this plan; 05-13 gates the committed report against the same
        // must-not-match table. A note written with those words would trip a gate this plan
        // also mandates, which would pressure an executor into weakening one or gutting the
        // other. So the note is pinned against the table here.
        for (label, needle) in must_not_match_literals() {
            assert!(
                !SINGLE_METHOD_NOTE.contains(needle),
                "the mandated note contains the must-not-match literal `{label}`"
            );
        }
        // The constrained vocabulary, asserted rather than described.
        let lower = SINGLE_METHOD_NOTE.to_lowercase();
        for forbidden in [
            "lora", // must not name the deferred method
            " vs ", // no comparative connective
            "versus",
            "compared to",
            "beside",
            "delta",       // not the phrase for a paired difference
            "significant", // not the word for a statistical verdict
        ] {
            assert!(
                !lower.contains(forbidden),
                "the note used the forbidden term `{forbidden}`: {SINGLE_METHOD_NOTE}"
            );
        }
        // And it DOES carry the two identifiers and the permitted phrase, so the reader can
        // follow the absence to its record rather than reading it as parity.
        assert!(SINGLE_METHOD_NOTE.contains("D-19"));
        assert!(SINGLE_METHOD_NOTE.contains("D-ITEM-05-15"));
        assert!(SINGLE_METHOD_NOTE.contains("a second method"));
    }

    #[test]
    fn setfit_bench_report_renders_the_two_peaks_as_separate_labelled_figures() {
        let rendered = render_resource_detail(&active_scope().resource);

        // TWO figures, TWO labels — never one column and never one combined number.
        assert!(rendered.contains("train peak RSS (training process)"));
        assert!(rendered.contains("inference peak RSS (cold child)"));
        // Each carries its own mechanism string beside its number.
        assert_eq!(
            rendered.matches("mechanism:").count(),
            2 * BENCH_SHOTS.len(),
            "every peak figure in every group prints its own mechanism"
        );

        // A SAMPLED figure carries its lower-bound label WHERE IT IS RENDERED. Built with a
        // sampled train mechanism so the label has something to attach to.
        let mut sampled = active_scope();
        for group in &mut sampled.resource {
            group.train_peak_rss_mechanisms = vec!["sysinfo_sampled_10hz".to_string()];
            group.train_peak_rss_mechanism_classes = vec![MechanismClass::SampledLowerBound];
        }
        let sampled_render = render_resource_detail(&sampled.resource);
        assert!(
            sampled_render.contains(SAMPLED_LOWER_BOUND_LABEL),
            "a sampled peak must carry its lower-bound label at the point of presentation, \
             not only in a methods paragraph"
        );
        // NON-VACUITY: the exact-mechanism control does NOT carry it, so the label is not
        // simply always printed.
        let mut exact = active_scope();
        for group in &mut exact.resource {
            group.train_peak_rss_mechanisms = vec!["vm_hwm".to_string()];
            group.train_peak_rss_mechanism_classes = vec![MechanismClass::ExactKernelHighWaterMark];
            group.inference_peak_rss_mechanisms = vec!["vm_hwm".to_string()];
            group.inference_peak_rss_mechanism_classes =
                vec![MechanismClass::ExactKernelHighWaterMark];
        }
        assert!(
            !render_resource_detail(&exact.resource).contains(SAMPLED_LOWER_BOUND_LABEL),
            "a label that is always printed carries no information"
        );
    }

    #[test]
    fn setfit_bench_verify_cell_door_emits_none_of_the_report_statistic_literals() {
        // T-05-11-06. The door is a DIAGNOSTIC, and it must not be readable as a partial
        // report. Its entire output is the pass prefix plus the scope note, so the assertion
        // is over the text it can possibly print.
        use crate::commands::setfit_bench::verify_cell::{DOOR_SCOPE_NOTE, PASS_LINE_PREFIX};
        let door_output = format!("{PASS_LINE_PREFIX}setfit/s16/seed29\n{DOOR_SCOPE_NOTE}");

        // Every statistic literal the REPORT emits. If the door ever grows a number, one of
        // these fires.
        for (label, needle) in [
            ("the quality table header", QUALITY_TABLE_HEADER),
            ("the size table header", SIZE_TABLE_HEADER),
            ("the delta table header", DELTA_TABLE_HEADER),
            ("the comparison header", RESOURCE_COMPARISON_HEADER),
            ("the seed-CI label", SEED_CI_LABEL),
            ("the estimation-first note", ESTIMATION_FIRST_NOTE),
            ("the degenerate-interval marker", CI_UNAVAILABLE),
        ] {
            assert!(
                !door_output.contains(needle),
                "the single-cell door emitted the report literal `{label}`; it is a diagnostic \
                 door and the report is the only door that publishes numbers"
            );
        }
        for word in ["mean", "std", "95%", "dispersion", "interval"] {
            assert!(
                !door_output.to_lowercase().contains(word),
                "the door's output contains the statistic word `{word}`"
            );
        }

        // NON-VACUITY: the same absence assertions would hold over an empty string, so the
        // door's output must actually say something, and it must say WHICH steps it applied.
        assert!(door_output.contains("steps 1+4+6"));
        assert!(door_output.contains("NOT the set-level steps"));
        assert!(door_output.contains("setfit/s16/seed29"));
    }

    #[test]
    fn setfit_bench_report_deferred_scope_still_renders_the_comparison() {
        // The two-method renderer is RETAINED and unexercised, not deleted. Restoring the arm
        // must restore a renderer that still works.
        let rendered = render_human(&ordinary());
        assert!(rendered.contains(DELTA_TABLE_HEADER));
        assert!(rendered.contains(RESOURCE_COMPARISON_HEADER));
        assert!(rendered.contains(COMPARISON_ROW_MARKER));
        assert!(rendered.contains(TWO_METHOD_TITLE));
        assert!(rendered.contains(ARTIFACT_BYTES_LABEL_TWO_METHOD));
        assert!(rendered.contains(SIZE_TABLE_FOOTNOTE_TWO_METHOD));
        assert!(rendered.contains(PROVENANCE_SOURCES_TWO_METHOD));
    }

    #[test]
    fn setfit_bench_report_prints_no_verdict_word_and_names_the_interval() {
        let rendered = render_human(&ordinary()).to_lowercase();
        // D-08. A binary verdict is precisely where few-shot seed sensitivity hides: rankings
        // that reverse across seeds become one word.
        for verdict in ["significant", "significantly", "better than", "worse than"] {
            assert!(
                !rendered.contains(verdict),
                "the report rendered the verdict word `{verdict}`; claim language is \
                 estimation-first (D-08)"
            );
        }
        // NON-VACUITY: every absence assertion above also holds over an empty string.
        assert!(
            rendered.contains("95% ci"),
            "the report must state the interval it publishes"
        );
        assert!(rendered.len() > 1_000, "the report must have rendered");
    }

    #[test]
    fn setfit_bench_report_human_output_carries_the_per_host_framing_line() {
        let rendered = render_human(&ordinary());
        assert!(
            rendered.contains(PER_HOST_FRAMING),
            "D-09: resource figures are as-deployed method costs on two deliberately different \
             hosts, and a report that does not say so states something no measurement supports"
        );
        assert!(rendered.contains(QUALITY_TABLE_HEADER));
        assert!(rendered.contains(SIZE_TABLE_HEADER));
    }

    #[test]
    fn setfit_bench_report_labels_a_mixed_mechanism_comparison_and_leaves_a_matched_one_alone() {
        // MIXED: a sampled LOWER BOUND beside an exact kernel high-water mark.
        let mixed = render_resource_comparison(&ordinary().resource);
        assert!(
            mixed.contains(INCOMPARABLE_NOTE),
            "a sysinfo_sampled_* figure beside a child_max_rss_* figure must be labelled AT THE \
             POINT OF COMPARISON, not in a methods paragraph: {mixed}"
        );

        // THE CONTROL, and it is the half that matters. A note that fires unconditionally is
        // boilerplate a reader learns to skip, which is the same as no note at all.
        let matched = synthetic_aggregate(None, "vm_hwm", MechanismClass::ExactKernelHighWaterMark);
        let matched = render_resource_comparison(&matched.resource);
        assert!(
            !matched.contains(INCOMPARABLE_NOTE),
            "two figures of the SAME mechanism class must NOT carry the incomparability note: \
             {matched}"
        );
        // Both renderings still print BOTH mechanism strings on every row.
        assert!(matched.contains("vm_hwm"));
        assert!(mixed.contains("sysinfo_sampled_10hz"));
        assert!(mixed.contains("child_max_rss_vm_hwm"));
    }

    #[test]
    fn setfit_bench_report_resource_detail_names_every_measurement_scope() {
        let rendered = render_resource_detail(&ordinary().resource);
        for scope in [
            "train peak RSS",
            "inference peak RSS",
            "cold latency (fresh child)",
            "warm latency (median of 10 after 3)",
        ] {
            assert!(
                rendered.contains(scope),
                "a number without its measurement boundary is not a measurement; `{scope}` is \
                 missing"
            );
        }
        // Peak RSS is NEVER rendered as a single unlabelled "peak memory" number.
        assert!(!rendered.contains("peak memory"));
        assert!(rendered.contains(ADAPTER_ONLY_LABEL));
    }

    #[test]
    fn setfit_bench_report_size_table_uses_deployable_and_never_the_adapter_only_figure() {
        let rendered = render_sizes(&ordinary().resource);
        assert!(
            rendered.contains("deployable_total_bytes"),
            "the cross-method size table must NAME the field it uses"
        );
        assert!(rendered.contains("18253611008"), "lora's deployable figure");
        assert!(rendered.contains("94371840"), "setfit's standalone figure");
        assert!(
            !rendered.contains("41943040"),
            "LoRA's ADAPTER-ONLY bytes must never appear in the cross-method size table. An \
             adapter of a few tens of megabytes beside a 90 MiB standalone classifier reads as \
             parity, while the deployable figures differ by the size of a 9B base model: \
             {rendered}"
        );
        // ... and the adapter figure IS shown, labelled, in the per-method detail — so its
        // absence above is a PLACEMENT rule rather than a suppression.
        let detail = render_resource_detail(&ordinary().resource);
        assert!(detail.contains("41943040"));
    }

    #[test]
    fn setfit_bench_report_renders_a_zero_variance_delta_as_a_named_absence() {
        let degenerate = synthetic_aggregate(
            Some(8),
            "sysinfo_sampled_10hz",
            MechanismClass::SampledLowerBound,
        );
        let rendered = render_deltas(&degenerate.deltas);
        assert!(
            rendered.contains(CI_UNAVAILABLE),
            "a degenerate interval is a VISIBLE named state, never a blank cell: {rendered}"
        );
        // The point estimate survives: it is well defined and it is what a reader wants.
        assert!(rendered.contains("0.1000"));
        // The non-degenerate levels still print an interval, so the branch is not a global
        // switch that silenced every CI.
        assert!(rendered.contains("[0.0900, 0.1100]"), "{rendered}");

        // The JSON carries the machine-readable reason and NO numeric bound.
        let payload = ReportPayload::new(&degenerate);
        let value = serde_json::to_value(&payload).expect("payload serializes");
        let ci = &value["detail"]["deltas"][0]["ci95"];
        assert_eq!(ci["null_reason"], ZERO_VARIANCE_NULL_REASON);
        assert!(ci.get("low").is_none(), "no numeric bound may be emitted");
        assert!(ci.get("high").is_none());
    }

    #[test]
    fn setfit_bench_report_json_payload_round_trips_with_per_seed_deltas_and_p_values() {
        let report = ordinary();
        let payload = ReportPayload::new(&report);
        let rendered = serde_json::to_string(&payload).expect("payload serializes");
        let value: serde_json::Value = serde_json::from_str(&rendered).expect("round trips");

        assert_eq!(value["schema"], REPORT_PAYLOAD_SCHEMA);
        assert_eq!(value["contract_id"], CLAIMS_CONTRACT_ID);
        // D-08: p-values may sit in the machine-readable DETAIL and may not appear in claim
        // language. The `detail` key is a structural statement of that boundary.
        let first = &value["detail"]["deltas"][0];
        assert_eq!(
            first["per_seed_deltas"]
                .as_array()
                .expect("per-seed deltas are an array")
                .len(),
            10
        );
        assert!(first["p_value"].is_number(), "p-values live in the detail");
        assert!(first["t_statistic"].is_number());
        // Both size fields travel, so a reader can check the placement rule for themselves.
        let lora = value["detail"]["resource"]
            .as_array()
            .expect("resource is an array")
            .iter()
            .find(|group| group["method"] == "lora")
            .expect("a lora group");
        assert!(lora["artifact_bytes"]["mean"].is_number());
        assert!(lora["deployable_total_bytes"]["mean"].is_number());
        assert!(lora["train_peak_rss_mechanisms"][0].is_string());
    }

    #[test]
    fn setfit_bench_report_quality_table_publishes_macro_f1_beside_f_avg() {
        let rendered = render_quality(&ordinary().quality);
        assert!(rendered.contains("F_avg"));
        assert!(rendered.contains("macro F1"));
        assert!(rendered.contains("setfit"));
        assert!(rendered.contains("lora"));
    }

    // --- The refusal path -----------------------------------------------------------------

    #[test]
    fn setfit_bench_report_refuses_an_incomplete_run_naming_the_cell_and_renders_nothing() {
        use entrenar::train::setfit::bench_row::RunManifest;

        let dir = tempfile::TempDir::new().expect("temp dir");
        // A freshly DECLARED manifest: all eighty cells pending, no rows on disk. That is what
        // a selectively omitted run looks like, taken to its limit.
        std::fs::write(
            dir.path().join("run-manifest.json"),
            RunManifest::declare()
                .to_file_bytes()
                .expect("manifest serializes"),
        )
        .expect("manifest write");

        let error = verified_aggregate(dir.path()).expect_err("an incomplete run is a refusal");
        let rendered = error.to_string();
        assert!(
            rendered.contains("setfit/s8/seed13"),
            "the refusal must NAME the failing cell: {rendered}"
        );
        assert!(
            rendered.contains("--record"),
            "the refusal must name the remedy: {rendered}"
        );
        // ZERO TABLE OUTPUT. The rendering functions are not reached AT ALL, which is a stronger
        // statement than "the output happened not to contain a header".
        assert!(
            !rendered.contains(QUALITY_TABLE_HEADER),
            "a refused run must produce no table: {rendered}"
        );
        assert!(!rendered.contains(SIZE_TABLE_HEADER));
    }

    #[test]
    fn setfit_bench_report_refuses_a_directory_with_no_run_manifest() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let error = verified_aggregate(dir.path()).expect_err("no manifest is a refusal");
        let rendered = error.to_string();
        assert!(rendered.contains("no run manifest"), "{rendered}");
        assert!(
            rendered.contains("cannot report what is missing"),
            "completeness is defined by the manifest and the contract, never by a listing: \
             {rendered}"
        );
    }
}

#[test]
fn setfit_bench_layout_and_filename_grammar_come_from_the_library() {
    // ONE spelling. The 05-10 gate resolves rows, locks and ledgers by these names, and a second
    // copy here could drift — reporting `row_file_missing` for a path this writer never used.
    let source = SETFIT_BENCH_SOURCE
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let delegation = needle(&[
        "pub(crate) use entrenar::train::setfit::bench_gate::",
        "row_file_name",
    ]);
    assert!(
        source.contains(&delegation),
        "row_file_name must be the library's, not a second copy"
    );
    assert_eq!(ROWS_DIR, entrenar::train::setfit::bench_gate::ROWS_DIR);
    assert_eq!(LOCKS_DIR, entrenar::train::setfit::bench_gate::LOCKS_DIR);
    assert_eq!(LEDGER_DIR, entrenar::train::setfit::bench_gate::LEDGER_DIR);
    assert_eq!(
        RUN_MANIFEST_FILE,
        entrenar::train::setfit::bench_gate::RUN_MANIFEST_FILE
    );
}
