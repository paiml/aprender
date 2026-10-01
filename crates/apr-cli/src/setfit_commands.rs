
/// Native SetFit classifier TRAINING (`setfit-apr-v1` artifacts).
///
/// # This namespace is training-only, and that is a locked decision (D-06)
///
/// `predict`, `eval` and `inspect` for a SetFit artifact are the GENERIC `apr`
/// commands — they take a model path like every other model path, and they work on
/// `setfit-apr-v1` because the artifact is an APR. Adding `apr setfit predict` here
/// would give a user two spellings of one operation and would make the SetFit
/// artifact look like a format that needs its own tooling, which is the opposite of
/// what shipping it as APR is for.
///
/// Training gets a namespace of its own because it is the one operation with no
/// generic form: it consumes Phase 2's prepared dataset directory and selection
/// manifest, which no other `apr` command knows about.
#[derive(Subcommand, Debug)]
pub enum SetfitCommands {
    /// Train a SetFit classifier from Phase 2 artifacts and write a verified APR
    ///
    /// Configuration is FILE-FIRST: the twelve training knobs come from --config,
    /// and only --seed and --device may override it. That is what makes a run
    /// reproducible from the reported configuration alone — a knob that exists only
    /// as a flag has no place in the artifact's recorded provenance.
    ///
    /// The dataset directory and the selection manifest are consumed exactly as
    /// `apr data tweet-eval-stance` and `apr data select` wrote them. This command
    /// introduces no new on-disk format.
    Train {
        /// Training configuration: `.toml` or `.json`, carrying all twelve knobs
        ///
        /// Deserialization IS validation — the file is parsed straight through the
        /// library's single validating constructor, so an unknown key or an invalid
        /// value is refused BEFORE anything is read from disk. There is no partial
        /// config and no per-knob default: an absent `root_seed` is an error, not a
        /// silently chosen number nobody would be able to reproduce.
        #[arg(long, value_name = "FILE")]
        config: PathBuf,

        /// Attested benchmark directory, as written by `apr data tweet-eval-stance`
        ///
        /// The canonical train/validation/test JSONL plus benchmark-manifest.json.
        /// A compatibility-profile, mixed, stale or forged directory is refused at
        /// the attested boundary before a single row is read into training.
        #[arg(long, value_name = "DIR")]
        data: PathBuf,

        /// The selection-manifest.json written by `apr data select`
        ///
        /// Strictly replayed against --data before training: the manifest names the
        /// rows, and a manifest whose digest, provenance or ordered list does not
        /// survive replay against THIS dataset is refused.
        #[arg(long, value_name = "FILE")]
        selection: PathBuf,

        /// Model checkout: tokenizer.json plus the encoder weights
        ///
        /// OFFLINE PREREQUISITE. This command never downloads. Obtain the checkout
        /// separately (for example `batuta hf pull`) and point --model-dir at the
        /// directory.
        ///
        /// Which checks apply depends on what the directory DECLARES. A config.json
        /// declaring the pinned all-MiniLM-L6-v2 architecture is held to the full
        /// pin: the tokenizer bytes must hash to the pinned digest, so a near-miss
        /// checkout is refused rather than silently producing a model whose
        /// tokenizer and encoder disagree. Any other BERT checkout (AlephBERT,
        /// DictaBERT, multilingual BERT) is validated structurally — architecture,
        /// activation, position scheme, dropout, dimension bounds, and the
        /// tokenizer/vocabulary pairing — but there is no digest to pin it to, so
        /// its provenance is recorded as `general-bert`.
        #[arg(long = "model-dir", value_name = "DIR")]
        model_dir: PathBuf,

        /// Where to write the `setfit-apr-v1` artifact
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,

        /// Override the config file's `root_seed`
        ///
        /// The override is merged through the library's public validated door and
        /// the WHOLE merged configuration is revalidated, so an override cannot
        /// smuggle past a check the file had to pass. The merged value is what the
        /// run records; the file's value is not preserved anywhere.
        #[arg(long, value_name = "SEED")]
        seed: Option<u64>,

        /// Override the config file's `device` (`cpu`, `cuda`, `cuda:N`, `auto`)
        ///
        /// An explicitly requested device that this host cannot provide is a HARD
        /// failure with a nonzero exit code. There is no silent fallback to CPU:
        /// a benchmark number produced on a device nobody asked for is worse than
        /// no number at all.
        #[arg(long, value_name = "SPEC")]
        device: Option<String>,

        /// Replace an existing --output file
        #[arg(long)]
        force: bool,

        /// Validate the request and the Phase 2 inputs, then stop
        ///
        /// Reports the MERGED resolved configuration and exits without loading
        /// --model-dir, without training and without writing anything. The encoder
        /// load is deliberately outside the dry run: it is a multi-hundred-megabyte
        /// read, and a pre-flight that costs as much as the thing it precedes is not
        /// a pre-flight.
        #[arg(long = "dry-run")]
        dry_run: bool,
    },

    /// The EVAL-03 benchmark matrix: one cell in, one digest-committed row out
    ///
    /// A namespace rather than a flag on `train` because a benchmark cell is not a
    /// training run with extra reporting: it trains, RELOADS the written artifact,
    /// measures the contracted resource protocol against the reloaded model, takes
    /// canonical test access through the Phase 3 lock chain, and emits a row whose
    /// digest the 05-10 gate recomputes. Contract: `setfit-benchmark-claims-v1`.
    Bench {
        #[command(subcommand)]
        command: BenchCommands,
    },
}

/// `apr setfit bench` — the benchmark matrix commands (D-13).
#[derive(Subcommand, Debug)]
pub enum BenchCommands {
    /// Execute ONE method/shot/seed cell, or ingest a row another host executed
    ///
    /// # Three mutually exclusive modes, and clap enforces the exclusion
    ///
    /// 1. EXECUTION (`--method --shots --seed --data --selection --bench-dir ...`):
    ///    train, reload, measure, evaluate, emit one row file plus a run-manifest
    ///    update. One process per cell — see `scripts/run_bench_cells.sh`.
    /// 2. RECORD (`--record <ROW_FILE> --bench-dir <DIR>`): ingest a row file that a
    ///    DIFFERENT host executed. The bytes are verified — digest, schema, cell
    ///    identity and filename agreement — and NOTHING is executed. This is the
    ///    GPU-host transport path (D-09).
    /// 3. COLD PROBE (`--cold-probe <ARTIFACT> --probe-text <FILE>`): the dedicated
    ///    fresh child the resource protocol requires. It loads the artifact, runs
    ///    exactly one classify, prints one `COLD_LATENCY_MS=<f64>` line and exits.
    ///    Nobody types this: `bench run` spawns it under `/usr/bin/time` and reads
    ///    the child's true kernel high-water mark.
    ///
    /// The cell key is CONTRACTED. `--shots` must be one of 8/16/32/64 and `--seed`
    /// one of the ten contracted seeds; 42 is deliberately NOT one of them, so a tool
    /// that defaults a seed to 42 is refused rather than silently sampling outside the
    /// protocol it claims to follow.
    Run {
        /// `setfit` or `lora`
        #[arg(long, value_name = "METHOD", conflicts_with_all = ["record", "cold_probe"])]
        method: Option<String>,

        /// Examples per class: 8, 16, 32 or 64
        #[arg(long, value_name = "SHOTS", conflicts_with_all = ["record", "cold_probe"])]
        shots: Option<u32>,

        /// One of the ten contracted seeds (13 17 23 29 31 37 41 43 47 53)
        #[arg(long, value_name = "SEED", conflicts_with_all = ["record", "cold_probe"])]
        seed: Option<u32>,

        /// Attested benchmark directory, as written by `apr data tweet-eval-stance`
        #[arg(long, value_name = "DIR", conflicts_with_all = ["record", "cold_probe"])]
        data: Option<PathBuf>,

        /// The selection-manifest.json this cell's rows come from — the PAIRING KEY
        ///
        /// EVAL-02's identical-sampled-ID guarantee is this file: both methods consume
        /// the same manifest for a given (shots, seed), and the row records its hash.
        #[arg(long, value_name = "FILE", conflicts_with_all = ["record", "cold_probe"])]
        selection: Option<PathBuf>,

        /// Where rows, locks, ledgers and the run manifest live
        ///
        /// Required by BOTH the execution and the record modes: the run manifest is
        /// the pre-declared expectation set that makes an omitted cell visible.
        #[arg(long = "bench-dir", value_name = "DIR", conflicts_with = "cold_probe")]
        bench_dir: Option<PathBuf>,

        /// Pinned all-MiniLM-L6-v2 checkout (the `setfit` method's encoder)
        ///
        /// OFFLINE PREREQUISITE — this command never downloads.
        #[arg(long = "model-dir", value_name = "DIR", conflicts_with_all = ["record", "cold_probe"])]
        model_dir: Option<PathBuf>,

        /// The base model the LoRA adapter applies to (the `lora` method)
        ///
        /// A separate flag from --model-dir because it is a different KIND of thing: the
        /// SetFit encoder is a checkout directory, and this is one `.apr` file whose bytes
        /// the row records as `base_model_bytes`. An adapter alone is not deployable, and
        /// `deployable_total_bytes` is base + adapter — so the base has to be nameable.
        #[arg(long = "base-model", value_name = "FILE", conflicts_with_all = ["record", "cold_probe"])]
        base_model: Option<PathBuf>,

        /// Optional training configuration, file-first per the house rule
        ///
        /// Absent means the FROZEN published defaults for the method, which is what a
        /// benchmark cell should use: a per-cell knob is a tuning surface, and tuning
        /// on the benchmark is the thing `no_selection_attestation` attests did not
        /// happen.
        #[arg(long, value_name = "FILE", conflicts_with_all = ["record", "cold_probe"])]
        config: Option<PathBuf>,

        /// Replace an existing row file (or ledger line) for this cell
        #[arg(long)]
        force: bool,

        /// Ingest a row file executed elsewhere — verification only, NO execution
        #[arg(long, value_name = "ROW_FILE", conflicts_with = "cold_probe")]
        record: Option<PathBuf>,

        /// The dedicated cold-measurement child (spawned by `bench run`, not typed)
        ///
        /// Hidden from `-h` and listed in `--help`: it is machinery, not a user
        /// surface, but a reader auditing the resource protocol must be able to find
        /// it without reading the source.
        #[arg(long = "cold-probe", value_name = "ARTIFACT", hide_short_help = true)]
        cold_probe: Option<PathBuf>,

        /// The LoRA base model, when the cold probe is measuring a LoRA cell
        ///
        /// Present: the probe reloads base + adapter through `ClassifyPipeline`.
        /// Absent: the probe reloads a standalone `setfit-apr-v1`.
        #[arg(
            long = "cold-probe-base",
            value_name = "BASE",
            hide_short_help = true,
            requires = "cold_probe"
        )]
        cold_probe_base: Option<PathBuf>,

        /// The single text the cold probe classifies
        #[arg(
            long = "probe-text",
            value_name = "FILE",
            hide_short_help = true,
            requires = "cold_probe"
        )]
        probe_text: Option<PathBuf>,
    },

    /// Verify a whole benchmark directory and render the claims report
    ///
    /// # Verified data only — there is no partial-data mode
    ///
    /// The report refuses, naming the cell (and, for provenance, the FILE whose bytes
    /// disagreed), when any expected cell is missing or `pending`, when a row's digest
    /// does not match, when a row's payload disagrees with the slot it was filed under,
    /// when a (shots, seed) pair's two rows consumed DIFFERENT selection manifests, when
    /// a committed lock record or candidate ledger does not hash to what its row claims,
    /// or when a LoRA cell's no-selection attestation does not hold. Nothing is rendered
    /// on any of those paths: a partial table is a number, and a number is what a reader
    /// takes away.
    ///
    /// # Estimation-first
    ///
    /// Point estimates, dispersion and paired 95% CIs. No binary verdict is printed;
    /// p-values live in the `--json` detail only (D-08), because a verdict is precisely
    /// where few-shot seed sensitivity hides — rankings that reverse across seeds become
    /// one word.
    Report {
        /// Where rows, locks, ledgers and the run manifest live
        #[arg(long = "bench-dir", value_name = "DIR")]
        bench_dir: PathBuf,

        /// Write the machine-readable detail to this file as well
        ///
        /// The same payload `--json` prints: per-seed deltas, p-values, every mechanism
        /// string and both size fields.
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
    },

    /// Verify ONE declared cell's own evidence - a diagnostic door, never a report
    ///
    /// # Which steps it applies
    ///
    /// `verify_run`'s steps 1 + 4 + 6, over one cell and in that order: the manifest's
    /// own digest; step 3's per-entry rule applied to THAT entry only; the row's file,
    /// schema, envelope digest, manifest-digest agreement and slot agreement; then
    /// provenance recomputed from the committed lock (and, for a second method, ledger)
    /// bytes.
    ///
    /// # Which steps it deliberately EXCLUDES, and why
    ///
    /// NOT step 2 (expectation-set equality), NOT step 3's sweep over every entry, NOT
    /// step 5 (pairing) and NOT step 7. At pilot time the manifest declares 40 cells
    /// with 39 still `pending` - which is precisely the state step 3's sweep refuses on
    /// - so a door that inherited the set-level checks could never pass on the cell it
    /// exists to check. `bench report` over a copy holding one row likewise refuses at
    /// completeness BEFORE the row loop, so the pilot row's own bytes are never read at
    /// all. This door reads them.
    ///
    /// # It emits NO statistic
    ///
    /// No mean, no dispersion, no interval, no aggregate. `bench report` is the only
    /// door that publishes numbers; a per-cell door that printed statistics would be a
    /// partial-data report under another name. It therefore CONFLICTS with the
    /// report-shaped output flag rather than trusting a reader not to combine them.
    VerifyCell {
        /// Where rows, locks, ledgers and the run manifest live
        #[arg(long = "bench-dir", value_name = "DIR")]
        bench_dir: PathBuf,

        /// `setfit` or `lora`
        ///
        /// This is the ROW-VALIDITY vocabulary, which admits both. Whether a cell is in
        /// the ACTIVE expectation set is a different question, and asking for one that
        /// is not is refused by the gate rather than by clap.
        #[arg(long, value_name = "METHOD", default_value = "setfit")]
        method: String,

        /// Examples per class: 8, 16, 32 or 64
        #[arg(long, value_name = "SHOTS")]
        shots: u32,

        /// One of the ten contracted seeds (13 17 23 29 31 37 41 43 47 53)
        #[arg(long, value_name = "SEED")]
        seed: u32,

        /// REFUSED. This door emits no machine-readable report payload
        ///
        /// Declared only so that a report-shaped invocation is REJECTED with an
        /// explanation rather than silently accepted: a diagnostic door that took a
        /// report flag would invite being read as a partial report.
        #[arg(long, value_name = "FILE", conflicts_with = "bench_dir", hide = true)]
        out: Option<PathBuf>,
    },
}
