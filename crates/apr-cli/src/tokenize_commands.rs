/// Tokenizer training pipeline subcommands (forjar-style plan/apply).
///
/// Thin CLI wrappers around aprender's BPE training infrastructure.
/// Trains a BPE vocabulary from a text corpus for use in model training.
#[derive(Subcommand, Debug)]
pub enum TokenizeCommands {
    /// Validate inputs and estimate tokenizer training time/resources.
    ///
    /// Checks that the input corpus exists, counts lines/bytes, estimates
    /// vocabulary coverage, and reports expected training time. Outputs a
    /// serializable plan manifest (text, JSON, or YAML).
    ///
    /// Analogous to `forjar plan` — shows what will happen before committing.
    Plan {
        /// Path to training corpus (text file, one document per line)
        #[arg(long, value_name = "FILE")]
        data: PathBuf,
        /// Target vocabulary size
        #[arg(long, default_value = "32000")]
        vocab_size: usize,
        /// Tokenizer algorithm: bpe, wordpiece, unigram
        #[arg(long, default_value = "bpe")]
        algorithm: String,
        /// Output directory for trained tokenizer
        #[arg(short, long, default_value = "./tokenizer-output")]
        output: PathBuf,
        /// Output format: text, json, yaml
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Train a tokenizer on the corpus.
    ///
    /// Reads the input corpus, trains a BPE/WordPiece/Unigram tokenizer,
    /// and writes vocab.json + merges.txt to the output directory.
    ///
    /// Analogous to `forjar apply` — commits resources and executes the plan.
    Apply {
        /// Path to training corpus (text file, one document per line)
        #[arg(long, value_name = "FILE")]
        data: PathBuf,
        /// Target vocabulary size
        #[arg(long, default_value = "32000")]
        vocab_size: usize,
        /// Tokenizer algorithm: bpe, wordpiece, unigram
        #[arg(long, default_value = "bpe")]
        algorithm: String,
        /// Output directory for trained tokenizer
        #[arg(short, long, default_value = "./tokenizer-output")]
        output: PathBuf,
        /// Maximum number of lines to read from corpus (0 = all)
        #[arg(long, default_value = "0")]
        max_lines: usize,
    },

    /// Train BPE on a JSONL corpus per contracts/tokenizer-bpe-v1.yaml (MODEL-2).
    ///
    /// Walks `--corpus` (file or directory of `.jsonl` files), extracting the
    /// `content` field from each line, applies `--normalization` (NFC default),
    /// and trains a BPE tokenizer with the target vocab size. Writes
    /// `vocab.json` (token→id) and `merges.txt` (one `a b` pair per line, in
    /// merge order) to `--output`.
    Train {
        /// Path to corpus: a `.jsonl` file or a directory containing `.jsonl` files.
        /// Each line must be a JSON object with a `content` field.
        #[arg(long, value_name = "PATH")]
        corpus: PathBuf,
        /// Target vocabulary size. Default 50_257 matches GPT-2 convention
        /// (50_000 BPE merges + 256 byte-level fallback tokens + 1 sentinel)
        /// and the MODEL-2 albor tokenizer contract (tokenizer-bpe-v1 v1.2.0).
        #[arg(long, default_value = "50257")]
        vocab_size: usize,
        /// Minimum frequency a byte-pair must reach before BPE merges it into
        /// a new vocabulary token (honored by `entrenar::tokenizer::BPETokenizer`
        /// per task #103). Pairs below this threshold are left unmerged —
        /// contract INV-TOK-002 of `contracts/tokenizer-bpe-v1.yaml`.
        #[arg(long, default_value = "2")]
        min_frequency: usize,
        /// Output directory; will contain vocab.json and merges.txt.
        #[arg(long, default_value = "./tokenizer-output")]
        output: PathBuf,
        /// Unicode normalization form applied to each document before training.
        #[arg(long, default_value = "nfc")]
        normalization: String,
    },

    /// Import a HuggingFace tokenizer.json into aprender's two-file
    /// vocab.json + merges.txt layout per
    /// `contracts/apr-cli-tokenize-import-hf-v1.yaml` (§50.4 step 5g.0).
    ///
    /// Reads `<INPUT>` (a HF tokenizer.json with `model.type == "BPE"`),
    /// extracts `model.vocab` → `<OUTPUT>/vocab.json`, `model.merges` →
    /// `<OUTPUT>/merges.txt` (one space-separated merge per line), and
    /// writes `<OUTPUT>/manifest.json` with extraction provenance
    /// (source path, sha256, vocab_size, merges_count, timestamp).
    ///
    /// Non-BPE inputs (Unigram, WordPiece) are rejected fail-fast with a
    /// clear error citing the contract id.
    ///
    /// Unblocks fine-tuning from public HF checkpoints (Qwen2.5/Llama2/
    /// Mistral) which distribute as a single tokenizer.json. The output
    /// dir is consumable by `apr tokenize encode-corpus --tokenizer <DIR>`
    /// and `apr pretrain --tokenizer <DIR>` without modification.
    ImportHf {
        /// Path to input HuggingFace tokenizer.json (BPE model required).
        #[arg(long, value_name = "FILE")]
        input: PathBuf,
        /// Output directory; will contain vocab.json + merges.txt + manifest.json.
        #[arg(long, value_name = "DIR")]
        output: PathBuf,
        /// Include `added_tokens` in vocab.json (default: BPE state machine only).
        /// Use this when the downstream consumer needs special tokens (e.g.,
        /// `<|im_start|>`, `<|endoftext|>`) materialized in vocab.json itself.
        #[arg(long, default_value_t = false)]
        include_added_tokens: bool,
    },

    /// Encode a JSONL corpus into `.bin` shards per contracts/pretokenize-bin-v1.yaml.
    ///
    /// Loads a trained BPE tokenizer (vocab.json + merges.txt) from `--tokenizer`,
    /// reads `--corpus` (file or directory of `.jsonl` files), encodes the
    /// `--content-field` of each line to u32 tokens, and writes
    /// `shard-NNNN.bin` files (flat little-endian u32 streams) into `--output`.
    /// The output format is precisely what `ShardBatchIter` (aprender-train)
    /// expects at MODEL-2 pretrain read time.
    ///
    /// Root-cause fix for the pretokenize-to-bin gap documented in
    /// memory/project_shard_reader_bin_format.md — replaces a Python shim
    /// that was flagged as MUDA on 2026-04-19.
    #[cfg(feature = "training")]
    EncodeCorpus {
        /// Path to JSONL corpus file or directory of `.jsonl` files.
        #[arg(long, value_name = "PATH")]
        corpus: PathBuf,
        /// Directory containing vocab.json + merges.txt from `apr tokenize train`.
        #[arg(long, value_name = "DIR")]
        tokenizer: PathBuf,
        /// Output directory for shard-NNNN.bin + manifest.json.
        #[arg(long, value_name = "DIR")]
        output: PathBuf,
        /// Target tokens per shard (shard closes once this limit is reached).
        #[arg(long, default_value = "10000000")]
        shard_tokens: usize,
        /// JSONL field to encode (default: `content`).
        #[arg(long, default_value = "content")]
        content_field: String,
        /// Unicode normalization (must match tokenizer training).
        #[arg(long, default_value = "nfc")]
        normalization: String,
        /// EOS insertion policy: none|between|after.
        #[arg(long, default_value = "between")]
        eos_policy: String,
        /// Number of rayon workers for per-document BPE encoding.
        ///
        /// Defaults to `std::thread::available_parallelism()` (logical CPU count).
        /// Set to `1` to force the single-threaded byte-identical legacy path.
        /// Set to a fixed N to bound memory or share the host with other jobs.
        ///
        /// Output shard order is preserved: chunked encoding keeps original
        /// document order regardless of worker count (issue #1547,
        /// contracts/apr-tokenize-parallel-bpe-v1.yaml `parallel_correctness`).
        #[arg(long, value_name = "N")]
        num_workers: Option<usize>,
        /// Suppress per-document progress emission to stderr (issue #1547,
        /// contract v1.2.0). Default: emit a `[progress] doc=N/T tokens=K
        /// rate=X.X docs/s eta=...` line every `--progress-interval-docs`
        /// docs OR `--progress-interval-seconds` seconds (whichever fires
        /// first). Useful for CI / log-scraping callers that prefer silence.
        #[arg(long, default_value_t = false)]
        quiet: bool,
        /// Emit a progress line at most every N docs (default 1000). Pair
        /// with `--progress-interval-seconds` — whichever bound is reached
        /// first triggers emission. Issue #1547 contract v1.2.0.
        #[arg(long, value_name = "N", default_value_t = 1000)]
        progress_interval_docs: u64,
        /// Emit a progress line at most every S seconds (default 60). Pair
        /// with `--progress-interval-docs` — whichever bound is reached
        /// first triggers emission. Issue #1547 contract v1.2.0.
        #[arg(long, value_name = "S", default_value_t = 60)]
        progress_interval_seconds: u64,
        /// Pre-flight only: estimate total tokens / shards / wall time
        /// without writing any output. Reads `--estimate-sample-docs`
        /// (default 1000), encodes them, observes (tokens, wall-time-
        /// per-doc), and extrapolates against the total document count.
        /// Emits `[estimate]` lines on stderr; no shards or manifest are
        /// written. Operator pre-flight gate before multi-day encode
        /// runs (issue #1547 contract v1.3.0).
        #[arg(long, default_value_t = false)]
        estimate_only: bool,
        /// Number of documents to sample for `--estimate-only`
        /// extrapolation (default: 1000). Larger samples → tighter
        /// per-doc rate estimate but longer pre-flight wall.
        #[arg(long, value_name = "N", default_value_t = 1000)]
        estimate_sample_docs: u64,
    },

    /// Reconstruct manifest.json from existing shard-NNNN.bin files.
    ///
    /// `apr tokenize encode-corpus` writes manifest.json only on clean
    /// process exit. If the encoder is killed (operator SIGINT, OOM,
    /// crash, power loss) AFTER all shards flush but BEFORE manifest
    /// write, the corpus on disk is consumable by `ShardBatchIter` but
    /// has no provenance file for ship audit / dashboards.
    ///
    /// `repair-manifest` is the cheap recovery path: it scans
    /// `<OUTPUT>/shard-*.bin`, computes shard_count + total_tokens
    /// from file sizes (each shard is a flat little-endian u32 stream;
    /// tokens = file_size / 4), and writes a schema-conforming
    /// `manifest.json`. Idempotent: runs twice are byte-identical
    /// modulo `repaired_at` timestamp.
    ///
    /// Contract: `contracts/apr-tokenize-repair-manifest-v1.yaml`.
    /// Motivating instance: SHIP-TWO §56 5g.1 corpus (228 shards
    /// flushed, manifest missing).
    #[cfg(feature = "training")]
    RepairManifest {
        /// Output directory containing shard-NNNN.bin files.
        /// `manifest.json` will be written into this directory.
        #[arg(long, value_name = "DIR")]
        output: PathBuf,
        /// Optional tokenizer directory; when provided, `vocab.json`
        /// is read for the manifest's `vocab_size` field. Without it,
        /// `vocab_size` is recorded as `null` (provenance-incomplete
        /// but otherwise valid).
        #[arg(long, value_name = "DIR")]
        tokenizer: Option<PathBuf>,
        /// Emit the manifest body as JSON to stdout (in addition to
        /// writing to disk).
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}
