
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
}
