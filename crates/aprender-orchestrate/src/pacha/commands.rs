//! Pacha CLI Command Definitions
//!
//! This module defines the `PachaCommand` enum which represents all CLI commands
//! available in the Pacha model registry interface.

use clap::Subcommand;

// ============================================================================
// PACHA-CLI-001: Command Definitions
// ============================================================================

/// Pacha model registry commands
#[derive(Subcommand, Debug, Clone)]
pub enum PachaCommand {
    /// Pull a model from registry or HuggingFace
    Pull {
        /// Model reference (e.g., llama3, llama3:8b, hf://meta-llama/Llama-3-8B)
        #[arg(value_name = "MODEL")]
        model: String,

        /// Force re-download even if cached
        #[arg(short, long)]
        force: bool,

        /// Specific quantization (e.g., q4_k_m, q8_0, f16)
        #[arg(short, long)]
        quant: Option<String>,
    },

    /// List cached models
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,

        /// Output format (table, json)
        #[arg(short, long, default_value = "table")]
        format: String,
    },

    /// Remove a model from cache
    Rm {
        /// Model reference to remove
        #[arg(value_name = "MODEL")]
        model: String,

        /// Remove all versions of this model
        #[arg(short, long)]
        all: bool,

        /// Skip confirmation
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Show model information
    Show {
        /// Model reference
        #[arg(value_name = "MODEL")]
        model: String,

        /// Show full metadata (including tensors)
        #[arg(short, long)]
        full: bool,
    },

    /// Search for models
    Search {
        /// Search query
        #[arg(value_name = "QUERY")]
        query: String,

        /// Limit results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Show available aliases
    Aliases {
        /// Filter by pattern
        #[arg(value_name = "PATTERN")]
        pattern: Option<String>,
    },

    /// Add a custom alias
    Alias {
        /// Alias name
        #[arg(value_name = "NAME")]
        name: String,

        /// Target URI
        #[arg(value_name = "TARGET")]
        target: String,
    },

    /// Show cache statistics
    Stats,

    /// Clean up old/unused models
    Prune {
        /// Days since last access (default: 30)
        #[arg(short, long, default_value = "30")]
        days: u64,

        /// Dry run (show what would be removed)
        #[arg(short = 'n', long)]
        dry_run: bool,
    },

    /// Pin a model (prevent eviction)
    Pin {
        /// Model reference to pin
        #[arg(value_name = "MODEL")]
        model: String,
    },

    /// Unpin a model
    Unpin {
        /// Model reference to unpin
        #[arg(value_name = "MODEL")]
        model: String,
    },

    /// Run interactive chat with a model
    Run {
        /// Model reference
        #[arg(value_name = "MODEL")]
        model: String,

        /// System prompt
        #[arg(short, long)]
        system: Option<String>,

        /// Load parameters from Modelfile
        #[arg(short, long, value_name = "FILE")]
        modelfile: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Max tokens to generate
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Context window size
        #[arg(long, default_value = "4096")]
        context: usize,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Generate a signing key pair
    Keygen {
        /// Output path for private key (default: ~/.pacha/signing-key.pem)
        #[arg(short, long)]
        output: Option<String>,

        /// Signer identity (e.g., email)
        #[arg(short, long)]
        identity: Option<String>,

        /// Force overwrite existing key
        #[arg(short, long)]
        force: bool,
    },

    /// Sign a model file
    Sign {
        /// Model file or reference to sign
        #[arg(value_name = "MODEL")]
        model: String,

        /// Path to signing key (default: ~/.pacha/signing-key.pem)
        #[arg(short, long)]
        key: Option<String>,

        /// Output signature file (default: <model>.sig)
        #[arg(short, long)]
        output: Option<String>,

        /// Signer identity
        #[arg(short, long)]
        identity: Option<String>,
    },

    /// Verify a model signature
    Verify {
        /// Model file or reference to verify
        #[arg(value_name = "MODEL")]
        model: String,

        /// Signature file (default: <model>.sig)
        #[arg(short, long)]
        signature: Option<String>,

        /// Expected signer public key (hex)
        #[arg(short, long)]
        key: Option<String>,
    },

    /// Encrypt a model file for distribution
    Encrypt {
        /// Model file to encrypt
        #[arg(value_name = "MODEL")]
        model: String,

        /// Output file (default: <model>.enc)
        #[arg(short, long)]
        output: Option<String>,

        /// Read password from environment variable
        #[arg(long, value_name = "VAR")]
        password_env: Option<String>,
    },

    /// Decrypt an encrypted model file
    Decrypt {
        /// Encrypted model file
        #[arg(value_name = "FILE")]
        file: String,

        /// Output file (default: original name without .enc)
        #[arg(short, long)]
        output: Option<String>,

        /// Read password from environment variable
        #[arg(long, value_name = "VAR")]
        password_env: Option<String>,
    },
}
