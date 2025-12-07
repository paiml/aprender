//! Command-line interface for Monte Carlo simulations

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Monte Carlo simulation tool for financial modeling and risk analysis
#[derive(Parser, Debug)]
#[command(name = "aprender-monte-carlo")]
#[command(author = "paiml")]
#[command(version = "0.1.0")]
#[command(about = "Monte Carlo simulations for finance and business forecasting")]
#[command(long_about = r#"
Monte Carlo simulation tool for:
  - Stock market analysis using S&P 500 historical data
  - Business revenue forecasting with Bayesian models
  - Custom data analysis from CSV files

Examples:
  # Simulate 30-year retirement portfolio using S&P 500 returns
  aprender-monte-carlo sp500 --years 30 --initial 100000 --simulations 10000

  # Analyze custom return data from CSV
  aprender-monte-carlo csv --file returns.csv --initial 50000 --years 10

  # Business revenue forecast
  aprender-monte-carlo revenue --file products.csv --quarters 8
"#)]
pub struct Cli {
    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Table)]
    pub format: OutputFormat,

    /// Random seed for reproducibility
    #[arg(short, long, default_value_t = 42)]
    pub seed: u64,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// The subcommand to run
    #[command(subcommand)]
    pub command: Commands,
}

/// Available commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run simulation using embedded S&P 500 historical data
    Sp500 {
        /// Simulation horizon in years
        #[arg(short, long, default_value_t = 30)]
        years: u32,

        /// Number of simulations to run
        #[arg(short = 'n', long, default_value_t = 10000)]
        simulations: usize,

        /// Initial portfolio value
        #[arg(short, long, default_value_t = 100_000.0)]
        initial: f64,

        /// Annual withdrawal rate (e.g., 0.04 for 4%)
        #[arg(short, long)]
        withdrawal_rate: Option<f64>,

        /// Use inflation-adjusted returns
        #[arg(long)]
        real_returns: bool,
    },

    /// Load and simulate from a CSV file
    Csv {
        /// Path to CSV file with returns data
        #[arg(short, long)]
        file: PathBuf,

        /// Column name containing returns (default: first numeric column)
        #[arg(short, long)]
        column: Option<String>,

        /// Simulation horizon in years
        #[arg(short = 'y', long, default_value_t = 10)]
        years: u32,

        /// Number of simulations to run
        #[arg(short = 'n', long, default_value_t = 10000)]
        simulations: usize,

        /// Initial value
        #[arg(short, long, default_value_t = 100.0)]
        initial: f64,
    },

    /// Business revenue forecasting
    Revenue {
        /// Path to CSV file with product data
        #[arg(short, long)]
        file: PathBuf,

        /// Forecast horizon in quarters
        #[arg(short, long, default_value_t = 4)]
        quarters: u32,

        /// Number of simulations to run
        #[arg(short = 'n', long, default_value_t = 10000)]
        simulations: usize,

        /// Use Bayesian prior from historical data
        #[arg(long)]
        bayesian: bool,
    },

    /// Show embedded S&P 500 data statistics
    Stats {
        /// Show monthly statistics
        #[arg(long)]
        monthly: bool,

        /// Show decade-by-decade breakdown
        #[arg(long)]
        decades: bool,
    },
}

/// Output format options
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable table format
    Table,
    /// JSON format for programmatic use
    Json,
    /// CSV format for spreadsheets
    Csv,
}

impl Cli {
    /// Parse command-line arguments
    #[must_use]
    pub fn parse_args() -> Self {
        Self::parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_sp500() {
        let cli = Cli::try_parse_from([
            "aprender-monte-carlo",
            "sp500",
            "--years",
            "20",
            "--simulations",
            "5000",
        ])
        .expect("Should parse");

        match cli.command {
            Commands::Sp500 {
                years, simulations, ..
            } => {
                assert_eq!(years, 20);
                assert_eq!(simulations, 5000);
            }
            _ => panic!("Expected Sp500 command"),
        }
    }

    #[test]
    fn test_cli_parse_csv() {
        let cli = Cli::try_parse_from([
            "aprender-monte-carlo",
            "csv",
            "--file",
            "test.csv",
            "--initial",
            "50000",
        ])
        .expect("Should parse");

        match cli.command {
            Commands::Csv { initial, file, .. } => {
                assert!((initial - 50000.0).abs() < 0.01);
                assert_eq!(file.to_string_lossy(), "test.csv");
            }
            _ => panic!("Expected Csv command"),
        }
    }

    #[test]
    fn test_output_format() {
        let cli = Cli::try_parse_from(["aprender-monte-carlo", "--format", "json", "sp500"])
            .expect("Should parse");

        assert_eq!(cli.format, OutputFormat::Json);
    }
}
