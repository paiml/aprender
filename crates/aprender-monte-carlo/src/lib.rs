//! # aprender-monte-carlo
//!
//! CLI tool and utility library for Monte Carlo simulations.
//!
//! This crate provides:
//! - **CLI Interface**: Run simulations from the command line
//! - **CSV Data Loading**: Load historical data from CSV files
//! - **S&P 500 Data**: Embedded historical S&P 500 data (1928-present)
//! - **Business Revenue Modeling**: Product revenue forecasting
//!
//! ## Core Monte Carlo functionality
//!
//! The core simulation engine, risk metrics, and models are provided by the
//! main `aprender` crate's `monte_carlo` module. This crate re-exports those
//! for convenience and adds CLI-specific functionality.
//!
//! ## Quick Start (Library)
//!
//! ```rust,ignore
//! use aprender_monte_carlo::prelude::*;
//!
//! // Load S&P 500 data
//! let sp500 = Sp500Data::load();
//!
//! // Run bootstrap simulation
//! let model = EmpiricalBootstrap::new(100.0, sp500.monthly_returns());
//! let engine = MonteCarloEngine::reproducible(42).with_n_simulations(10_000);
//! let result = engine.simulate(&model, &TimeHorizon::years(10));
//!
//! // Generate risk report
//! let report = RiskReport::from_paths(&result.paths, 0.02)?;
//! println!("{}", report.summary());
//! ```
//!
//! ## Quick Start (CLI)
//!
//! ```bash
//! # Run S&P 500 simulation
//! aprender-monte-carlo sp500 --years 30 --simulations 10000
//!
//! # Load custom CSV data
//! aprender-monte-carlo csv --file returns.csv --initial 100000
//!
//! # Business revenue forecast
//! aprender-monte-carlo revenue --file products.csv --quarters 4
//! ```

#![forbid(unsafe_code)]
#![warn(
    missing_docs,
    rust_2018_idioms,
    unreachable_pub,
    unused_results,
    clippy::all,
    clippy::pedantic
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

pub mod cli;
pub mod data;
pub mod models;

// Re-export core Monte Carlo functionality from aprender
pub use aprender::monte_carlo::{
    engine, error, models as core_models, risk, MonteCarloError, Result,
};

/// Prelude for convenient imports
pub mod prelude {
    // Core engine types from aprender
    pub use aprender::monte_carlo::prelude::*;

    // CLI-specific types
    pub use crate::cli::{Cli, Commands, OutputFormat};
    pub use crate::data::{CsvLoader, Sp500Data};
    pub use crate::models::BayesianRevenueModel;
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_prelude_imports() {
        // Verify core Monte Carlo types are accessible
        let _rng = MonteCarloRng::new(42);
        let _engine = MonteCarloEngine::reproducible(42);
        let _horizon = TimeHorizon::years(1);
    }

    #[test]
    fn test_sp500_data_available() {
        // Verify S&P 500 data is embedded
        let sp500 = Sp500Data::load();
        assert!(sp500.len() > 1000, "Should have many months of data");
    }
}
