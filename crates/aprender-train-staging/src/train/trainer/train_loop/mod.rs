//! Multi-epoch training loops
//!
//! This module provides the main training loop implementations for the `Trainer`:
//! - `basic`: Training loop without validation (`Trainer::train`)
//! - `validation`: Training loop with validation support (`Trainer::train_with_val`)

mod basic;
mod validation;

#[cfg(test)]
mod tests;
