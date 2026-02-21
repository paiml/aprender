//! Synthetic data generation for shell completion training
//!
//! Three strategies:
//! 1. CLI Command Templates - realistic dev command patterns
//! 2. Mutation Engine - variations on real history
//! 3. Coverage-Guided - fill gaps in n-gram coverage

use std::collections::{HashMap, HashSet};

/// CLI command template generator
pub struct CommandGenerator {
    templates: Vec<CommandTemplate>,
}

/// A command template with slots for variation
#[derive(Clone)]
struct CommandTemplate {
    base: &'static str,
    variants: Vec<&'static str>,
    flags: Vec<&'static str>,
    args: Vec<&'static str>,
}

include!("synthetic_command_generator.rs");
include!("synthetic_default_command_mutator.rs");
