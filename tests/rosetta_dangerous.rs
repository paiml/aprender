#![allow(clippy::disallowed_methods)]
//! Rosetta Dangerous Tests (ROSETTA-001)
//!
//! "Bold conjectures, and severe attempts to refute them." -- K. Popper
//!
//! These tests seek to REFUTE the conversion matrix, not confirm it.
//! Edge cases that could falsify the entire system.

use std::f32;
use std::path::PathBuf;

use aprender::format::rosetta::{FormatType, RosettaStone};

include!("includes/rosetta_dangerous_part_01.rs");
include!("includes/rosetta_dangerous_part_02.rs");
