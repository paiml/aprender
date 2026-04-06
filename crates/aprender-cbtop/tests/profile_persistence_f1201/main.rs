#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-028: Profile Persistence and Rotation
//!
//! F1201-F1210: Profile persistence falsification tests
//!
//! These tests verify the profile management module for:
//! - Profile save/load operations
//! - Name validation
//! - CLI overlay merging
//! - Export/import functionality

mod additional_tests;
mod persistence_tests;
