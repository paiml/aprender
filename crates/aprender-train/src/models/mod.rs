//! Model architecture scaffolds for aprender-train.
//!
//! This module holds frozen, contract-driven architectural configs for
//! models trained by `aprender-train` / `entrenar`. Each submodule pins
//! its dimensions to a YAML contract in `contracts/model-families/` and
//! uses compile-time `const` assertions to refuse to build if the config
//! drifts from the contract.
//!
//! Submodules here are scaffolds only — forward/backward passes and
//! tensor allocation are implemented elsewhere. Nothing is re-exported
//! to the crate root; binding these configs into public API is a
//! follow-up once the scaffolds are reviewed.

#![allow(dead_code)]

pub mod llama_370m;
