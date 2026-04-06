//! Static Analysis Passes
//!
//! This module contains the static analysis passes for PTX code:
//! - Type Checker: Validates register types match operations
//! - Control Flow Analyzer: Constructs CFG, validates barrier synchronization
//! - Data Flow Analyzer: Tracks value propagation, detects "loaded value" bug
//! - Address Space Validator: Validates correct address space usage

mod address_space;
mod control_flow;
mod data_flow;
mod type_checker;

pub use address_space::{AddressSpaceValidator, GenericSharedBug};
pub use control_flow::{BarrierViolation, CfgNode, ControlFlowAnalyzer, ControlFlowGraph};
pub use data_flow::{
    ComputedAddrFromLoadedBug, DataFlowAnalyzer, LoadedValueBug, UsePoint, ValueSource,
};
pub use type_checker::TypeChecker;
