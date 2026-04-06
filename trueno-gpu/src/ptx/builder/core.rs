//! Core trait for KernelBuilder state access.
//!
//! This module defines the `KernelBuilderCore` trait which provides
//! internal state access for extension traits.

use super::super::instructions::PtxInstruction;
use super::super::registers::RegisterAllocator;

/// Internal trait for accessing KernelBuilder state.
///
/// Extension traits (arithmetic, memory, control, etc.) require this trait
/// to access the builder's internal state. This pattern allows splitting
/// the large KernelBuilder impl block into focused, testable modules.
pub trait KernelBuilderCore {
    /// Access the register allocator
    fn registers_mut(&mut self) -> &mut RegisterAllocator;

    /// Access the instruction buffer
    fn instructions_mut(&mut self) -> &mut Vec<PtxInstruction>;

    /// Access the label list
    fn labels_mut(&mut self) -> &mut Vec<String>;
}
