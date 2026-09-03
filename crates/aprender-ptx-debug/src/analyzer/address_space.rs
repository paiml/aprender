//! Address Space Validator - validates correct address space usage

use crate::bugs::Severity;
use crate::parser::types::{Modifier, Opcode};
use crate::parser::{Instruction, Operand, PtxModule, SourceLocation, Statement};
use std::collections::HashSet;

/// Defect: Generic addressing of shared memory
#[derive(Debug, Clone)]
pub struct GenericSharedBug {
    /// Source location
    pub location: SourceLocation,
    /// Instruction that triggered the bug
    pub instruction: Instruction,
    /// Severity
    pub severity: Severity,
    /// Fix suggestion
    pub fix: String,
}

/// Address Space Validator
pub struct AddressSpaceValidator {
    /// Registers holding cvta.shared results (generic shared addresses)
    shared_base_regs: HashSet<String>,
}

impl AddressSpaceValidator {
    /// Create a new address space validator
    pub fn new() -> Self {
        Self {
            shared_base_regs: HashSet::new(),
        }
    }

    /// Detect generic addressing of shared memory (F021)
    ///
    /// WRONG: cvta.shared.u64 %rd, smem; ld.u32 [%rd]
    /// RIGHT: ld.shared.u32 [smem_offset]
    pub fn detect_generic_shared_access(&mut self, module: &PtxModule) -> Vec<GenericSharedBug> {
        let mut bugs = Vec::new();

        for kernel in &module.kernels {
            self.shared_base_regs.clear();

            for stmt in &kernel.body {
                if let Statement::Instruction(instr) = stmt {
                    self.scan_generic_shared_instruction(instr, &mut bugs);
                }
            }
        }

        bugs
    }

    /// Track a `cvta.shared` destination, then flag a generic ld/st that uses it.
    fn scan_generic_shared_instruction(
        &mut self,
        instr: &Instruction,
        bugs: &mut Vec<GenericSharedBug>,
    ) {
        // Track cvta.shared destinations
        if instr.opcode == Opcode::Cvta && self.has_shared_modifier(instr) {
            if let Some(Operand::Register(dest)) = instr.operands.first() {
                self.shared_base_regs.insert(dest.clone());
            }
        }

        // Detect generic ld/st using tracked registers
        if (instr.opcode != Opcode::Ld && instr.opcode != Opcode::St)
            || self.has_space_modifier(instr)
        {
            return;
        }

        // Check if address operand uses a generic shared register
        let addr_operand = if instr.opcode == Opcode::Ld {
            instr.operands.get(1)
        } else {
            instr.operands.first()
        };

        if let Some(operand) = addr_operand {
            if self.uses_generic_shared_reg(operand) {
                bugs.push(GenericSharedBug {
                    location: instr.location.clone(),
                    instruction: instr.clone(),
                    severity: Severity::Critical,
                    fix: "Use ld.shared with 32-bit offset instead".into(),
                });
            }
        }
    }

    /// Detect shared memory using 64-bit addresses (F022)
    pub fn detect_shared_mem_u64(&self, module: &PtxModule) -> Vec<GenericSharedBug> {
        let mut bugs = Vec::new();

        for kernel in &module.kernels {
            for stmt in &kernel.body {
                if let Statement::Instruction(instr) = stmt {
                    // Check for ld.shared.u64 or st.shared.u64 with 64-bit address
                    if self.has_shared_modifier(instr) && self.has_u64_modifier(instr) {
                        bugs.push(GenericSharedBug {
                            location: instr.location.clone(),
                            instruction: instr.clone(),
                            severity: Severity::High,
                            fix: "Use 32-bit offset for shared memory addressing".into(),
                        });
                    }
                }
            }
        }

        bugs
    }

    /// Detect cvta.shared inside loops (F083)
    pub fn detect_loop_cvta_shared(&self, module: &PtxModule) -> Vec<GenericSharedBug> {
        let mut bugs = Vec::new();

        for kernel in &module.kernels {
            let mut in_loop = false;
            let mut loop_start_labels = HashSet::new();

            for stmt in &kernel.body {
                match stmt {
                    Statement::Label(label) => {
                        // Simple heuristic: label containing "loop" starts a loop
                        if label.to_lowercase().contains("loop") {
                            in_loop = true;
                            loop_start_labels.insert(label.clone());
                        }
                    }
                    Statement::Instruction(instr) => {
                        self.scan_loop_cvta_instruction(
                            instr,
                            &mut in_loop,
                            &loop_start_labels,
                            &mut bugs,
                        );
                    }
                    _ => {}
                }
            }
        }

        bugs
    }

    /// Clear the loop flag on a backward branch, then flag `cvta.shared` seen inside a loop.
    fn scan_loop_cvta_instruction(
        &self,
        instr: &Instruction,
        in_loop: &mut bool,
        loop_start_labels: &HashSet<String>,
        bugs: &mut Vec<GenericSharedBug>,
    ) {
        // Check for backward branch (loop end)
        if instr.opcode == Opcode::Bra {
            for operand in &instr.operands {
                if let Operand::Label(target) = operand {
                    if loop_start_labels.contains(target) {
                        *in_loop = false;
                    }
                }
            }
        }

        // Detect cvta.shared inside loop
        if *in_loop && instr.opcode == Opcode::Cvta && self.has_shared_modifier(instr) {
            bugs.push(GenericSharedBug {
                location: instr.location.clone(),
                instruction: instr.clone(),
                severity: Severity::High,
                fix: "Move cvta.shared outside loop to reduce register pressure".into(),
            });
        }
    }

    fn has_shared_modifier(&self, instr: &Instruction) -> bool {
        instr
            .modifiers
            .iter()
            .any(|m| matches!(m, Modifier::Shared))
    }

    fn has_space_modifier(&self, instr: &Instruction) -> bool {
        instr
            .modifiers
            .iter()
            .any(|m| m.as_address_space().is_some())
    }

    fn has_u64_modifier(&self, instr: &Instruction) -> bool {
        instr
            .modifiers
            .iter()
            .any(|m| matches!(m, Modifier::U64 | Modifier::B64))
    }

    fn uses_generic_shared_reg(&self, operand: &Operand) -> bool {
        match operand {
            Operand::Register(reg) => self.shared_base_regs.contains(reg),
            Operand::Memory(addr) => {
                // Check if the memory address contains a generic shared register
                self.shared_base_regs.iter().any(|reg| addr.contains(reg))
            }
            _ => false,
        }
    }
}

impl Default for AddressSpaceValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    // F021: No cvta.shared followed by generic ld/st
    #[test]
    fn f021_no_generic_shared_access() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .u32 %r<10>;
                ld.shared.u32 %r0, [%r1];
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).expect("parser creation should succeed");
        let module = parser.parse().expect("parsing should succeed");

        let mut validator = AddressSpaceValidator::new();
        let bugs = validator.detect_generic_shared_access(&module);

        assert!(
            bugs.is_empty(),
            "F021: Should have no generic shared access bugs"
        );
    }

    // F023: Direct .shared addressing preferred
    #[test]
    fn f023_direct_shared_addressing() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .u32 %r<10>;
                ld.shared.u32 %r0, [%r1];
                st.shared.u32 [%r2], %r0;
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).expect("parser creation should succeed");
        let module = parser.parse().expect("parsing should succeed");

        let mut validator = AddressSpaceValidator::new();
        let bugs = validator.detect_generic_shared_access(&module);

        assert!(
            bugs.is_empty(),
            "F023: Direct shared addressing should not trigger bugs"
        );
    }
}
