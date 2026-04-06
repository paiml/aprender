//! Type Checker - validates register types match operation requirements

use crate::bugs::Severity;
use crate::parser::types::{Opcode, PtxType};
use crate::parser::{Instruction, Operand, PtxModule, SourceLocation};
use std::collections::HashMap;

/// Type error
#[derive(Debug, Clone)]
pub struct TypeError {
    /// Error message
    pub message: String,
    /// Source location
    pub location: SourceLocation,
    /// Severity
    pub severity: Severity,
}

/// Type Checker pass
pub struct TypeChecker {
    register_types: HashMap<String, PtxType>,
    errors: Vec<TypeError>,
}

impl TypeChecker {
    /// Create a new type checker
    pub fn new() -> Self {
        Self { register_types: HashMap::new(), errors: Vec::new() }
    }

    /// Analyze a module for type errors
    pub fn analyze(&mut self, module: &PtxModule) -> Vec<TypeError> {
        self.errors.clear();

        for kernel in &module.kernels {
            // Register declared types
            for reg in &kernel.registers {
                self.register_types.insert(reg.name.clone(), reg.ty);
            }

            // Check instructions
            for stmt in &kernel.body {
                if let crate::parser::Statement::Instruction(instr) = stmt {
                    if let Err(e) = self.check_instruction(instr) {
                        self.errors.push(e);
                    }
                }
            }
        }

        self.errors.clone()
    }

    /// Check instruction operand types match expected types
    pub fn check_instruction(&mut self, instr: &Instruction) -> Result<(), TypeError> {
        match instr.opcode {
            Opcode::Ld => self.check_load(instr),
            Opcode::St => self.check_store(instr),
            Opcode::Mov => self.check_mov(instr),
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => self.check_arithmetic(instr),
            Opcode::And | Opcode::Or | Opcode::Xor => self.check_bitwise(instr),
            Opcode::Setp => self.check_setp(instr),
            Opcode::Cvt | Opcode::Cvta => self.check_conversion(instr),
            _ => Ok(()),
        }
    }

    fn check_load(&self, instr: &Instruction) -> Result<(), TypeError> {
        // Get instruction type from modifiers
        let instr_type = self.get_instruction_type(instr);

        // Check destination register type matches
        if let Some(Operand::Register(dest)) = instr.operands.first() {
            if let Some(reg_type) = self.register_types.get(dest) {
                if let Some(it) = instr_type {
                    if !self.types_compatible(*reg_type, it) {
                        return Err(TypeError {
                            message: format!(
                                "Load destination type mismatch: register {} is {:?}, instruction is {:?}",
                                dest, reg_type, it
                            ),
                            location: instr.location.clone(),
                            severity: Severity::Medium,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    fn check_store(&self, instr: &Instruction) -> Result<(), TypeError> {
        // Get instruction type from modifiers
        let instr_type = self.get_instruction_type(instr);

        // Check source register type matches (second operand for store)
        if let Some(Operand::Register(src)) = instr.operands.get(1) {
            if let Some(reg_type) = self.register_types.get(src) {
                if let Some(it) = instr_type {
                    if !self.types_compatible(*reg_type, it) {
                        return Err(TypeError {
                            message: format!(
                                "Store source type mismatch: register {} is {:?}, instruction is {:?}",
                                src, reg_type, it
                            ),
                            location: instr.location.clone(),
                            severity: Severity::Medium,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    fn check_mov(&self, instr: &Instruction) -> Result<(), TypeError> {
        // Check source and destination have compatible types
        if instr.operands.len() >= 2 {
            if let (Some(Operand::Register(dest)), Some(Operand::Register(src))) =
                (instr.operands.first(), instr.operands.get(1))
            {
                if let (Some(&dest_type), Some(&src_type)) =
                    (self.register_types.get(dest), self.register_types.get(src))
                {
                    if dest_type.size_bytes() != src_type.size_bytes() {
                        return Err(TypeError {
                            message: format!(
                                "Mov type mismatch: {} ({:?}) and {} ({:?}) have different sizes",
                                dest, dest_type, src, src_type
                            ),
                            location: instr.location.clone(),
                            severity: Severity::Medium,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    fn check_arithmetic(&self, instr: &Instruction) -> Result<(), TypeError> {
        // All operands should have consistent types
        let instr_type = self.get_instruction_type(instr);

        for operand in &instr.operands {
            if let Operand::Register(reg) = operand {
                if let Some(reg_type) = self.register_types.get(reg) {
                    if let Some(it) = instr_type {
                        if !self.types_compatible(*reg_type, it) {
                            return Err(TypeError {
                                message: format!(
                                    "Arithmetic operand type mismatch: {} is {:?}, expected {:?}",
                                    reg, reg_type, it
                                ),
                                location: instr.location.clone(),
                                severity: Severity::Medium,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn check_bitwise(&self, instr: &Instruction) -> Result<(), TypeError> {
        // Bitwise operations require integer/bit types
        let instr_type: Option<PtxType> = self.get_instruction_type(instr);

        if let Some(it) = instr_type {
            if it.is_float() {
                return Err(TypeError {
                    message: "Bitwise operation on floating point type".into(),
                    location: instr.location.clone(),
                    severity: Severity::High,
                });
            }
        }

        Ok(())
    }

    fn check_setp(&self, instr: &Instruction) -> Result<(), TypeError> {
        // First operand should be predicate
        if let Some(Operand::Register(dest)) = instr.operands.first() {
            if let Some(reg_type) = self.register_types.get(dest) {
                if *reg_type != PtxType::Pred {
                    return Err(TypeError {
                        message: format!(
                            "setp destination {} should be predicate, found {:?}",
                            dest, reg_type
                        ),
                        location: instr.location.clone(),
                        severity: Severity::High,
                    });
                }
            }
        }

        Ok(())
    }

    fn check_conversion(&self, _instr: &Instruction) -> Result<(), TypeError> {
        // Conversion instructions are generally flexible
        Ok(())
    }

    fn get_instruction_type(&self, instr: &Instruction) -> Option<PtxType> {
        for modifier in &instr.modifiers {
            if let Some(ty) = modifier.as_type() {
                return Some(ty);
            }
        }
        None
    }

    fn types_compatible(&self, reg_type: PtxType, instr_type: PtxType) -> bool {
        // Same type is always compatible
        if reg_type == instr_type {
            return true;
        }

        // Same size untyped (bXX) is compatible with typed
        if reg_type.size_bytes() == instr_type.size_bytes() {
            return true;
        }

        false
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    // F011: Load dest type matches instruction type
    #[test]
    fn f011_load_dest_type_matches() {
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

        let mut checker = TypeChecker::new();
        let errors = checker.analyze(&module);

        // Should have no errors - u32 register with u32 load
        assert!(errors.is_empty(), "F011: Type mismatch errors: {:?}", errors);
    }

    // F017: Predicate registers used correctly
    #[test]
    fn f017_predicate_register_usage() {
        // This test validates that setp destinations are predicate registers
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .pred %p<10>;
                .reg .u32 %r<10>;
                setp.eq.u32 %p0, %r0, %r1;
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).expect("parser creation should succeed");
        let module = parser.parse().expect("parsing should succeed");

        let mut checker = TypeChecker::new();
        let errors = checker.analyze(&module);

        assert!(errors.is_empty(), "F017: Predicate usage errors: {:?}", errors);
    }
}
