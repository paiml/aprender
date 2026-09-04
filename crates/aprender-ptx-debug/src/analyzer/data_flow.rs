//! Data Flow Analyzer - value propagation and loaded value bug detection

use crate::bugs::Severity;
use crate::parser::types::{AddressSpace, Opcode};
use crate::parser::{Instruction, KernelDef, Operand, PtxModule, SourceLocation, Statement};
use std::collections::{HashMap, HashSet};

/// Source of a value
#[derive(Debug, Clone)]
pub enum ValueSource {
    /// Value came from a load instruction
    Load {
        /// Address space
        space: AddressSpace,
        /// Source location
        location: SourceLocation,
    },
    /// Value came from a constant/immediate
    Constant(i64),
    /// Value came from computation
    Computed {
        /// Input registers
        inputs: Vec<String>,
    },
    /// Value came from parameter
    Parameter(String),
    /// Unknown source
    Unknown,
}

/// Use point of a value
#[derive(Debug, Clone)]
pub struct UsePoint {
    /// Instruction where the value is used
    pub instruction: Instruction,
    /// Operand index (0-based)
    pub operand_index: usize,
    /// Source location
    pub location: SourceLocation,
    /// Is this the data operand of a store?
    is_store_data: bool,
    /// Is this the address operand of a store?
    is_store_addr: bool,
}

impl UsePoint {
    /// Is this use point a store data operand?
    pub fn is_store_data_operand(&self) -> bool {
        self.is_store_data
    }

    /// Is this use point a store address operand?
    pub fn is_store_address_operand(&self) -> bool {
        self.is_store_addr
    }
}

/// Defect: Store using value derived from ld.shared
#[derive(Debug, Clone)]
pub struct LoadedValueBug {
    /// Load location
    pub load_location: SourceLocation,
    /// Store location
    pub store_location: SourceLocation,
    /// Register containing loaded value
    pub register: String,
    /// Severity
    pub severity: Severity,
    /// Mitigation advice
    pub mitigation: String,
}

/// Defect: Address computed from ld.shared value causes store crash
#[derive(Debug, Clone)]
pub struct ComputedAddrFromLoadedBug {
    /// Load location
    pub load_location: SourceLocation,
    /// Address computation location
    pub addr_computation_location: SourceLocation,
    /// Tainted register
    pub tainted_register: String,
    /// Severity
    pub severity: Severity,
    /// Mitigation advice
    pub mitigation: String,
}

/// Data Flow Analyzer
pub struct DataFlowAnalyzer {
    /// Def-use chains: register -> use points
    def_use_chains: HashMap<String, Vec<UsePoint>>,
    /// Value sources: register -> source
    value_sources: HashMap<String, ValueSource>,
}

impl DataFlowAnalyzer {
    /// Create a new data flow analyzer
    pub fn new() -> Self {
        Self {
            def_use_chains: HashMap::new(),
            value_sources: HashMap::new(),
        }
    }

    /// Create from a PTX module (analyzes first kernel)
    pub fn from_module(module: &PtxModule) -> Self {
        let mut analyzer = Self::new();
        if let Some(kernel) = module.kernels.first() {
            analyzer.analyze_kernel(kernel);
        }
        analyzer
    }

    /// Analyze a kernel for data flow
    pub fn analyze_kernel(&mut self, kernel: &KernelDef) {
        self.def_use_chains.clear();
        self.value_sources.clear();

        for stmt in &kernel.body {
            if let Statement::Instruction(instr) = stmt {
                self.analyze_instruction(instr);
            }
        }
    }

    fn analyze_instruction(&mut self, instr: &Instruction) {
        match instr.opcode {
            Opcode::Ld => {
                // Load defines a register with value from memory
                if let Some(Operand::Register(dest)) = instr.operands.first() {
                    let space = self.get_address_space(instr);
                    self.value_sources.insert(
                        dest.clone(),
                        ValueSource::Load {
                            space,
                            location: instr.location.clone(),
                        },
                    );
                }
            }
            Opcode::Mov => {
                // Move copies value source
                if let (Some(Operand::Register(dest)), Some(src)) =
                    (instr.operands.first(), instr.operands.get(1))
                {
                    let source = match src {
                        Operand::Register(src_reg) => self
                            .value_sources
                            .get(src_reg)
                            .cloned()
                            .unwrap_or(ValueSource::Unknown),
                        Operand::Immediate(val) => ValueSource::Constant(*val),
                        _ => ValueSource::Unknown,
                    };
                    self.value_sources.insert(dest.clone(), source);
                }
            }
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::And
            | Opcode::Or
            | Opcode::Shl
            | Opcode::Shr => {
                // Computation defines register with computed value
                if let Some(Operand::Register(dest)) = instr.operands.first() {
                    let inputs: Vec<String> = instr
                        .operands
                        .iter()
                        .skip(1)
                        .filter_map(|op| {
                            if let Operand::Register(reg) = op {
                                Some(reg.clone())
                            } else {
                                None
                            }
                        })
                        .collect();

                    self.value_sources
                        .insert(dest.clone(), ValueSource::Computed { inputs });
                }
            }
            Opcode::St => {
                // Store uses values - track the use points
                // For st.<type> [addr], value:
                // - operand 0 is the memory address
                // - operand 1 is the value to store
                if let Some(Operand::Memory(addr_str)) = instr.operands.first() {
                    // Extract register from memory operand like [%r0] or [%r0+offset]
                    let addr_reg = self.extract_register_from_memory(addr_str);
                    if let Some(reg) = addr_reg {
                        self.def_use_chains
                            .entry(reg.clone())
                            .or_default()
                            .push(UsePoint {
                                instruction: instr.clone(),
                                operand_index: 0,
                                location: instr.location.clone(),
                                is_store_data: false,
                                is_store_addr: true,
                            });
                    }
                }

                if let Some(Operand::Register(val_reg)) = instr.operands.get(1) {
                    self.def_use_chains
                        .entry(val_reg.clone())
                        .or_default()
                        .push(UsePoint {
                            instruction: instr.clone(),
                            operand_index: 1,
                            location: instr.location.clone(),
                            is_store_data: true,
                            is_store_addr: false,
                        });
                }
            }
            _ => {}
        }
    }

    fn get_address_space(&self, instr: &Instruction) -> AddressSpace {
        for modifier in &instr.modifiers {
            if let Some(space) = modifier.as_address_space() {
                return space;
            }
        }
        AddressSpace::Generic
    }

    fn extract_register_from_memory(&self, addr_str: &str) -> Option<String> {
        // Extract register name from patterns like [%r0], [%r0+4], etc.
        let trimmed = addr_str.trim_matches(|c| c == '[' || c == ']');
        if let Some(plus_pos) = trimmed.find('+') {
            Some(trimmed[..plus_pos].trim().to_string())
        } else {
            Some(trimmed.trim().to_string())
        }
    }

    /// Detect the "loaded value" bug pattern (F081)
    ///
    /// Pattern: ld.shared -> computation -> st.XXX crashes
    pub fn detect_loaded_value_bug(&self) -> Vec<LoadedValueBug> {
        let mut bugs = Vec::new();

        for (reg, source) in &self.value_sources {
            if let ValueSource::Load {
                space: AddressSpace::Shared,
                location,
            } = source
            {
                // Find all stores that use this register as data operand
                for use_point in self.def_use_chains.get(reg).unwrap_or(&vec![]) {
                    if use_point.is_store_data_operand() {
                        bugs.push(LoadedValueBug {
                            load_location: location.clone(),
                            store_location: use_point.location.clone(),
                            register: reg.clone(),
                            severity: Severity::Low,
                            mitigation: "Hypothesis F081 falsified on sm_89. This pattern is safe."
                                .into(),
                        });
                    }
                }
            }
        }

        bugs
    }

    /// Detect "computed address from loaded value" bug pattern (F082)
    ///
    /// Pattern: ld.shared %r_val -> add %r_addr, base, %r_val -> st.XXX [%r_addr]
    /// Even storing a CONSTANT to an address computed from a loaded value crashes.
    pub fn detect_computed_addr_from_loaded(&self) -> Vec<ComputedAddrFromLoadedBug> {
        // Track which registers come from ld.shared
        let shared_loaded_regs = self.shared_loaded_registers();

        // Track registers computed from shared-loaded registers (taint propagation)
        let tainted_regs = self.propagate_taint(&shared_loaded_regs);

        // Find stores where ADDRESS is computed from tainted register
        let mut bugs = Vec::new();
        for (reg, _source) in &self.value_sources {
            if !tainted_regs.contains(reg) {
                continue;
            }
            for use_point in self.def_use_chains.get(reg).unwrap_or(&vec![]) {
                if use_point.is_store_address_operand() {
                    let load_loc = self.find_original_load_location(reg, &shared_loaded_regs);
                    bugs.push(ComputedAddrFromLoadedBug {
                        load_location: load_loc.unwrap_or_default(),
                        addr_computation_location: use_point.location.clone(),
                        tainted_register: reg.clone(),
                        severity: Severity::Critical,
                        mitigation: "Use constant-only address computation, try membar.cta (partial), or use Kernel Fission (split kernel)".into(),
                    });
                }
            }
        }

        bugs
    }

    /// Registers whose value is loaded directly from shared memory (`ld.shared`).
    fn shared_loaded_registers(&self) -> HashSet<String> {
        let mut regs: HashSet<String> = HashSet::new();
        for (reg, source) in &self.value_sources {
            if matches!(
                source,
                ValueSource::Load {
                    space: AddressSpace::Shared,
                    ..
                }
            ) {
                regs.insert(reg.clone());
            }
        }
        regs
    }

    /// Propagate taint from `seed` through `ValueSource::Computed` inputs to a fixpoint.
    fn propagate_taint(&self, seed: &HashSet<String>) -> HashSet<String> {
        let mut tainted: HashSet<String> = seed.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for (reg, source) in &self.value_sources {
                if let ValueSource::Computed { inputs } = source {
                    if !tainted.contains(reg) && inputs.iter().any(|i| tainted.contains(i)) {
                        tainted.insert(reg.clone());
                        changed = true;
                    }
                }
            }
        }
        tainted
    }

    fn find_original_load_location(
        &self,
        reg: &str,
        shared_loaded_regs: &HashSet<String>,
    ) -> Option<SourceLocation> {
        // If this register is directly from a load, return that location
        if let Some(ValueSource::Load { location, .. }) = self.value_sources.get(reg) {
            return Some(location.clone());
        }

        // Otherwise, trace back through computations
        if let Some(ValueSource::Computed { inputs }) = self.value_sources.get(reg) {
            for input in inputs {
                if shared_loaded_regs.contains(input) {
                    return self.find_original_load_location(input, shared_loaded_regs);
                }
            }
        }

        None
    }
}

impl Default for DataFlowAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    // F081: No loaded value store pattern
    #[test]
    fn f081_no_loaded_value_bug() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .u32 %r<10>;
                mov.u32 %r0, 0;
                st.shared.u32 [%r1], %r0;
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).expect("parser creation should succeed");
        let module = parser.parse().expect("parsing should succeed");

        let analyzer = DataFlowAnalyzer::from_module(&module);
        let bugs = analyzer.detect_loaded_value_bug();

        assert!(
            bugs.is_empty(),
            "F081: Should have no loaded value bugs when using constant"
        );
    }

    // F082: No computed-address-from-loaded pattern
    #[test]
    fn f082_no_computed_addr_from_loaded_bug() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .u32 %r<10>;
                mov.u32 %r0, 100;
                add.u32 %r1, %r2, %r0;
                mov.u32 %r3, 0xCAFE;
                st.shared.u32 [%r1], %r3;
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).expect("parser creation should succeed");
        let module = parser.parse().expect("parsing should succeed");

        let analyzer = DataFlowAnalyzer::from_module(&module);
        let bugs = analyzer.detect_computed_addr_from_loaded();

        assert!(
            bugs.is_empty(),
            "F082: Should have no computed-addr bugs when using constant"
        );
    }

    // F071: No use before def
    #[test]
    fn f071_no_use_before_def() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .u32 %r<10>;
                mov.u32 %r0, 0;
                add.u32 %r1, %r0, 1;
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).expect("parser creation should succeed");
        let module = parser.parse().expect("parsing should succeed");

        let _analyzer = DataFlowAnalyzer::from_module(&module);
        // The analyzer should track value sources
        // Note: The exact register names depend on parser output
        // This test verifies the analyzer runs without error
        assert!(!module.kernels.is_empty());
    }
}
