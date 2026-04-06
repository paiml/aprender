//! PTX Abstract Syntax Tree definitions

use super::types::{AddressSpace, Modifier, Opcode, PtxType, SmTarget};

/// PTX Module - top-level AST node
#[derive(Debug, Clone, Default)]
pub struct PtxModule {
    /// PTX version (major, minor)
    pub version: (u8, u8),
    /// Target SM architecture
    pub target: SmTarget,
    /// Address size (32 or 64)
    pub address_size: u8,
    /// Global declarations
    pub globals: Vec<GlobalDecl>,
    /// Kernel definitions
    pub kernels: Vec<KernelDef>,
    /// Function definitions
    pub functions: Vec<FunctionDef>,
}

/// Kernel definition
#[derive(Debug, Clone)]
pub struct KernelDef {
    /// Kernel name
    pub name: String,
    /// Is this an entry point (.entry vs .func)
    pub is_entry: bool,
    /// Parameters
    pub params: Vec<Param>,
    /// Register declarations
    pub registers: Vec<RegisterDecl>,
    /// Shared memory declarations
    pub shared_mem: Vec<SharedMemDecl>,
    /// Body statements
    pub body: Vec<Statement>,
}

/// Function definition (non-entry)
#[derive(Debug, Clone)]
pub struct FunctionDef {
    /// Function name
    pub name: String,
    /// Return type
    pub return_type: Option<PtxType>,
    /// Parameters
    pub params: Vec<Param>,
    /// Register declarations
    pub registers: Vec<RegisterDecl>,
    /// Body statements
    pub body: Vec<Statement>,
}

/// Global declaration
#[derive(Debug, Clone)]
pub struct GlobalDecl {
    /// Name
    pub name: String,
    /// Address space
    pub space: AddressSpace,
    /// Type
    pub ty: PtxType,
    /// Size in bytes
    pub size: usize,
    /// Initial value (if any)
    pub init: Option<Vec<u8>>,
}

/// Parameter declaration
#[derive(Debug, Clone)]
pub struct Param {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub ty: PtxType,
}

/// Register declaration
#[derive(Debug, Clone)]
pub struct RegisterDecl {
    /// Register name (e.g., "%r0" or "%r<10>")
    pub name: String,
    /// Register type
    pub ty: PtxType,
}

/// Shared memory declaration
#[derive(Debug, Clone)]
pub struct SharedMemDecl {
    /// Name
    pub name: String,
    /// Size in bytes
    pub size: usize,
    /// Element type
    pub ty: PtxType,
}

/// Statement in a kernel/function body
#[derive(Debug, Clone)]
pub enum Statement {
    /// Label definition
    Label(String),
    /// Instruction
    Instruction(Instruction),
    /// Directive within body
    Directive(Directive),
    /// Comment (preserved for debugging)
    Comment(String),
}

/// PTX Instruction
#[derive(Debug, Clone)]
pub struct Instruction {
    /// Opcode
    pub opcode: Opcode,
    /// Modifiers (.shared, .u32, etc.)
    pub modifiers: Vec<Modifier>,
    /// Operands
    pub operands: Vec<Operand>,
    /// Predicate (if any)
    pub predicate: Option<Predicate>,
    /// Source location
    pub location: SourceLocation,
}

/// Directive within body
#[derive(Debug, Clone)]
pub enum Directive {
    /// .loc directive for debug info
    Loc { file: u32, line: u32, column: u32 },
    /// .pragma directive
    Pragma(String),
    /// Other directive
    Other(String),
}

/// Instruction operand
#[derive(Debug, Clone)]
pub enum Operand {
    /// Register (%r0, %rd1, etc.)
    Register(String),
    /// Memory reference ([%r0], [%r0+4], etc.)
    Memory(String),
    /// Immediate value
    Immediate(i64),
    /// Floating point immediate
    ImmediateFloat(f64),
    /// Label reference (for branches)
    Label(String),
    /// Vector operand {%r0, %r1, %r2, %r3}
    Vector(Vec<Operand>),
}

impl Operand {
    /// Check if this operand is a register
    pub fn is_register(&self) -> bool {
        matches!(self, Operand::Register(_))
    }

    /// Check if this operand is a memory reference
    pub fn is_memory(&self) -> bool {
        matches!(self, Operand::Memory(_))
    }

    /// Get the register name if this is a register operand
    pub fn as_register(&self) -> Option<&str> {
        match self {
            Operand::Register(name) => Some(name),
            _ => None,
        }
    }
}

impl std::fmt::Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::Register(name) => write!(f, "{}", name),
            Operand::Memory(addr) => write!(f, "{}", addr),
            Operand::Immediate(val) => write!(f, "{}", val),
            Operand::ImmediateFloat(val) => write!(f, "{}", val),
            Operand::Label(name) => write!(f, "{}", name),
            Operand::Vector(ops) => {
                write!(f, "{{")?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", op)?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Predicate for conditional execution
#[derive(Debug, Clone)]
pub struct Predicate {
    /// Predicate register name
    pub register: String,
    /// Is negated (@!p)
    pub negated: bool,
}

/// Source location for error reporting
#[derive(Debug, Clone, Default)]
pub struct SourceLocation {
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
    /// Source file (if known)
    pub file: Option<String>,
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(file) = &self.file {
            write!(f, "{}:{}:{}", file, self.line, self.column)
        } else {
            write!(f, "{}:{}", self.line, self.column)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operand_display() {
        assert_eq!(format!("{}", Operand::Register("%r0".into())), "%r0");
        assert_eq!(format!("{}", Operand::Memory("[%r0]".into())), "[%r0]");
        assert_eq!(format!("{}", Operand::Immediate(42)), "42");
    }

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation { line: 10, column: 5, file: Some("test.ptx".into()) };
        assert_eq!(format!("{}", loc), "test.ptx:10:5");

        let loc = SourceLocation { line: 10, column: 5, file: None };
        assert_eq!(format!("{}", loc), "10:5");
    }

    #[test]
    fn test_operand_type_checks() {
        let reg = Operand::Register("%r0".into());
        assert!(reg.is_register());
        assert!(!reg.is_memory());

        let mem = Operand::Memory("[%r0]".into());
        assert!(!mem.is_register());
        assert!(mem.is_memory());
    }
}
