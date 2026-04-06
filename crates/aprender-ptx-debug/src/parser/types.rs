//! PTX Type System definitions

/// PTX data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxType {
    /// 8-bit signed integer
    S8,
    /// 16-bit signed integer
    S16,
    /// 32-bit signed integer
    S32,
    /// 64-bit signed integer
    S64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 16-bit floating point
    F16,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 8-bit untyped
    B8,
    /// 16-bit untyped
    B16,
    /// 32-bit untyped
    B32,
    /// 64-bit untyped
    B64,
    /// Predicate (boolean)
    Pred,
}

impl PtxType {
    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            PtxType::S8 | PtxType::U8 | PtxType::B8 => 1,
            PtxType::S16 | PtxType::U16 | PtxType::B16 | PtxType::F16 => 2,
            PtxType::S32 | PtxType::U32 | PtxType::B32 | PtxType::F32 => 4,
            PtxType::S64 | PtxType::U64 | PtxType::B64 | PtxType::F64 => 8,
            PtxType::Pred => 1,
        }
    }

    /// Is this a signed type
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            PtxType::S8 | PtxType::S16 | PtxType::S32 | PtxType::S64
        )
    }

    /// Is this a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, PtxType::F16 | PtxType::F32 | PtxType::F64)
    }

    /// Is this a 64-bit type
    pub fn is_64bit(&self) -> bool {
        matches!(
            self,
            PtxType::S64 | PtxType::U64 | PtxType::B64 | PtxType::F64
        )
    }
}

impl std::fmt::Display for PtxType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            PtxType::S8 => ".s8",
            PtxType::S16 => ".s16",
            PtxType::S32 => ".s32",
            PtxType::S64 => ".s64",
            PtxType::U8 => ".u8",
            PtxType::U16 => ".u16",
            PtxType::U32 => ".u32",
            PtxType::U64 => ".u64",
            PtxType::F16 => ".f16",
            PtxType::F32 => ".f32",
            PtxType::F64 => ".f64",
            PtxType::B8 => ".b8",
            PtxType::B16 => ".b16",
            PtxType::B32 => ".b32",
            PtxType::B64 => ".b64",
            PtxType::Pred => ".pred",
        };
        write!(f, "{}", s)
    }
}

/// Address space qualifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    /// Generic (unqualified) address
    Generic,
    /// Global memory
    Global,
    /// Shared memory (per-block)
    Shared,
    /// Local memory (per-thread)
    Local,
    /// Constant memory
    Const,
    /// Parameter space
    Param,
    /// Texture memory
    Texture,
    /// Surface memory
    Surface,
}

impl std::fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AddressSpace::Generic => "",
            AddressSpace::Global => ".global",
            AddressSpace::Shared => ".shared",
            AddressSpace::Local => ".local",
            AddressSpace::Const => ".const",
            AddressSpace::Param => ".param",
            AddressSpace::Texture => ".tex",
            AddressSpace::Surface => ".surf",
        };
        write!(f, "{}", s)
    }
}

/// SM (Streaming Multiprocessor) target architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SmTarget {
    /// Unknown/unspecified
    #[default]
    Unknown,
    /// SM 5.0 (Maxwell)
    Sm50,
    /// SM 5.2 (Maxwell)
    Sm52,
    /// SM 6.0 (Pascal)
    Sm60,
    /// SM 6.1 (Pascal)
    Sm61,
    /// SM 7.0 (Volta)
    Sm70,
    /// SM 7.5 (Turing)
    Sm75,
    /// SM 8.0 (Ampere)
    Sm80,
    /// SM 8.6 (Ampere)
    Sm86,
    /// SM 8.9 (Ada Lovelace)
    Sm89,
    /// SM 9.0 (Hopper)
    Sm90,
}

impl SmTarget {
    /// Minimum PTX version for this target
    pub fn min_ptx_version(&self) -> (u8, u8) {
        match self {
            SmTarget::Unknown => (1, 0),
            SmTarget::Sm50 | SmTarget::Sm52 => (4, 0),
            SmTarget::Sm60 | SmTarget::Sm61 => (5, 0),
            SmTarget::Sm70 => (6, 0),
            SmTarget::Sm75 => (6, 3),
            SmTarget::Sm80 | SmTarget::Sm86 => (7, 0),
            SmTarget::Sm89 => (7, 8),
            SmTarget::Sm90 => (8, 0),
        }
    }

    /// Does this target support Tensor Cores
    pub fn has_tensor_cores(&self) -> bool {
        matches!(
            self,
            SmTarget::Sm70
                | SmTarget::Sm75
                | SmTarget::Sm80
                | SmTarget::Sm86
                | SmTarget::Sm89
                | SmTarget::Sm90
        )
    }
}

/// PTX Opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    // Data Movement
    /// Load
    Ld,
    /// Store
    St,
    /// Move
    Mov,
    /// Convert address space
    Cvta,
    /// Convert type
    Cvt,

    // Arithmetic
    /// Add
    Add,
    /// Subtract
    Sub,
    /// Multiply
    Mul,
    /// Divide
    Div,
    /// Remainder
    Rem,
    /// Multiply-add
    Mad,
    /// Fused multiply-add
    Fma,
    /// Negate
    Neg,
    /// Absolute value
    Abs,
    /// Minimum
    Min,
    /// Maximum
    Max,

    // Logic
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Bitwise NOT
    Not,
    /// Shift left
    Shl,
    /// Shift right
    Shr,

    // Comparison
    /// Set predicate
    Setp,
    /// Select
    Selp,

    // Control Flow
    /// Branch
    Bra,
    /// Call function
    Call,
    /// Return
    Ret,
    /// Exit kernel
    Exit,

    // Synchronization
    /// Barrier
    Bar,
    /// Memory barrier
    MemBar,
    /// Atomic operation
    Atom,
    /// Reduction operation
    Red,

    // Special
    /// Texture load
    Tex,
    /// Texture load 4
    Tld4,
    /// Surface load
    Suld,
    /// Surface store
    Sust,
    /// Warp shuffle
    Shfl,
    /// Warp vote
    Vote,
    /// Matrix multiply-accumulate
    Mma,
    /// Warp MMA
    Wmma,
    /// Load matrix
    LdMatrix,
    /// Copy (async)
    Cp,
    /// Prefetch
    Prefetch,

    /// Unknown opcode
    Unknown,
}

impl Opcode {
    /// Is this a load instruction
    pub fn is_load(&self) -> bool {
        matches!(
            self,
            Opcode::Ld | Opcode::Tex | Opcode::Tld4 | Opcode::Suld | Opcode::LdMatrix
        )
    }

    /// Is this a store instruction
    pub fn is_store(&self) -> bool {
        matches!(self, Opcode::St | Opcode::Sust)
    }

    /// Is this a memory operation
    pub fn is_memory_op(&self) -> bool {
        self.is_load() || self.is_store() || matches!(self, Opcode::Atom | Opcode::Red)
    }

    /// Is this a synchronization instruction
    pub fn is_sync(&self) -> bool {
        matches!(self, Opcode::Bar | Opcode::MemBar)
    }

    /// Is this a branch instruction
    pub fn is_branch(&self) -> bool {
        matches!(
            self,
            Opcode::Bra | Opcode::Call | Opcode::Ret | Opcode::Exit
        )
    }
}

/// Instruction modifiers
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Modifier {
    // Address space
    /// .shared
    Shared,
    /// .global
    Global,
    /// .local
    Local,
    /// .const
    Const,
    /// .param
    Param,

    // Types
    /// .u32
    U32,
    /// .u64
    U64,
    /// .s32
    S32,
    /// .s64
    S64,
    /// .f32
    F32,
    /// .f64
    F64,
    /// .b32
    B32,
    /// .b64
    B64,

    // Synchronization
    /// .sync
    Sync,
    /// .cta
    Cta,
    /// .gl
    Gl,
    /// .sys
    Sys,

    // Atomic
    /// .add (atomic add)
    AtomicAdd,
    /// .cas (compare and swap)
    AtomicCas,
    /// .exch (exchange)
    AtomicExch,
    /// .min
    AtomicMin,
    /// .max
    AtomicMax,

    // Other
    /// Other modifier
    Other(String),
}

impl Modifier {
    /// Get the address space if this is an address space modifier
    pub fn as_address_space(&self) -> Option<AddressSpace> {
        match self {
            Modifier::Shared => Some(AddressSpace::Shared),
            Modifier::Global => Some(AddressSpace::Global),
            Modifier::Local => Some(AddressSpace::Local),
            Modifier::Const => Some(AddressSpace::Const),
            Modifier::Param => Some(AddressSpace::Param),
            _ => None,
        }
    }

    /// Get the type if this is a type modifier
    pub fn as_type(&self) -> Option<PtxType> {
        match self {
            Modifier::U32 => Some(PtxType::U32),
            Modifier::U64 => Some(PtxType::U64),
            Modifier::S32 => Some(PtxType::S32),
            Modifier::S64 => Some(PtxType::S64),
            Modifier::F32 => Some(PtxType::F32),
            Modifier::F64 => Some(PtxType::F64),
            Modifier::B32 => Some(PtxType::B32),
            Modifier::B64 => Some(PtxType::B64),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_type_size() {
        assert_eq!(PtxType::U8.size_bytes(), 1);
        assert_eq!(PtxType::U16.size_bytes(), 2);
        assert_eq!(PtxType::U32.size_bytes(), 4);
        assert_eq!(PtxType::U64.size_bytes(), 8);
        assert_eq!(PtxType::F32.size_bytes(), 4);
        assert_eq!(PtxType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_ptx_type_properties() {
        assert!(PtxType::S32.is_signed());
        assert!(!PtxType::U32.is_signed());
        assert!(PtxType::F32.is_float());
        assert!(!PtxType::U32.is_float());
        assert!(PtxType::U64.is_64bit());
        assert!(!PtxType::U32.is_64bit());
    }

    #[test]
    fn test_sm_target_ptx_version() {
        assert!(SmTarget::Sm90.min_ptx_version() >= (8, 0));
        assert!(SmTarget::Sm70.min_ptx_version() >= (6, 0));
    }

    #[test]
    fn test_opcode_categories() {
        assert!(Opcode::Ld.is_load());
        assert!(Opcode::St.is_store());
        assert!(Opcode::Bar.is_sync());
        assert!(Opcode::Bra.is_branch());
    }

    #[test]
    fn test_modifier_conversion() {
        assert_eq!(
            Modifier::Shared.as_address_space(),
            Some(AddressSpace::Shared)
        );
        assert_eq!(Modifier::U32.as_type(), Some(PtxType::U32));
    }
}
