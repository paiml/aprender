//! PTX Instructions
//!
//! Defines the PTX instruction set and operations.

use super::registers::VirtualReg;
use super::types::{PtxStateSpace, PtxType};

/// PTX operation codes
#[derive(Debug, Clone, PartialEq)]
pub enum PtxOp {
    // ===== Arithmetic =====
    /// Add two values
    Add,
    /// Subtract
    Sub,
    /// Multiply (emits mul.lo for integers, mul for floats)
    Mul,
    /// Multiply wide: u32 x u32 -> u64 (or s32 x s32 -> s64)
    /// Emits `mul.wide.u32` or `mul.wide.s32`.
    /// Separate from Mul to avoid fragile source-type inference at emission time.
    MulWide,
    /// Multiply-add (fused)
    Mad,
    /// Multiply-add low bits
    MadLo,
    /// Divide
    Div,
    /// Remainder
    Rem,
    /// Absolute value
    Abs,
    /// Negate
    Neg,
    /// Minimum
    Min,
    /// Maximum
    Max,

    // ===== Floating Point Special =====
    /// Reciprocal
    Rcp,
    /// Reciprocal square root
    Rsqrt,
    /// Square root
    Sqrt,
    /// Sine (approximate)
    Sin,
    /// Cosine (approximate)
    Cos,
    /// Exponential base 2 (approximate)
    Ex2,
    /// Logarithm base 2 (approximate)
    Lg2,
    /// Fused multiply-add
    Fma,

    // ===== Comparison =====
    /// Set predicate (comparison)
    Setp,

    // ===== Logical/Bitwise =====
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
    /// Shift right (logical)
    Shr,

    // ===== Data Movement =====
    /// Move/copy
    Mov,
    /// Load from memory
    Ld,
    /// Store to memory
    St,
    /// Load parameter
    LdParam,
    /// Convert type
    Cvt,
    /// Convert address between state spaces (cvta)
    /// Used to convert shared/global addresses to generic pointers for WMMA
    Cvta,
    /// Select based on predicate
    Selp,

    // ===== Warp-Level =====
    /// Warp shuffle down
    ShflDown,
    /// Warp shuffle up
    ShflUp,
    /// Warp shuffle XOR
    ShflBfly,
    /// Warp shuffle indexed
    ShflIdx,
    /// Warp vote all
    VoteAll,
    /// Warp vote any
    VoteAny,
    /// Warp vote ballot
    VoteBallot,

    // ===== Control Flow =====
    /// Branch
    Bra,
    /// Call
    Call,
    /// Return
    Ret,
    /// Exit kernel
    Exit,
    /// Barrier synchronization
    Bar,
    /// Memory fence
    MemBar,

    // ===== Texture/Surface =====
    /// Texture load
    Tex,
    /// Surface load
    Suld,
    /// Surface store
    Sust,

    // ===== Atomic =====
    /// Atomic add
    AtomAdd,
    /// Atomic min
    AtomMin,
    /// Atomic max
    AtomMax,
    /// Atomic exchange
    AtomExch,
    /// Atomic compare-and-swap
    AtomCas,

    // ===== Tensor Core (WMMA) =====
    /// WMMA load matrix A fragment
    WmmaLoadA,
    /// WMMA load matrix B fragment
    WmmaLoadB,
    /// WMMA load accumulator fragment
    WmmaLoadC,
    /// WMMA matrix multiply-accumulate
    WmmaMma,
    /// WMMA store accumulator
    WmmaStoreD,

    // ===== Tensor Core (MMA.sync — SM 8.0+) =====
    /// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    /// Smaller fragments than WMMA (16×8 output vs 16×16), higher IPC.
    /// Per thread: A=4 regs (8 FP16), B=2 regs (4 FP16), C/D=4 regs (4 FP32).
    MmaSync,

    // ===== Matrix Load (ldmatrix — SM 7.5+) =====
    /// ldmatrix.sync.aligned.m8n8.x4.shared.b16
    /// Loads 4 8×8 FP16 matrices from shared memory in one instruction.
    /// Each thread loads one row (8 FP16). Replaces ~16 ld.shared instructions.
    LdMatrix,
    /// ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
    /// Transposed variant: loads 2 8×8 matrices with transpose.
    /// Used for B fragment of mma.sync.row.col (B stored row-major, needed col-major).
    LdMatrixTrans,

    // ===== DP4A Integer Dot Product =====
    /// DP4A unsigned dot product (4 u8 pairs to u32)
    Dp4a,
    /// DP4A unsigned-signed dot product
    Dp4aUS,
    /// DP4A signed dot product (4 s8 pairs to s32)
    Dp4aS32,

    // ===== Bit Manipulation =====
    /// Population count (count set bits)
    Popc,
    /// Count leading zeros
    Clz,
    /// Find highest set bit position
    Bfind,
    /// Bit field extract
    Bfe,
    /// Bit field insert
    Bfi,

    // ===== Special Loads =====
    /// Load volatile (bypass cache)
    LdVolatile,

    // ===== Cache Hints =====
    /// Prefetch global memory to L2 cache (no-fault hint)
    Prefetch,

    // ===== Async Copy (SM 8.0+) =====
    /// cp.async: asynchronous global→shared copy without registers.
    /// Copies `size` bytes (4, 8, or 16) from global to shared memory.
    /// Does not consume registers — data goes directly to shared memory.
    CpAsync,
    /// cp.async.commit_group: commit all outstanding cp.async operations
    CpAsyncCommitGroup,
    /// cp.async.wait_group N: wait until at most N groups are outstanding
    CpAsyncWaitGroup,

    // ===== Warp Vote =====
    /// Generic warp vote (ballot)
    Vote,
}

/// Comparison operators for setp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
    /// Less than (unsigned)
    Lo,
    /// Less than or equal (unsigned)
    Ls,
    /// Greater than (unsigned)
    Hi,
    /// Greater than or equal (unsigned)
    Hs,
}

impl CmpOp {
    /// Convert to PTX string
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Lo => "lo",
            Self::Ls => "ls",
            Self::Hi => "hi",
            Self::Hs => "hs",
        }
    }
}

/// WMMA matrix layout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WmmaLayout {
    /// Row-major layout
    #[default]
    RowMajor,
    /// Column-major layout
    ColMajor,
}

impl WmmaLayout {
    /// Convert to PTX string
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::RowMajor => "row",
            Self::ColMajor => "col",
        }
    }
}

/// WMMA shape configuration (M x N x K tile size)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WmmaShape {
    /// M dimension (rows of A, rows of C/D)
    pub m: u32,
    /// N dimension (cols of B, cols of C/D)
    pub n: u32,
    /// K dimension (cols of A, rows of B)
    pub k: u32,
}

impl WmmaShape {
    /// Standard 16x16x16 tensor core tile
    pub const M16N16K16: Self = Self { m: 16, n: 16, k: 16 };
    /// 8x32x16 tile for different aspect ratios
    pub const M8N32K16: Self = Self { m: 8, n: 32, k: 16 };
    /// 32x8x16 tile for different aspect ratios
    pub const M32N8K16: Self = Self { m: 32, n: 8, k: 16 };

    /// Convert to PTX shape string
    #[must_use]
    pub fn to_ptx_string(self) -> String {
        format!("m{}n{}k{}", self.m, self.n, self.k)
    }
}

/// Rounding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (default)
    #[default]
    Rn,
    /// Round toward zero
    Rz,
    /// Round toward positive infinity
    Rp,
    /// Round toward negative infinity
    Rm,
    /// Round to nearest integer
    Rni,
    /// Round to integer toward zero (truncate)
    Rzi,
    /// Round to integer toward positive infinity (ceil)
    Rpi,
    /// Round to integer toward minus infinity (floor)
    Rmi,
}

impl RoundingMode {
    /// Convert to PTX string
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::Rn => ".rn",
            Self::Rz => ".rz",
            Self::Rp => ".rp",
            Self::Rm => ".rm",
            Self::Rni => ".rni",
            Self::Rzi => ".rzi",
            Self::Rpi => ".rpi",
            Self::Rmi => ".rmi",
        }
    }
}

/// A single PTX instruction
#[derive(Debug, Clone)]
pub struct PtxInstruction {
    /// Operation
    pub op: PtxOp,
    /// Data type (destination type for cvt)
    pub ty: PtxType,
    /// Source type (for cvt instructions with different src/dst types)
    pub src_type: Option<PtxType>,
    /// Destination register (if any)
    pub dst: Option<Operand>,
    /// Multiple destination registers (for vector loads like ld.v4.f32)
    pub dsts: Vec<Operand>,
    /// Source operands
    pub srcs: Vec<Operand>,
    /// Predicate guard (optional)
    pub predicate: Option<Predicate>,
    /// State space (for memory ops)
    pub state_space: Option<PtxStateSpace>,
    /// Rounding mode (for FP ops)
    pub rounding: Option<RoundingMode>,
    /// Label (for branch targets)
    pub label: Option<String>,
}

/// Instruction operand
#[derive(Debug, Clone)]
pub enum Operand {
    /// Virtual register
    Reg(VirtualReg),
    /// Special register
    SpecialReg(super::registers::PtxReg),
    /// Immediate integer
    ImmI64(i64),
    /// Immediate unsigned
    ImmU64(u64),
    /// Immediate float
    ImmF32(f32),
    /// Immediate double
    ImmF64(f64),
    /// Parameter name
    Param(String),
    /// Memory address (base + offset)
    Addr {
        /// Base register
        base: VirtualReg,
        /// Offset in bytes
        offset: i32,
    },
    /// Label reference
    Label(String),
}

/// Predicate for conditional execution
#[derive(Debug, Clone)]
pub struct Predicate {
    /// Predicate register
    pub reg: VirtualReg,
    /// Negated?
    pub negated: bool,
}

impl PtxInstruction {
    /// Create a new instruction
    #[must_use]
    pub fn new(op: PtxOp, ty: PtxType) -> Self {
        Self {
            op,
            ty,
            src_type: None,
            dst: None,
            dsts: Vec::new(),
            srcs: Vec::new(),
            predicate: None,
            state_space: None,
            rounding: None,
            label: None,
        }
    }

    /// Set source type (for cvt instructions with different src/dst types)
    #[must_use]
    pub fn with_src_type(mut self, src_type: PtxType) -> Self {
        self.src_type = Some(src_type);
        self
    }

    /// Set destination
    #[must_use]
    pub fn dst(mut self, dst: Operand) -> Self {
        // For vector types, push to dsts instead
        if matches!(self.ty, PtxType::V2F32 | PtxType::V4F32) {
            self.dsts.push(dst);
        } else {
            self.dst = Some(dst);
        }
        self
    }

    /// Push a destination operand to the vector destination list
    /// Used for WMMA instructions that always have multiple destinations
    #[must_use]
    pub fn push_dst(mut self, dst: Operand) -> Self {
        self.dsts.push(dst);
        self
    }

    /// Add source operand
    #[must_use]
    pub fn src(mut self, src: Operand) -> Self {
        self.srcs.push(src);
        self
    }

    /// Set predicate guard
    #[must_use]
    pub fn predicated(mut self, pred: Predicate) -> Self {
        self.predicate = Some(pred);
        self
    }

    /// Set state space
    #[must_use]
    pub fn space(mut self, space: PtxStateSpace) -> Self {
        self.state_space = Some(space);
        self
    }

    /// Set rounding mode
    #[must_use]
    pub fn rounding(mut self, mode: RoundingMode) -> Self {
        self.rounding = Some(mode);
        self
    }

    /// Set label
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

#[cfg(test)]
mod tests;
