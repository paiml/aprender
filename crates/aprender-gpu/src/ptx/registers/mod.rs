//! PTX Register Management
//!
//! Provides register allocation with liveness analysis to prevent spills (Muda).

use super::types::PtxType;
use std::collections::HashMap;
use std::fmt;

/// Special PTX registers (read-only hardware registers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxReg {
    /// Thread ID X dimension
    TidX,
    /// Thread ID Y dimension
    TidY,
    /// Thread ID Z dimension
    TidZ,
    /// Block ID X dimension (CTA = Cooperative Thread Array)
    CtaIdX,
    /// Block ID Y dimension
    CtaIdY,
    /// Block ID Z dimension
    CtaIdZ,
    /// Block dimension X (threads per block)
    NtidX,
    /// Block dimension Y
    NtidY,
    /// Block dimension Z
    NtidZ,
    /// Grid dimension X (blocks per grid)
    NctaIdX,
    /// Grid dimension Y
    NctaIdY,
    /// Grid dimension Z
    NctaIdZ,
    /// Warp ID within block
    WarpId,
    /// Lane ID within warp (0-31)
    LaneId,
    /// SM ID (multiprocessor)
    SmId,
    /// Clock counter (low 32 bits)
    Clock,
    /// Clock counter (64 bits)
    Clock64,
}

impl PtxReg {
    /// Convert to PTX string representation
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::TidX => "%tid.x",
            Self::TidY => "%tid.y",
            Self::TidZ => "%tid.z",
            Self::CtaIdX => "%ctaid.x",
            Self::CtaIdY => "%ctaid.y",
            Self::CtaIdZ => "%ctaid.z",
            Self::NtidX => "%ntid.x",
            Self::NtidY => "%ntid.y",
            Self::NtidZ => "%ntid.z",
            Self::NctaIdX => "%nctaid.x",
            Self::NctaIdY => "%nctaid.y",
            Self::NctaIdZ => "%nctaid.z",
            Self::WarpId => "%warpid",
            Self::LaneId => "%laneid",
            Self::SmId => "%smid",
            Self::Clock => "%clock",
            Self::Clock64 => "%clock64",
        }
    }

    /// Get the data type of this special register
    #[must_use]
    pub const fn data_type(self) -> PtxType {
        match self {
            Self::Clock64 => PtxType::U64,
            _ => PtxType::U32,
        }
    }
}

/// Virtual register (pre-allocation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualReg {
    id: u32,
    ty: PtxType,
}

impl VirtualReg {
    /// Create a new virtual register
    #[must_use]
    pub const fn new(id: u32, ty: PtxType) -> Self {
        Self { id, ty }
    }

    /// Get register ID
    #[must_use]
    pub const fn id(self) -> u32 {
        self.id
    }

    /// Get register type
    #[must_use]
    pub const fn ty(self) -> PtxType {
        self.ty
    }

    /// Convert to PTX string
    #[must_use]
    pub fn to_ptx_string(self) -> String {
        format!("{}{}", self.ty.register_prefix(), self.id)
    }

    /// Write PTX string representation to a buffer (zero-allocation)
    ///
    /// This is more efficient than `to_ptx_string()` when building large PTX
    /// output as it avoids intermediate String allocations.
    #[inline]
    pub fn write_to<W: fmt::Write>(self, w: &mut W) -> fmt::Result {
        write!(w, "{}{}", self.ty.register_prefix(), self.id)
    }
}

impl fmt::Display for VirtualReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.ty.register_prefix(), self.id)
    }
}

/// Physical register (post-allocation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalReg(pub u32);

/// Live range for register allocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRange {
    /// Start instruction index
    pub start: usize,
    /// End instruction index (exclusive)
    pub end: usize,
}

impl LiveRange {
    /// Create a new live range
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Check if this range overlaps with another
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// Register pressure report
#[derive(Debug, Clone, PartialEq)]
pub struct RegisterPressure {
    /// Maximum simultaneous live registers
    pub max_live: usize,
    /// Number of spills to local memory
    pub spill_count: usize,
    /// Register utilization (0.0-1.0)
    pub utilization: f64,
}

/// Register allocator with liveness analysis
/// Per Xiao et al. [47] - prevents register spills
#[derive(Debug, Clone)]
pub struct RegisterAllocator {
    /// Per-type register counters (each type has its own namespace via unique prefix)
    type_counters: HashMap<PtxType, u32>,
    /// Live ranges for each virtual register (key is (type, id))
    live_ranges: HashMap<(PtxType, u32), LiveRange>,
    /// Allocated virtual registers by type
    allocated: Vec<VirtualReg>,
    /// Current instruction index
    current_instruction: usize,
    /// Spill count (should be zero - Muda)
    spill_count: usize,
}

impl RegisterAllocator {
    /// Create a new register allocator
    #[must_use]
    pub fn new() -> Self {
        Self {
            type_counters: HashMap::new(),
            live_ranges: HashMap::new(),
            allocated: Vec::new(),
            current_instruction: 0,
            spill_count: 0,
        }
    }

    /// Allocate a new virtual register with per-prefix ID
    ///
    /// Per-prefix counters (NOT per-type) because some types share prefixes:
    /// U64+B64 → %rd, U32+B32 share none (U32=%r, B32=%rb), U8+B8 → %rs, etc.
    /// Using per-type IDs caused duplicate register names (e.g., both U64 and B64
    /// allocating `%rd0`), breaking ptxas.
    pub fn allocate_virtual(&mut self, ty: PtxType) -> VirtualReg {
        // Look up existing counter by matching register prefix
        // (find any existing type that shares this prefix)
        let prefix = ty.register_prefix();
        let existing_ty_with_prefix =
            self.type_counters.keys().find(|t| t.register_prefix() == prefix).copied();

        let lookup_ty = existing_ty_with_prefix.unwrap_or(ty);
        let id = *self.type_counters.get(&lookup_ty).unwrap_or(&0);
        self.type_counters.insert(lookup_ty, id + 1);

        let vreg = VirtualReg::new(id, ty);
        self.allocated.push(vreg);

        // Start live range at current instruction
        self.live_ranges.insert(
            (ty, id),
            LiveRange::new(self.current_instruction, self.current_instruction + 1),
        );

        vreg
    }

    /// Extend the live range of a register to current instruction
    pub fn extend_live_range(&mut self, vreg: VirtualReg) {
        if let Some(range) = self.live_ranges.get_mut(&(vreg.ty(), vreg.id())) {
            range.end = self.current_instruction + 1;
        }
    }

    /// Advance to next instruction
    pub fn next_instruction(&mut self) {
        self.current_instruction += 1;
    }

    /// Get register pressure report
    #[must_use]
    pub fn pressure_report(&self) -> RegisterPressure {
        // Calculate max simultaneous live registers
        let max_live = self.allocated.len(); // Simplified - actual would check overlaps

        RegisterPressure {
            max_live,
            spill_count: self.spill_count,
            utilization: max_live as f64 / 256.0, // Max 256 registers per thread
        }
    }

    /// Generate register declarations for PTX
    ///
    /// Each type now has a unique register prefix (U32 → %r, S32 → %ri, etc.)
    /// so simple per-type grouping is sufficient — no prefix collisions.
    #[must_use]
    pub fn emit_declarations(&self) -> String {
        let mut decls = String::new();

        // Group by register prefix (u64 and b64 share %rd, etc.)
        // BUG FIX: previously grouped by PtxType → duplicate .reg lines for
        // types sharing a prefix (U64+B64→%rd, U8+B8→%rs, etc.).
        // Now groups by prefix; declaration type is taken from first type in group.
        let mut by_prefix: HashMap<&'static str, (PtxType, Vec<&VirtualReg>)> = HashMap::new();
        for vreg in &self.allocated {
            let prefix = vreg.ty().register_prefix();
            by_prefix.entry(prefix).or_insert_with(|| (vreg.ty(), Vec::new())).1.push(vreg);
        }

        // Emit declarations
        // NOTE: PTX only supports register declarations of 16-bit or wider types.
        // 8-bit types (.u8, .s8, .b8) must be widened to their 16-bit equivalent.
        // Values are automatically zero/sign-extended to 16 bits by the hardware.
        for (prefix, (ty, regs)) in by_prefix {
            if !regs.is_empty() {
                let count = regs.len();
                let decl_type = ty.register_declaration_type();
                decls.push_str(&format!("    .reg {}  {}<{}>;\n", decl_type, prefix, count));
            }
        }

        decls
    }
}

impl Default for RegisterAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
