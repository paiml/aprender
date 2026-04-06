//! PTX Type System
//!
//! Defines PTX data types and state spaces.

use std::fmt;

/// PTX data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxType {
    /// Predicate (1-bit boolean)
    Pred,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 8-bit signed integer
    S8,
    /// 16-bit signed integer
    S16,
    /// 32-bit signed integer
    S32,
    /// 64-bit signed integer
    S64,
    /// 16-bit floating point (half precision)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 32-bit floating point (single precision)
    F32,
    /// 64-bit floating point (double precision)
    F64,
    /// 8-bit untyped (for byte operations)
    B8,
    /// 16-bit untyped
    B16,
    /// 32-bit untyped
    B32,
    /// 64-bit untyped
    B64,
    /// Vector of 2 x f32 (for vectorized loads)
    V2F32,
    /// Vector of 4 x f32 (for vectorized loads)
    V4F32,
}

impl PtxType {
    /// Get size in bytes
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Pred | Self::U8 | Self::S8 | Self::B8 => 1,
            Self::U16 | Self::S16 | Self::F16 | Self::BF16 | Self::B16 => 2,
            Self::U32 | Self::S32 | Self::F32 | Self::B32 => 4,
            Self::U64 | Self::S64 | Self::F64 | Self::B64 | Self::V2F32 => 8,
            Self::V4F32 => 16,
        }
    }

    /// Get size in bits
    #[must_use]
    pub const fn size_bits(self) -> usize {
        self.size_bytes() * 8
    }

    /// Convert to PTX string representation
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::Pred => ".pred",
            Self::U8 => ".u8",
            Self::U16 => ".u16",
            Self::U32 => ".u32",
            Self::U64 => ".u64",
            Self::S8 => ".s8",
            Self::S16 => ".s16",
            Self::S32 => ".s32",
            Self::S64 => ".s64",
            Self::F16 => ".f16",
            Self::BF16 => ".bf16",
            Self::F32 => ".f32",
            Self::F64 => ".f64",
            Self::B8 => ".b8",
            Self::B16 => ".b16",
            Self::B32 => ".b32",
            Self::B64 => ".b64",
            Self::V2F32 => ".v2.f32",
            Self::V4F32 => ".v4.f32",
        }
    }

    /// Check if this is a floating point type
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64 | Self::V2F32 | Self::V4F32)
    }

    /// Check if this is a signed integer type
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::S8 | Self::S16 | Self::S32 | Self::S64)
    }

    /// Check if this is an unsigned integer type
    #[must_use]
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    /// Get the register prefix for this type
    ///
    /// PTX register conventions:
    /// - %p: predicates
    /// - %r: 32-bit unsigned integer (u32)
    /// - %ri: 32-bit signed integer (s32) — MUST be separate from %r
    /// - %rd: 64-bit integer (u64, s64)
    /// - %h: 16-bit half/bfloat
    /// - %f: 32-bit float
    /// - %fd: 64-bit double
    ///
    /// CRITICAL: U32 and S32 MUST use different prefixes. PTX requires that
    /// register declarations and instruction types match. `.reg .u32 %r<N>`
    /// declares u32 registers; using `mov.s32 %r0, ...` on them is INVALID.
    /// Fix for CUDA_ERROR_INVALID_PTX in Q8QuantizeKernel (Refs GH-219).
    /// Get the PTX type string for register declarations.
    ///
    /// PTX only supports register declarations of 16-bit or wider types.
    /// 8-bit types (.u8, .s8, .b8) are widened to their 16-bit equivalent.
    /// Per PTX ISA: "8-bit data are stored in 16-bit registers."
    #[must_use]
    pub const fn register_declaration_type(self) -> &'static str {
        match self {
            Self::U8 => ".u16", // 8-bit → 16-bit for register declaration
            Self::S8 => ".s16",
            Self::B8 => ".b16",
            _ => self.to_ptx_string(),
        }
    }

    /// Get the register prefix for this type
    #[must_use]
    pub const fn register_prefix(self) -> &'static str {
        match self {
            Self::Pred => "%p",
            Self::U8 | Self::B8 => "%rs",   // 8-bit unsigned/bitfield
            Self::S8 => "%rsi",             // 8-bit signed
            Self::U16 | Self::B16 => "%rh", // 16-bit unsigned/bitfield
            Self::S16 => "%rhi",            // 16-bit signed
            Self::U32 => "%r",              // 32-bit unsigned
            Self::S32 => "%ri",             // 32-bit signed (separate from %r!)
            Self::B32 => "%rb",             // 32-bit bitfield (WMMA fragments)
            Self::U64 | Self::B64 => "%rd", // 64-bit unsigned/bitfield
            Self::S64 => "%rdi",            // 64-bit signed
            Self::F16 | Self::BF16 => "%h",
            Self::F32 | Self::V2F32 | Self::V4F32 => "%f", // f32 and vector f32 use %f registers
            Self::F64 => "%fd",
        }
    }
}

impl fmt::Display for PtxType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_ptx_string())
    }
}

/// PTX state spaces (memory hierarchy)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxStateSpace {
    /// Register (fastest, per-thread)
    Reg,
    /// Shared memory (fast, per-block, 48KB-164KB)
    Shared,
    /// Global memory (slow, device-wide, GBs)
    Global,
    /// Local memory (slow, per-thread spill)
    Local,
    /// Constant memory (cached, read-only, 64KB)
    Const,
    /// Texture memory (cached, read-only, spatial locality)
    Tex,
    /// Parameter space (kernel arguments)
    Param,
}

impl PtxStateSpace {
    /// Convert to PTX string
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::Reg => ".reg",
            Self::Shared => ".shared",
            Self::Global => ".global",
            Self::Local => ".local",
            Self::Const => ".const",
            Self::Tex => ".tex",
            Self::Param => ".param",
        }
    }

    /// Check if this state space is cached
    #[must_use]
    pub const fn is_cached(self) -> bool {
        matches!(self, Self::Const | Self::Tex)
    }

    /// Check if this state space is per-thread
    #[must_use]
    pub const fn is_per_thread(self) -> bool {
        matches!(self, Self::Reg | Self::Local)
    }
}

impl fmt::Display for PtxStateSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_ptx_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_sizes() {
        assert_eq!(PtxType::Pred.size_bytes(), 1);
        assert_eq!(PtxType::U8.size_bytes(), 1);
        assert_eq!(PtxType::U16.size_bytes(), 2);
        assert_eq!(PtxType::U32.size_bytes(), 4);
        assert_eq!(PtxType::U64.size_bytes(), 8);
        assert_eq!(PtxType::F16.size_bytes(), 2);
        assert_eq!(PtxType::F32.size_bytes(), 4);
        assert_eq!(PtxType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_type_bits() {
        assert_eq!(PtxType::U8.size_bits(), 8);
        assert_eq!(PtxType::U32.size_bits(), 32);
        assert_eq!(PtxType::U64.size_bits(), 64);
    }

    #[test]
    fn test_float_detection() {
        assert!(PtxType::F16.is_float());
        assert!(PtxType::F32.is_float());
        assert!(PtxType::F64.is_float());
        assert!(PtxType::BF16.is_float());
        assert!(!PtxType::U32.is_float());
        assert!(!PtxType::S32.is_float());
    }

    #[test]
    fn test_signed_detection() {
        assert!(PtxType::S8.is_signed());
        assert!(PtxType::S32.is_signed());
        assert!(!PtxType::U32.is_signed());
        assert!(!PtxType::F32.is_signed());
    }

    #[test]
    fn test_state_space_strings() {
        assert_eq!(PtxStateSpace::Global.to_ptx_string(), ".global");
        assert_eq!(PtxStateSpace::Shared.to_ptx_string(), ".shared");
        assert_eq!(PtxStateSpace::Reg.to_ptx_string(), ".reg");
    }

    #[test]
    fn test_display_impl() {
        assert_eq!(format!("{}", PtxType::F32), ".f32");
        assert_eq!(format!("{}", PtxStateSpace::Global), ".global");
    }

    #[test]
    fn test_unsigned_detection() {
        assert!(PtxType::U8.is_unsigned());
        assert!(PtxType::U16.is_unsigned());
        assert!(PtxType::U32.is_unsigned());
        assert!(PtxType::U64.is_unsigned());
        assert!(!PtxType::S32.is_unsigned());
        assert!(!PtxType::F32.is_unsigned());
    }

    #[test]
    fn test_state_space_cached() {
        assert!(PtxStateSpace::Const.is_cached());
        assert!(PtxStateSpace::Tex.is_cached());
        assert!(!PtxStateSpace::Global.is_cached());
        assert!(!PtxStateSpace::Shared.is_cached());
    }

    #[test]
    fn test_state_space_per_thread() {
        assert!(PtxStateSpace::Reg.is_per_thread());
        assert!(PtxStateSpace::Local.is_per_thread());
        assert!(!PtxStateSpace::Global.is_per_thread());
        assert!(!PtxStateSpace::Shared.is_per_thread());
    }

    #[test]
    fn test_register_prefix() {
        assert_eq!(PtxType::Pred.register_prefix(), "%p");
        assert_eq!(PtxType::F16.register_prefix(), "%h");
        assert_eq!(PtxType::BF16.register_prefix(), "%h");
        assert_eq!(PtxType::F32.register_prefix(), "%f");
        assert_eq!(PtxType::F64.register_prefix(), "%fd");
        assert_eq!(PtxType::U32.register_prefix(), "%r");
        assert_eq!(PtxType::S32.register_prefix(), "%ri"); // Separate from %r (fix for CUDA_ERROR_INVALID_PTX)
        assert_eq!(PtxType::U64.register_prefix(), "%rd");
        assert_eq!(PtxType::S64.register_prefix(), "%rdi"); // Separate from %rd
        assert_eq!(PtxType::U8.register_prefix(), "%rs");
        assert_eq!(PtxType::S8.register_prefix(), "%rsi"); // Separate from %rs
        assert_eq!(PtxType::U16.register_prefix(), "%rh");
        assert_eq!(PtxType::S16.register_prefix(), "%rhi"); // Separate from %rh
    }

    #[test]
    fn test_all_type_strings() {
        // Test all PTX type strings
        assert_eq!(PtxType::Pred.to_ptx_string(), ".pred");
        assert_eq!(PtxType::S8.to_ptx_string(), ".s8");
        assert_eq!(PtxType::S16.to_ptx_string(), ".s16");
        assert_eq!(PtxType::S64.to_ptx_string(), ".s64");
        assert_eq!(PtxType::B8.to_ptx_string(), ".b8");
        assert_eq!(PtxType::B16.to_ptx_string(), ".b16");
        assert_eq!(PtxType::B32.to_ptx_string(), ".b32");
        assert_eq!(PtxType::B64.to_ptx_string(), ".b64");
        assert_eq!(PtxType::BF16.to_ptx_string(), ".bf16");
    }

    #[test]
    fn test_all_state_space_strings() {
        assert_eq!(PtxStateSpace::Local.to_ptx_string(), ".local");
        assert_eq!(PtxStateSpace::Param.to_ptx_string(), ".param");
        assert_eq!(PtxStateSpace::Tex.to_ptx_string(), ".tex");
        assert_eq!(PtxStateSpace::Const.to_ptx_string(), ".const");
    }

    #[test]
    fn test_state_space_display() {
        assert_eq!(format!("{}", PtxStateSpace::Shared), ".shared");
        assert_eq!(format!("{}", PtxStateSpace::Reg), ".reg");
        assert_eq!(format!("{}", PtxStateSpace::Local), ".local");
    }

    #[test]
    fn test_byte_type_sizes() {
        assert_eq!(PtxType::B8.size_bytes(), 1);
        assert_eq!(PtxType::B16.size_bytes(), 2);
        assert_eq!(PtxType::B32.size_bytes(), 4);
        assert_eq!(PtxType::B64.size_bytes(), 8);
        assert_eq!(PtxType::S8.size_bytes(), 1);
        assert_eq!(PtxType::S16.size_bytes(), 2);
    }

    #[test]
    fn test_vector_types() {
        // V2F32: 2 x f32 = 8 bytes
        assert_eq!(PtxType::V2F32.size_bytes(), 8);
        assert_eq!(PtxType::V2F32.size_bits(), 64);
        assert!(PtxType::V2F32.is_float());
        assert!(!PtxType::V2F32.is_signed());
        assert!(!PtxType::V2F32.is_unsigned());
        assert_eq!(PtxType::V2F32.register_prefix(), "%f");
        assert_eq!(PtxType::V2F32.to_ptx_string(), ".v2.f32");

        // V4F32: 4 x f32 = 16 bytes
        assert_eq!(PtxType::V4F32.size_bytes(), 16);
        assert_eq!(PtxType::V4F32.size_bits(), 128);
        assert!(PtxType::V4F32.is_float());
        assert!(!PtxType::V4F32.is_signed());
        assert!(!PtxType::V4F32.is_unsigned());
        assert_eq!(PtxType::V4F32.register_prefix(), "%f");
        assert_eq!(PtxType::V4F32.to_ptx_string(), ".v4.f32");
    }

    #[test]
    fn test_b32_register_prefix() {
        // B32 uses special %rb prefix for WMMA fragments
        assert_eq!(PtxType::B32.register_prefix(), "%rb");
    }

    #[test]
    fn test_type_display_all() {
        // Test Display for all types
        assert_eq!(format!("{}", PtxType::Pred), ".pred");
        assert_eq!(format!("{}", PtxType::V2F32), ".v2.f32");
        assert_eq!(format!("{}", PtxType::V4F32), ".v4.f32");
        assert_eq!(format!("{}", PtxType::B32), ".b32");
    }
}
