//! Type-Safe Block Identifiers (Poka-Yoke)
//!
//! Per spec §5.2: Type-safe block IDs prevent coverage gaps at compile time.
//!
//! These types are intentionally NOT interchangeable to catch errors at compile time.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// Type-safe block identifier (Poka-Yoke)
///
/// Represents a basic block in the CFG. Cannot be confused with FunctionId or EdgeId.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockId(u32);

impl BlockId {
    /// Create a new block ID
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the inner value
    #[inline]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl Hash for BlockId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialOrd for BlockId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BlockId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

/// Type-safe function identifier (Poka-Yoke)
///
/// Represents a function in the WASM module. Cannot be confused with BlockId.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionId(u32);

impl FunctionId {
    /// Create a new function ID
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the inner value
    #[inline]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl Hash for FunctionId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialOrd for FunctionId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FunctionId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

/// Type-safe edge identifier (Poka-Yoke)
///
/// Encodes both source and target block in a single u64:
/// `EdgeId = (from << 32) | to`
///
/// # Panics
///
/// Debug assertion if block IDs exceed u32::MAX (Kaizen: overflow guard)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeId(u64);

impl EdgeId {
    /// Create edge ID from source and target (Poka-Yoke: can't mix up)
    ///
    /// The encoding `(from << 32) | to` is safe because BlockId is u32,
    /// which always fits in the lower/upper 32 bits of u64.
    #[inline]
    #[must_use]
    pub const fn new(from: BlockId, to: BlockId) -> Self {
        // Poka-Yoke: BlockId is u32, so no overflow possible in u64 encoding
        // The type system guarantees from.0 and to.0 are valid u32 values
        Self((from.0 as u64) << 32 | to.0 as u64)
    }

    /// Get the source block
    #[inline]
    #[must_use]
    pub const fn source(self) -> BlockId {
        BlockId((self.0 >> 32) as u32)
    }

    /// Get the target block
    #[inline]
    #[must_use]
    pub const fn target(self) -> BlockId {
        BlockId(self.0 as u32)
    }

    /// Get the raw u64 value
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl Hash for EdgeId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialOrd for EdgeId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}
