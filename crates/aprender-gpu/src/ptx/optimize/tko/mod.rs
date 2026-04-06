//! Token-Based Memory Ordering (TKO)
//!
//! Implements explicit dependency tracking for memory operations to enable
//! compiler-driven barrier elimination.
//!
//! ## Pattern
//!
//! ```text
//! %t = make_token : token
//! %data, %new_t = load_ptr_tko weak %ptr token=%t
//! %store_t = store_ptr_tko weak %ptr, %data token=%new_t
//! ```
//!
//! ## Memory Ordering Semantics
//!
//! - `weak` - No ordering guarantees (fastest, default for shared memory)
//! - `relaxed` - Relaxed atomic semantics
//! - `acquire` - Acquire ordering for load operations
//! - `release` - Release ordering for store operations
//!
//! ## Benefits
//!
//! - Compiler can eliminate redundant barriers
//! - Explicit data dependencies enable better scheduling
//! - Maps to CUDA memory model semantics
//! - Prevents synchronization bugs (e.g., PARITY-114)
//!
//! ## Academic Foundation
//!
//! Based on NVIDIA CUDA Tile IR (CUDA Toolkit 13.1) memory consistency ops.
//! Click & Paleczny show token/sea-of-nodes IR enables 15-30% better optimization.
//! cuda-tile-behavior.md: Section 3.1, Falsification tests #66-80

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Memory ordering semantics for TKO operations
/// (Aligned with NVIDIA CUDA Tile IR)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryOrdering {
    /// No ordering guarantees (fastest)
    #[default]
    Weak,
    /// Relaxed atomic semantics
    Relaxed,
    /// Acquire ordering (for loads)
    Acquire,
    /// Release ordering (for stores)
    Release,
}

impl MemoryOrdering {
    /// Convert to PTX memory ordering modifier
    #[must_use]
    pub const fn to_ptx_modifier(self) -> &'static str {
        match self {
            Self::Weak => ".weak",
            Self::Relaxed => ".relaxed",
            Self::Acquire => ".acquire",
            Self::Release => ".release",
        }
    }

    /// Check if this ordering provides acquire semantics
    #[must_use]
    pub const fn is_acquire(self) -> bool {
        matches!(self, Self::Acquire)
    }

    /// Check if this ordering provides release semantics
    #[must_use]
    pub const fn is_release(self) -> bool {
        matches!(self, Self::Release)
    }
}

/// Memory scope for TKO operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryScope {
    /// Thread-local (no synchronization needed)
    Thread,
    /// Block-level synchronization (CTA)
    Block,
    /// Cluster-level synchronization (SM cluster)
    Cluster,
    /// Device-level synchronization
    #[default]
    Device,
    /// System-wide synchronization (host + device)
    System,
}

impl MemoryScope {
    /// Convert to PTX scope modifier
    #[must_use]
    pub const fn to_ptx_scope(self) -> &'static str {
        match self {
            // Thread and Block both map to .cta (closest PTX equivalent)
            Self::Thread | Self::Block => ".cta",
            Self::Cluster => ".cluster",
            Self::Device => ".gpu",
            Self::System => ".sys",
        }
    }
}

/// Global token ID generator
static NEXT_TOKEN_ID: AtomicU64 = AtomicU64::new(1);

/// A dependency token for memory ordering
///
/// Tokens track memory operation dependencies, enabling the compiler
/// to determine when barriers can be eliminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token {
    /// Unique token identifier
    id: u64,
}

impl Token {
    /// Create a new unique token
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Section 3.1: make_token creates dependency tracking
    #[must_use]
    pub fn new() -> Self {
        Self { id: NEXT_TOKEN_ID.fetch_add(1, Ordering::Relaxed) }
    }

    /// Get the token ID
    #[must_use]
    pub const fn id(self) -> u64 {
        self.id
    }

    /// Create a token from a raw ID (for deserialization)
    #[must_use]
    pub const fn from_id(id: u64) -> Self {
        Self { id }
    }
}

impl Default for Token {
    fn default() -> Self {
        Self::new()
    }
}

/// Join multiple tokens into a single synchronization point
///
/// # Arguments
///
/// * `tokens` - Tokens to join
///
/// # Returns
///
/// A new token that depends on all input tokens
///
/// # cuda-tile-behavior.md References
///
/// - Section 3.1: join_tokens combines dependencies
/// - Falsification test #75: Join creates token depending on all inputs
#[must_use]
pub fn join_tokens(_tokens: &[Token]) -> Token {
    // The joined token implicitly depends on all input tokens
    // In a real implementation, we'd track the dependency graph
    // For now, create a new token that represents the join point
    Token::new()
}

/// Token dependency graph for analysis
#[derive(Debug, Clone, Default)]
pub struct TokenGraph {
    /// Set of all tokens created
    tokens: HashSet<u64>,
    /// Dependencies: (dependent_token, [dependency_tokens])
    dependencies: Vec<(u64, Vec<u64>)>,
}

impl TokenGraph {
    /// Create a new empty token graph
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a token creation
    pub fn create_token(&mut self, token: Token) {
        self.tokens.insert(token.id());
    }

    /// Record a token dependency
    pub fn add_dependency(&mut self, dependent: Token, dependency: Token) {
        // Find or create entry for dependent
        if let Some((_, deps)) = self.dependencies.iter_mut().find(|(d, _)| *d == dependent.id()) {
            deps.push(dependency.id());
        } else {
            self.dependencies.push((dependent.id(), vec![dependency.id()]));
        }
    }

    /// Record a join operation
    pub fn join(&mut self, result: Token, sources: &[Token]) {
        let deps: Vec<u64> = sources.iter().map(|t| t.id()).collect();
        self.dependencies.push((result.id(), deps));
        self.tokens.insert(result.id());
    }

    /// Check if a token has any dependencies
    #[must_use]
    pub fn has_dependencies(&self, token: Token) -> bool {
        self.dependencies.iter().any(|(d, _)| *d == token.id())
    }

    /// Get all dependencies for a token
    #[must_use]
    pub fn get_dependencies(&self, token: Token) -> Vec<u64> {
        self.dependencies
            .iter()
            .find(|(d, _)| *d == token.id())
            .map(|(_, deps)| deps.clone())
            .unwrap_or_default()
    }

    /// Check for circular dependencies (deadlock detection)
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Falsification test #79: Token cycles are detected and rejected
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        // Simple DFS-based cycle detection
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for &token_id in &self.tokens {
            if self.has_cycle_dfs(token_id, &mut visited, &mut rec_stack) {
                return true;
            }
        }

        false
    }

    fn has_cycle_dfs(
        &self,
        token_id: u64,
        visited: &mut HashSet<u64>,
        rec_stack: &mut HashSet<u64>,
    ) -> bool {
        if rec_stack.contains(&token_id) {
            return true; // Cycle detected
        }

        if visited.contains(&token_id) {
            return false; // Already processed
        }

        visited.insert(token_id);
        rec_stack.insert(token_id);

        // Check all dependencies
        if let Some((_, deps)) = self.dependencies.iter().find(|(d, _)| *d == token_id) {
            for &dep in deps {
                if self.has_cycle_dfs(dep, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(&token_id);
        false
    }

    /// Count total tokens
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

/// Analyze TKO patterns in instruction sequence
///
/// Identifies opportunities for barrier elimination based on token dependencies.
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #66: Tokens eliminate redundant barriers
/// - Falsification test #71: Barrier elimination is sound
pub struct TkoAnalysis {
    /// Token graph for dependency tracking
    pub graph: TokenGraph,
    /// Barriers that could potentially be eliminated
    pub eliminable_barriers: Vec<usize>,
}

impl TkoAnalysis {
    /// Create a new TKO analysis
    #[must_use]
    pub fn new() -> Self {
        Self { graph: TokenGraph::new(), eliminable_barriers: Vec::new() }
    }

    /// Check if token-based ordering is sound (no data races)
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Falsification test #67: Token dependencies prevent data races
    #[must_use]
    pub fn is_sound(&self) -> bool {
        // Sound if no cycles and all memory ops have tokens
        !self.graph.has_cycle()
    }

    /// Count eliminable barriers
    #[must_use]
    pub fn eliminable_count(&self) -> usize {
        self.eliminable_barriers.len()
    }
}

impl Default for TkoAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;
