//! F082 Falsification Tests
//!
//! These tests apply Popperian falsification to the F082 hypothesis:
//! "Computed Address From Loaded Value causes crash when ld.shared value
//! is used to compute address for st.global"
//!
//! Alternative Hypothesis: The bug is in 32->64 bit conversion during
//! address computation, not the dependency chain itself.

#[cfg(feature = "cuda")]
mod global_computed_addr;
#[cfg(feature = "cuda")]
mod membar_fix;
#[cfg(feature = "cuda")]
mod reproduce_and_shared;
