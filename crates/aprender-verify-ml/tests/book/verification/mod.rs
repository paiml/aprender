//! Verification oracle chapter examples

// `python-oracle`: these rows execute python3 and the clean-room image has none.
// See the feature's comment in this crate's Cargo.toml. NOT a skip -- where the
// feature is on, they still fail hard without an interpreter.
#[cfg(feature = "python-oracle")]
mod oracles;
