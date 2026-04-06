//! mdBook page generation for contracts.
//!
//! Generates per-contract markdown pages with `KaTeX` equations and
//! Mermaid dependency graphs, plus SUMMARY.md integration.

mod page;
mod summary;

pub use page::generate_contract_page;
pub use summary::update_summary;
