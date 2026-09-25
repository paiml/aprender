//! trueno-rag binary. The command surface lives in the library
//! (`aprender_rag_cli`) so `apr rag` can reach the same code; see the module
//! docs there.

/// Printed before anything runs: this binary is one of the duplicates `apr`
/// absorbed (#4060, EPIC #4057). It still works, through the same library code
/// `apr rag` calls, so scripts that call it keep running until its consumers
/// have moved; then the bin target is removed.
const DEPRECATED: &str = "warning: `trueno-rag` is deprecated and will be removed; run `apr rag` instead (the same code path). See paiml/aprender#4057.";

fn main() -> anyhow::Result<()> {
    eprintln!("{DEPRECATED}");
    aprender_rag_cli::run()
}
