//! trueno-rag binary. The command surface lives in the library
//! (`aprender_rag_cli`) so `apr rag` can reach the same code; see the module
//! docs there.

fn main() -> anyhow::Result<()> {
    sovereign_update::hook!("trueno-rag"); // EPIC #4232: `trueno-rag update`, and the startup notice
    aprender_rag_cli::run()
}
