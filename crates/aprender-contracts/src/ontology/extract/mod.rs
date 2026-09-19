//! ONT-001 §3.7 — extractors: how each entity type becomes RDF. Every extractor is pure Rust, deterministic
//! (R-15), and registered in Σ `extractors[]` with its reader gate (R-11). ONT-4b implements `pv_contract`; the
//! rest are declared in Σ and arrive with their rows (ONT-4b2: code, lean; ONT-4c: readme, llm_context, apr_model, csv).

pub mod pv_contract;
