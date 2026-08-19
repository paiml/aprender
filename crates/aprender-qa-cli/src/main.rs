//! APR QA CLI
//!
//! Command-line interface for running model qualification playbooks.
//!
//! The command surface itself lives in `aprender_qa_cli::cli` so that other
//! binaries can dispatch into it; this target is only a shim.

#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::ptr_arg)]

fn main() {
    aprender_qa_cli::cli::run();
}
