//! apr - APR Model Operations CLI
//!
//! Entry point shim. The program is `apr_cli::cli_main`, the SAME function the
//! root facade's `src/bin/apr.rs` calls: both packages define a binary named
//! `apr`, and G0.5 (#2582) found the two had been kept equal by hand-copying
//! the prologue, and had already diverged (the dhat allocator existed only
//! here). `entry_point_is_shared` in lib.rs fails if a prologue comes back.

fn main() -> std::process::ExitCode {
    apr_cli::cli_main()
}
