//! A non-`mod.rs` module that owns a child directory (the Rust 2018 layout):
//! `mod twig;` here declares `src/flat/twig.rs`, which lives in a directory
//! that holds no `mod.rs` and no file declaring it.
pub mod twig;
