//! In-memory model registry implementation

mod registry;
mod traits_impl;

#[cfg(test)]
mod tests;

pub use registry::InMemoryRegistry;
