//! Sovereign Distribution Manifest (ENT-016)
//!
//! Provides distribution packaging for air-gapped deployment scenarios,
//! like old-school Linux .ISO hosting at universities.

mod component;
mod format;
mod sovereign;
mod tier;

pub use component::ComponentManifest;
pub use format::DistributionFormat;
pub use sovereign::SovereignDistribution;
pub use tier::DistributionTier;

#[cfg(test)]
mod tests;
