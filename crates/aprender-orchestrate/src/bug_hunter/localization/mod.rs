//! Advanced Fault Localization Module (BH-16 to BH-20)
//!
//! Implements research-based fault localization techniques:
//! - BH-16: Mutation-Based Fault Localization (MBFL)
//! - BH-17: Causal Fault Localization
//! - BH-18: Predictive Mutation Testing
//! - BH-19: Multi-Channel Fault Localization
//! - BH-20: Semantic Crash Bucketing
//!
//! References:
//! - Papadakis & Le Traon (2015) "Metallaxis-FL" - IEEE TSE
//! - Baah et al. (2010) "Causal Inference for Statistical Fault Localization" - ISSTA
//! - Zhang et al. (2018) "Predictive Mutation Testing" - IEEE TSE
//! - Li et al. (2021) "DeepFL" - ISSTA
//! - Cui et al. (2016) "RETracer" - ICSE

mod crash_bucketing;
mod multi_channel;
mod scoring;

pub use crash_bucketing::{CrashBucket, CrashBucketer, CrashInfo, RootCausePattern, StackFrame};
pub use multi_channel::MultiChannelLocalizer;
pub use scoring::{MutationData, ScoredLocation, SpectrumData, TestCoverage};

#[cfg(test)]
mod tests_crash_bucketing;
#[cfg(test)]
mod tests_multi_channel;
#[cfg(test)]
mod tests_scoring;
#[cfg(test)]
mod tests_semantic;
