//! Cubic pruning schedule methods.
//!
//! This module implements the cubic pruning schedule which provides faster
//! initial pruning that gradually slows down as training progresses.
//!
//! Formula: s_t = s_f * (1 - (1 - t/T)^3)

mod core;

#[cfg(test)]
mod proptests;
#[cfg(test)]
mod tests;
