//! Mean Squared Error, Mean Absolute Error, and Huber losses

mod huber_loss;
mod l1_loss;
mod mse_loss;

#[cfg(test)]
mod tests;

pub use huber_loss::{HuberLoss, SmoothL1Loss};
pub use l1_loss::L1Loss;
pub use mse_loss::MSELoss;
