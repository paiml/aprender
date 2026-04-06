//! Model loading and teacher model abstraction
//!
//! Provides format-agnostic model loading with memory estimation.

mod memory;
mod safetensors;
mod teacher;

#[cfg(test)]
mod tests;

pub use memory::MemoryEstimate;
pub use safetensors::SafeTensorsTeacher;
pub use teacher::TeacherModel;
