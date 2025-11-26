//! Neural network model serialization.
//!
//! Provides SafeTensors-compatible serialization for nn modules.
//! This enables saving trained models and loading them for inference.
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::{Sequential, Linear, ReLU, Module};
//! use aprender::nn::serialize::{save_model, load_state_dict};
//!
//! // Build and train a model
//! let model = Sequential::new()
//!     .add(Linear::new(784, 256))
//!     .add(ReLU::new())
//!     .add(Linear::new(256, 10));
//!
//! // Save model weights
//! save_model(&model, "model.safetensors").unwrap();
//!
//! // Load weights into a new model
//! let mut new_model = Sequential::new()
//!     .add(Linear::new(784, 256))
//!     .add(ReLU::new())
//!     .add(Linear::new(256, 10));
//!
//! let state = load_state_dict("model.safetensors").unwrap();
//! load_into_model(&mut new_model, &state).unwrap();
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use super::module::Module;
use crate::autograd::Tensor;
use crate::serialization::safetensors::{extract_tensor, load_safetensors, save_safetensors};

/// State dictionary: mapping from parameter names to tensor data and shapes.
pub type StateDict = BTreeMap<String, (Vec<f32>, Vec<usize>)>;

/// Extract state dictionary from a module.
///
/// Returns a mapping of parameter names (by index) to their data and shapes.
///
/// # Arguments
///
/// * `module` - The module to extract state from
/// * `prefix` - Optional prefix for parameter names (for nested modules)
///
/// # Example
///
/// ```ignore
/// let model = Linear::new(10, 5);
/// let state = state_dict(&model, "");
/// // Contains: {"0": (weight_data, [5, 10]), "1": (bias_data, [5])}
/// ```
pub fn state_dict<M: Module + ?Sized>(module: &M, prefix: &str) -> StateDict {
    let mut state = StateDict::new();

    for (i, param) in module.parameters().iter().enumerate() {
        let name = if prefix.is_empty() {
            format!("{i}")
        } else {
            format!("{prefix}.{i}")
        };

        state.insert(name, (param.data().to_vec(), param.shape().to_vec()));
    }

    state
}

/// Load state dictionary into a module.
///
/// # Arguments
///
/// * `module` - The module to load state into
/// * `state` - State dictionary to load
/// * `prefix` - Optional prefix for parameter names
///
/// # Errors
///
/// Returns an error if:
/// - Parameter count mismatch
/// - Shape mismatch between state and module parameters
pub fn load_state_dict_into<M: Module + ?Sized>(
    module: &mut M,
    state: &StateDict,
    prefix: &str,
) -> Result<(), String> {
    let params = module.parameters_mut();

    for (i, param) in params.into_iter().enumerate() {
        let name = if prefix.is_empty() {
            format!("{i}")
        } else {
            format!("{prefix}.{i}")
        };

        let (data, shape) = state
            .get(&name)
            .ok_or_else(|| format!("Missing parameter '{name}' in state dict"))?;

        // Verify shape matches
        if param.shape() != shape.as_slice() {
            return Err(format!(
                "Shape mismatch for parameter '{name}': expected {:?}, got {:?}",
                param.shape(),
                shape
            ));
        }

        // Load data into parameter
        *param = Tensor::new(data, shape).requires_grad();
    }

    Ok(())
}

/// Save a module's parameters to a SafeTensors file.
///
/// # Arguments
///
/// * `module` - The module to save
/// * `path` - File path to write to
///
/// # Example
///
/// ```ignore
/// let model = Linear::new(10, 5);
/// save_model(&model, "linear.safetensors").unwrap();
/// ```
pub fn save_model<M: Module + ?Sized, P: AsRef<Path>>(module: &M, path: P) -> Result<(), String> {
    let state = state_dict(module, "");
    save_safetensors(path, &state)
}

/// Load a state dictionary from a SafeTensors file.
///
/// # Arguments
///
/// * `path` - File path to read from
///
/// # Returns
///
/// State dictionary that can be loaded into a module.
pub fn load_state_dict<P: AsRef<Path>>(path: P) -> Result<StateDict, String> {
    let (metadata, raw_data) = load_safetensors(path)?;

    let mut state = StateDict::new();

    for (name, tensor_meta) in metadata {
        let data = extract_tensor(&raw_data, &tensor_meta)?;
        state.insert(name, (data, tensor_meta.shape));
    }

    Ok(state)
}

/// Load parameters from a SafeTensors file into a module.
///
/// Convenience function that combines `load_state_dict` and `load_state_dict_into`.
///
/// # Arguments
///
/// * `module` - The module to load parameters into
/// * `path` - File path to read from
///
/// # Example
///
/// ```ignore
/// let mut model = Linear::new(10, 5);
/// load_model(&mut model, "linear.safetensors").unwrap();
/// ```
pub fn load_model<M: Module + ?Sized, P: AsRef<Path>>(
    module: &mut M,
    path: P,
) -> Result<(), String> {
    let state = load_state_dict(path)?;
    load_state_dict_into(module, &state, "")
}

/// Get the number of parameters that would be saved.
pub fn count_parameters<M: Module + ?Sized>(module: &M) -> usize {
    module.parameters().iter().map(|p| p.numel()).sum()
}

/// Get the size in bytes that would be saved (F32 = 4 bytes per parameter).
pub fn model_size_bytes<M: Module + ?Sized>(module: &M) -> usize {
    count_parameters(module) * 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, ReLU, Sequential};

    #[test]
    fn test_state_dict_linear() {
        let layer = Linear::with_seed(10, 5, Some(42));
        let state = state_dict(&layer, "");

        assert_eq!(state.len(), 2); // weight + bias

        let (weight_data, weight_shape) = &state["0"];
        assert_eq!(weight_shape, &[5, 10]);
        assert_eq!(weight_data.len(), 50);

        let (bias_data, bias_shape) = &state["1"];
        assert_eq!(bias_shape, &[5]);
        assert_eq!(bias_data.len(), 5);
    }

    #[test]
    fn test_state_dict_sequential() {
        let model = Sequential::new()
            .add(Linear::with_seed(10, 8, Some(42)))
            .add(ReLU::new())
            .add(Linear::with_seed(8, 5, Some(43)));

        let state = state_dict(&model, "");

        // 2 Linear layers * 2 params each = 4
        assert_eq!(state.len(), 4);
    }

    #[test]
    fn test_load_state_dict_into() {
        let layer1 = Linear::with_seed(10, 5, Some(42));
        let state = state_dict(&layer1, "");

        let mut layer2 = Linear::with_seed(10, 5, Some(99)); // Different seed

        // Verify they're different initially
        assert_ne!(layer1.parameters()[0].data(), layer2.parameters()[0].data());

        // Load state
        load_state_dict_into(&mut layer2, &state, "").expect("load_state_dict_into should succeed");

        // Now they should match
        assert_eq!(layer1.parameters()[0].data(), layer2.parameters()[0].data());
    }

    #[test]
    fn test_save_and_load_model() {
        let path = "/tmp/test_nn_serialize.safetensors";

        let model1 = Linear::with_seed(10, 5, Some(42));
        save_model(&model1, path).expect("save_model should succeed");

        let mut model2 = Linear::with_seed(10, 5, Some(99));
        load_model(&mut model2, path).expect("load_model should succeed");

        // Verify weights match
        assert_eq!(model1.parameters()[0].data(), model2.parameters()[0].data());
        assert_eq!(model1.parameters()[1].data(), model2.parameters()[1].data());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_and_load_sequential() {
        let path = "/tmp/test_nn_serialize_seq.safetensors";

        let model1 = Sequential::new()
            .add(Linear::with_seed(10, 8, Some(42)))
            .add(ReLU::new())
            .add(Linear::with_seed(8, 5, Some(43)));

        save_model(&model1, path).expect("save_model should succeed");

        let mut model2 = Sequential::new()
            .add(Linear::with_seed(10, 8, Some(99)))
            .add(ReLU::new())
            .add(Linear::with_seed(8, 5, Some(100)));

        load_model(&mut model2, path).expect("load_model should succeed");

        // Verify all parameters match
        for (p1, p2) in model1.parameters().iter().zip(model2.parameters().iter()) {
            assert_eq!(p1.data(), p2.data());
            assert_eq!(p1.shape(), p2.shape());
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_state_dict_shape_mismatch() {
        let layer1 = Linear::with_seed(10, 5, Some(42));
        let state = state_dict(&layer1, "");

        // Different architecture
        let mut layer2 = Linear::with_seed(20, 10, Some(99));

        let result = load_state_dict_into(&mut layer2, &state, "");
        assert!(result.is_err());
        let err = result.expect_err("Should fail with shape mismatch");
        assert!(err.contains("Shape mismatch"));
    }

    #[test]
    fn test_count_parameters() {
        let model = Sequential::new()
            .add(Linear::new(10, 8))  // 10*8 + 8 = 88
            .add(Linear::new(8, 5)); // 8*5 + 5 = 45

        assert_eq!(count_parameters(&model), 133);
    }

    #[test]
    fn test_model_size_bytes() {
        let model = Linear::new(10, 5); // 10*5 + 5 = 55 params

        assert_eq!(model_size_bytes(&model), 55 * 4); // 220 bytes
    }

    #[test]
    fn test_model_forward_after_load() {
        let path = "/tmp/test_nn_forward_after_load.safetensors";

        let model1 = Linear::with_seed(10, 5, Some(42));
        let x = Tensor::ones(&[2, 10]);
        let y1 = model1.forward(&x);

        save_model(&model1, path).expect("save_model should succeed");

        let mut model2 = Linear::with_seed(10, 5, Some(99));
        load_model(&mut model2, path).expect("load_model should succeed");

        let y2 = model2.forward(&x);

        // Forward passes should produce identical results
        assert_eq!(y1.data(), y2.data());

        std::fs::remove_file(path).ok();
    }
}
