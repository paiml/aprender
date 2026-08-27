//! Module trait for neural network layers.
//!
//! The Module trait defines the interface for all neural network components,
//! following `PyTorch`'s design (Paszke et al., 2019).

use crate::autograd::Tensor;

/// Base trait for all neural network modules.
///
/// Every layer, activation function, and container implements this trait,
/// providing a uniform interface for:
/// - Forward computation
/// - Parameter access (for optimizers)
/// - Training/evaluation mode switching
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{Module, Linear};
/// use aprender::autograd::Tensor;
///
/// let layer = Linear::new(10, 5);
/// let x = Tensor::randn(&[32, 10]);
/// let output = layer.forward(&x);  // [32, 5]
///
/// // Access parameters for gradient descent
/// for param in layer.parameters() {
///     println!("Shape: {:?}", param.shape());
/// }
/// ```
pub trait Module: Send + Sync {
    /// Perform forward computation.
    ///
    /// This is the main computation method. Given an input tensor,
    /// it returns the output tensor. The computation graph is
    /// automatically recorded for backpropagation.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Get references to all learnable parameters.
    ///
    /// Used by optimizers to iterate over parameters for gradient updates.
    /// Parameters are returned in a deterministic order.
    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    /// Get mutable references to all learnable parameters.
    ///
    /// Used by optimizers to update parameters in-place.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    /// Get `(name, parameter)` pairs for all learnable parameters.
    ///
    /// Leaf modules use local names (`"weight"`, `"bias"`); composites prefix each
    /// child's names with the child's local name and a dot.
    ///
    /// # Invariant (holds for EVERY implementor, including non-overriding ones)
    ///
    /// `named_parameters()` has the same length as [`Module::parameters`], and element
    /// `i` of each refers to the same tensor. The DEFAULT satisfies this **by
    /// construction**: it enumerates `parameters()` and emits stable numeric names
    /// (`"0"`, `"1"`, ...). An empty default would let a parameter-bearing implementor
    /// report zero names while reporting N positional parameters, silently breaking the
    /// freeze-group partitioning and optimizer grouping that address parameters by name.
    /// Overriding to supply semantic names is therefore always safe and must never change
    /// arity or order.
    ///
    /// # Naming rules for overrides
    ///
    /// 1. Named length and order ALWAYS equal positional length and order — by
    ///    construction for the default, by contract for every override.
    /// 2. Names cover learnable tensors ONLY. RNG state, seeds, and training-mode flags
    ///    are never named — they are not parameters and must not appear here.
    /// 3. Composite modules prefix each child's names with the child's local name and a
    ///    dot (e.g. `"0.weight"`, `"q_proj.bias"`).
    /// 4. Names must be unique within a single implementor's output.
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        self.parameters()
            .into_iter()
            .enumerate()
            .map(|(i, t)| (i.to_string(), t))
            .collect()
    }

    /// Get `(name, mutable parameter)` pairs for all learnable parameters.
    ///
    /// Mutable mirror of [`Module::named_parameters`]; the same invariant and naming
    /// rules apply, measured against [`Module::parameters_mut`].
    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        self.parameters_mut()
            .into_iter()
            .enumerate()
            .map(|(i, t)| (i.to_string(), t))
            .collect()
    }

    /// Set training mode recursively, propagating into child modules.
    ///
    /// [`Module::train`] and [`Module::eval`] are leaf-local by convention; composites
    /// override this method to recurse. Switching mode must never add, remove, or mutate
    /// a registered parameter — flipping `train -> eval -> train` leaves every parameter
    /// byte-identical.
    fn set_training(&mut self, training: bool) {
        if training {
            self.train();
        } else {
            self.eval();
        }
    }

    /// Refresh any cached computations after parameters have been modified.
    ///
    /// Called after loading weights via `parameters_mut()` to ensure
    /// derived values (like transposed weight matrices) are up-to-date.
    fn refresh_caches(&mut self) {
        // Default: no-op for modules without caches
    }

    /// Set the module to training mode.
    ///
    /// This affects layers like Dropout (active during training)
    /// and `BatchNorm` (uses batch statistics during training).
    fn train(&mut self) {
        // Default: no-op for stateless modules
    }

    /// Set the module to evaluation mode.
    ///
    /// This affects layers like Dropout (disabled during eval)
    /// and `BatchNorm` (uses running statistics during eval).
    fn eval(&mut self) {
        // Default: no-op for stateless modules
    }

    /// Check if the module is in training mode.
    fn training(&self) -> bool {
        true // Default: always training for stateless modules
    }

    /// Zero out gradients for all parameters.
    ///
    /// Should be called before each training iteration.
    fn zero_grad(&mut self) {
        for param in self.parameters_mut() {
            param.zero_grad_();
        }
    }

    /// Get the number of learnable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyModule {
        weight: Tensor,
    }

    impl DummyModule {
        fn new() -> Self {
            Self {
                weight: Tensor::ones(&[3, 3]),
            }
        }
    }

    impl Module for DummyModule {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.weight]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.weight]
        }
    }

    #[test]
    fn test_module_num_parameters() {
        let module = DummyModule::new();
        assert_eq!(module.num_parameters(), 9); // 3x3 = 9
    }

    #[test]
    fn test_module_parameters() {
        let module = DummyModule::new();
        let params = module.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].shape(), &[3, 3]);
    }

    #[test]
    fn test_module_forward() {
        let module = DummyModule::new();
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let output = module.forward(&input);
        assert_eq!(output.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_module_training() {
        let module = DummyModule::new();
        assert!(module.training());
    }

    #[test]
    fn test_module_zero_grad() {
        let mut module = DummyModule::new();
        module.zero_grad();
        // zero_grad should complete without panic
    }

    #[test]
    fn test_module_parameters_mut() {
        let mut module = DummyModule::new();
        let params = module.parameters_mut();
        assert_eq!(params.len(), 1);
    }

    // Test module that uses all default trait implementations
    struct MinimalModule;

    impl Module for MinimalModule {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }
    }

    #[test]
    fn test_module_default_parameters() {
        let module = MinimalModule;
        let params = module.parameters();
        assert!(params.is_empty());
    }

    #[test]
    fn test_module_default_parameters_mut() {
        let mut module = MinimalModule;
        let params = module.parameters_mut();
        assert!(params.is_empty());
    }

    #[test]
    fn test_module_default_refresh_caches() {
        let mut module = MinimalModule;
        module.refresh_caches(); // Should not panic
    }

    #[test]
    fn test_module_default_train() {
        let mut module = MinimalModule;
        module.train(); // Should not panic
    }

    #[test]
    fn test_module_default_eval() {
        let mut module = MinimalModule;
        module.eval(); // Should not panic
    }

    #[test]
    fn test_module_default_training() {
        let module = MinimalModule;
        assert!(module.training()); // Default is true
    }

    #[test]
    fn test_module_default_zero_grad() {
        let mut module = MinimalModule;
        module.zero_grad(); // Should not panic with empty params
    }

    #[test]
    fn test_module_default_num_parameters() {
        let module = MinimalModule;
        assert_eq!(module.num_parameters(), 0); // No parameters
    }

    // ---------------------------------------------------------------------
    // Named traversal defaults (ENC-04) + recursive mode propagation (ENC-05)
    // ---------------------------------------------------------------------

    /// Parameter-bearing module that implements ONLY the positional accessors and
    /// never overrides the named ones. This is the regression guard: an empty
    /// `named_parameters()` default would report 0 names for 3 parameters.
    struct PositionalOnlyModule {
        a: Tensor,
        b: Tensor,
        c: Tensor,
    }

    impl PositionalOnlyModule {
        fn new() -> Self {
            Self {
                a: Tensor::ones(&[2, 2]),
                b: Tensor::ones(&[3]),
                c: Tensor::ones(&[4, 1]),
            }
        }
    }

    impl Module for PositionalOnlyModule {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.a, &self.b, &self.c]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.a, &mut self.b, &mut self.c]
        }
    }

    #[test]
    fn test_module_named_default_is_not_empty_for_parameter_bearing_module() {
        let module = PositionalOnlyModule::new();
        let positional = module.parameters();
        let named = module.named_parameters();

        // The defect this default fixes: N parameters MUST yield N names.
        assert_eq!(
            named.len(),
            positional.len(),
            "named_parameters() must have the same arity as parameters()"
        );
        assert_eq!(named.len(), 3);

        let names: Vec<String> = named.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(
            names,
            vec!["0".to_string(), "1".to_string(), "2".to_string()]
        );
    }

    #[test]
    fn test_module_named_default_pairwise_agreement_with_positional() {
        let module = PositionalOnlyModule::new();
        let positional = module.parameters();
        let named = module.named_parameters();

        for (i, (_, tensor)) in named.iter().enumerate() {
            assert_eq!(
                tensor.shape(),
                positional[i].shape(),
                "element {i}: named and positional must refer to the same tensor"
            );
            assert_eq!(
                tensor.data(),
                positional[i].data(),
                "element {i}: data differs"
            );
        }
    }

    #[test]
    fn test_module_named_default_names_are_unique() {
        let module = PositionalOnlyModule::new();
        let named = module.named_parameters();
        let mut names: Vec<String> = named.iter().map(|(n, _)| n.clone()).collect();
        let total = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), total, "default names must be unique");
    }

    #[test]
    fn test_module_named_mut_default_matches_positional_arity() {
        let mut module = PositionalOnlyModule::new();
        let positional_len = module.parameters_mut().len();
        let named = module.named_parameters_mut();
        assert_eq!(named.len(), positional_len);
        let names: Vec<String> = named.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(
            names,
            vec!["0".to_string(), "1".to_string(), "2".to_string()]
        );
    }

    #[test]
    fn test_module_named_default_empty_for_parameterless_module() {
        let module = MinimalModule;
        assert!(module.named_parameters().is_empty());
        assert_eq!(module.named_parameters().len(), module.parameters().len());
    }

    #[test]
    fn test_module_named_mut_default_empty_for_parameterless_module() {
        let mut module = MinimalModule;
        assert!(module.named_parameters_mut().is_empty());
    }

    /// Module with real mode state, to exercise the `set_training` default.
    struct ModeTrackingModule {
        training: bool,
    }

    impl Module for ModeTrackingModule {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn training(&self) -> bool {
            self.training
        }
    }

    #[test]
    fn test_module_set_training_default_delegates_to_train_and_eval() {
        let mut module = ModeTrackingModule { training: true };

        module.set_training(false);
        assert!(
            !module.training(),
            "set_training(false) must delegate to eval()"
        );

        module.set_training(true);
        assert!(
            module.training(),
            "set_training(true) must delegate to train()"
        );
    }

    #[test]
    fn test_module_set_training_default_is_noop_for_stateless_module() {
        let mut module = MinimalModule;
        module.set_training(false);
        assert!(module.training()); // stateless default always reports true
        module.set_training(true);
        assert!(module.training());
    }

    #[test]
    fn test_module_set_training_does_not_change_parameters() {
        let mut module = PositionalOnlyModule::new();
        let before: Vec<Vec<u32>> = module
            .parameters()
            .iter()
            .map(|t| t.data().iter().map(|v| v.to_bits()).collect())
            .collect();

        module.set_training(true);
        module.set_training(false);
        module.set_training(true);

        let after: Vec<Vec<u32>> = module
            .parameters()
            .iter()
            .map(|t| t.data().iter().map(|v| v.to_bits()).collect())
            .collect();

        assert_eq!(
            before, after,
            "mode flips must leave parameters byte-identical"
        );
    }

    /// Cross-module reachability proof for the shared byte-identity helper defined
    /// in `nn/tests_named_module.rs`. This test lives in a DIFFERENT module on
    /// purpose: plan 01-06's encoder conformance tests call `snapshot_named` from
    /// outside `nn`, so `pub(crate)` visibility must actually hold. A private fn
    /// inside a `mod tests` block would fail to compile here.
    #[test]
    fn test_module_snapshot_named_helper_is_reachable_cross_module() {
        let module = PositionalOnlyModule::new();
        let snapshot = crate::nn::tests_named_module::snapshot_named(&module);
        assert_eq!(snapshot.len(), module.parameters().len());
        assert_eq!(snapshot[0].0, "0");
        assert_eq!(snapshot[0].1.len(), 4); // [2, 2] of ones
    }
}
