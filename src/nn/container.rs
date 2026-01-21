//! Container modules for composing neural networks.
//!
//! These containers allow building complex networks from simpler modules.

use super::module::Module;
use crate::autograd::Tensor;
use std::collections::HashMap;

/// Sequential container for chaining modules.
///
/// Modules are executed in order, with each module's output
/// becoming the next module's input.
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{Sequential, Linear, ReLU, Dropout};
///
/// let model = Sequential::new()
///     .add(Linear::new(784, 256))
///     .add(ReLU::new())
///     .add(Dropout::new(0.5))
///     .add(Linear::new(256, 10));
///
/// let x = Tensor::randn(&[32, 784]);
/// let output = model.forward(&x);  // [32, 10]
/// ```
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
    training: bool,
}

impl Sequential {
    /// Create an empty Sequential container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }

    /// Add a module to the sequence.
    ///
    /// Returns self for method chaining.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    /// Add a module by boxed trait object.
    #[must_use]
    pub fn add_boxed(mut self, module: Box<dyn Module>) -> Self {
        self.modules.push(module);
        self
    }

    /// Get the number of modules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the container is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.modules
            .iter()
            .fold(input.clone(), |x, module| module.forward(&x))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.modules
            .iter_mut()
            .flat_map(|m| m.parameters_mut())
            .collect()
    }

    fn train(&mut self) {
        self.training = true;
        for module in &mut self.modules {
            module.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for module in &mut self.modules {
            module.eval();
        }
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("num_modules", &self.modules.len())
            .field("training", &self.training)
            .finish()
    }
}

/// List of modules with index-based access.
///
/// Unlike Sequential, `ModuleList` doesn't define a forward pass.
/// It's useful for holding submodules that need custom control flow.
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{ModuleList, Linear, Module};
///
/// let layers = ModuleList::new()
///     .add(Linear::new(10, 10))
///     .add(Linear::new(10, 10))
///     .add(Linear::new(10, 10));
///
/// // Custom forward with residual connections
/// fn forward(layers: &ModuleList, x: &Tensor) -> Tensor {
///     let mut out = x.clone();
///     for i in 0..layers.len() {
///         let residual = out.clone();
///         out = layers.get(i).unwrap().forward(&out);
///         out = out.add(&residual);  // residual connection
///     }
///     out
/// }
/// ```
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
    training: bool,
}

impl ModuleList {
    /// Create an empty `ModuleList`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }

    /// Add a module to the list.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    /// Add a boxed module to the list.
    #[must_use]
    pub fn add_boxed(mut self, module: Box<dyn Module>) -> Self {
        self.modules.push(module);
        self
    }

    /// Get a module by index.
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.modules.get(index).map(AsRef::as_ref)
    }

    /// Get a mutable module by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Box<dyn Module>> {
        self.modules.get_mut(index)
    }

    /// Get the number of modules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Iterate over modules.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Module> {
        self.modules.iter().map(AsRef::as_ref)
    }
}

impl Default for ModuleList {
    fn default() -> Self {
        Self::new()
    }
}

// ModuleList doesn't implement forward - use get() for custom control flow
impl ModuleList {
    /// Get all parameters from all modules.
    #[must_use]
    pub fn parameters(&self) -> Vec<&Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    /// Get all mutable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.modules
            .iter_mut()
            .flat_map(|m| m.parameters_mut())
            .collect()
    }

    /// Set all modules to training mode.
    pub fn train(&mut self) {
        self.training = true;
        for module in &mut self.modules {
            module.train();
        }
    }

    /// Set all modules to evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
        for module in &mut self.modules {
            module.eval();
        }
    }
}

impl std::fmt::Debug for ModuleList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleList")
            .field("num_modules", &self.modules.len())
            .field("training", &self.training)
            .finish()
    }
}

/// Dictionary of named modules with string-based access.
///
/// Useful for building networks with multiple branches or when you need
/// to access modules by name rather than index.
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{ModuleDict, Linear, Module};
/// use aprender::autograd::Tensor;
///
/// let modules = ModuleDict::new()
///     .insert("encoder", Linear::new(784, 256))
///     .insert("decoder", Linear::new(256, 784))
///     .insert("classifier", Linear::new(256, 10));
///
/// let x = Tensor::randn(&[32, 784]);
/// let encoded = modules.get("encoder").unwrap().forward(&x);
/// let decoded = modules.get("decoder").unwrap().forward(&encoded);
/// let logits = modules.get("classifier").unwrap().forward(&encoded);
/// ```
pub struct ModuleDict {
    modules: HashMap<String, Box<dyn Module>>,
    /// Insertion order for deterministic iteration
    keys: Vec<String>,
    training: bool,
}

impl ModuleDict {
    /// Create an empty `ModuleDict`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            keys: Vec::new(),
            training: true,
        }
    }

    /// Insert a module with the given name.
    ///
    /// If a module with the same name exists, it will be replaced.
    pub fn insert<S: Into<String>, M: Module + 'static>(mut self, name: S, module: M) -> Self {
        let name = name.into();
        if !self.modules.contains_key(&name) {
            self.keys.push(name.clone());
        }
        self.modules.insert(name, Box::new(module));
        self
    }

    /// Insert a boxed module.
    pub fn insert_boxed<S: Into<String>>(mut self, name: S, module: Box<dyn Module>) -> Self {
        let name = name.into();
        if !self.modules.contains_key(&name) {
            self.keys.push(name.clone());
        }
        self.modules.insert(name, module);
        self
    }

    /// Get a module by name.
    pub fn get(&self, name: &str) -> Option<&dyn Module> {
        self.modules.get(name).map(AsRef::as_ref)
    }

    /// Get a mutable module by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Box<dyn Module>> {
        self.modules.get_mut(name)
    }

    /// Check if a module with the given name exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// Remove a module by name.
    pub fn remove(&mut self, name: &str) -> Option<Box<dyn Module>> {
        if let Some(module) = self.modules.remove(name) {
            self.keys.retain(|k| k != name);
            Some(module)
        } else {
            None
        }
    }

    /// Get the number of modules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get all module names in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.keys.iter().map(String::as_str)
    }

    /// Iterate over (name, module) pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &dyn Module)> {
        self.keys.iter().map(|k| {
            let module = self
                .modules
                .get(k)
                .map(AsRef::as_ref)
                .expect("key must exist");
            (k.as_str(), module)
        })
    }

    /// Get all parameters from all modules.
    #[must_use]
    pub fn parameters(&self) -> Vec<&Tensor> {
        self.keys
            .iter()
            .filter_map(|k| self.modules.get(k))
            .flat_map(|m| m.parameters())
            .collect()
    }

    /// Get all mutable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.modules
            .values_mut()
            .flat_map(|m| m.parameters_mut())
            .collect()
    }

    /// Set all modules to training mode.
    pub fn train(&mut self) {
        self.training = true;
        for module in self.modules.values_mut() {
            module.train();
        }
    }

    /// Set all modules to evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
        for module in self.modules.values_mut() {
            module.eval();
        }
    }

    /// Check if in training mode.
    #[must_use]
    pub fn training(&self) -> bool {
        self.training
    }
}

impl Default for ModuleDict {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ModuleDict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleDict")
            .field("keys", &self.keys)
            .field("training", &self.training)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Dropout, Linear, ReLU};

    #[test]
    fn test_sequential_empty() {
        let model = Sequential::new();
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);

        let x = Tensor::ones(&[1, 10]);
        let y = model.forward(&x);
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_sequential_single_layer() {
        let model = Sequential::new().add(Linear::with_seed(10, 5, Some(42)));

        assert_eq!(model.len(), 1);

        let x = Tensor::ones(&[2, 10]);
        let y = model.forward(&x);
        assert_eq!(y.shape(), &[2, 5]);
    }

    #[test]
    fn test_sequential_multiple_layers() {
        let model = Sequential::new()
            .add(Linear::with_seed(10, 8, Some(42)))
            .add(ReLU::new())
            .add(Linear::with_seed(8, 5, Some(43)));

        assert_eq!(model.len(), 3);

        let x = Tensor::ones(&[4, 10]);
        let y = model.forward(&x);
        assert_eq!(y.shape(), &[4, 5]);
    }

    #[test]
    fn test_sequential_parameters() {
        let model = Sequential::new()
            .add(Linear::new(10, 8))
            .add(ReLU::new())
            .add(Linear::new(8, 5));

        let params = model.parameters();
        // Linear(10, 8): weight + bias = 2
        // ReLU: 0
        // Linear(8, 5): weight + bias = 2
        // Total: 4
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_sequential_num_parameters() {
        let model = Sequential::new()
            .add(Linear::new(10, 8))
            .add(Linear::new(8, 5));

        // Linear(10, 8): 10*8 + 8 = 88
        // Linear(8, 5): 8*5 + 5 = 45
        // Total: 133
        assert_eq!(model.num_parameters(), 133);
    }

    #[test]
    fn test_sequential_train_eval() {
        let mut model = Sequential::new()
            .add(Linear::new(10, 5))
            .add(Dropout::new(0.5));

        assert!(model.training());

        model.eval();
        assert!(!model.training());

        model.train();
        assert!(model.training());
    }

    #[test]
    fn test_module_list_basic() {
        let layers = ModuleList::new()
            .add(Linear::new(10, 10))
            .add(Linear::new(10, 10));

        assert_eq!(layers.len(), 2);

        // Access by index
        let layer0 = layers.get(0).expect("layer 0 should exist");
        let x = Tensor::ones(&[1, 10]);
        let _ = layer0.forward(&x);
    }

    #[test]
    fn test_module_list_parameters() {
        let layers = ModuleList::new()
            .add(Linear::new(10, 5))
            .add(Linear::new(5, 3));

        // Linear(10, 5): 2 params
        // Linear(5, 3): 2 params
        assert_eq!(layers.parameters().len(), 4);
    }

    #[test]
    fn test_module_list_iterate() {
        let layers = ModuleList::new()
            .add(Linear::new(10, 10))
            .add(Linear::new(10, 10))
            .add(Linear::new(10, 10));

        let mut count = 0;
        for _ in layers.iter() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    // ModuleDict tests

    #[test]
    fn test_module_dict_empty() {
        let dict = ModuleDict::new();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
    }

    #[test]
    fn test_module_dict_insert_and_get() {
        let dict = ModuleDict::new()
            .insert("layer1", Linear::new(10, 5))
            .insert("layer2", Linear::new(5, 3));

        assert_eq!(dict.len(), 2);
        assert!(dict.contains("layer1"));
        assert!(dict.contains("layer2"));
        assert!(!dict.contains("layer3"));

        let layer1 = dict.get("layer1").expect("layer1 should exist");
        let x = Tensor::ones(&[2, 10]);
        let y = layer1.forward(&x);
        assert_eq!(y.shape(), &[2, 5]);
    }

    #[test]
    fn test_module_dict_parameters() {
        let dict = ModuleDict::new()
            .insert("encoder", Linear::new(10, 5))
            .insert("decoder", Linear::new(5, 10));

        // 2 params per Linear (weight + bias) * 2 = 4
        assert_eq!(dict.parameters().len(), 4);
    }

    #[test]
    fn test_module_dict_keys_order() {
        let dict = ModuleDict::new()
            .insert("first", Linear::new(10, 10))
            .insert("second", Linear::new(10, 10))
            .insert("third", Linear::new(10, 10));

        let keys: Vec<&str> = dict.keys().collect();
        assert_eq!(keys, vec!["first", "second", "third"]);
    }

    #[test]
    fn test_module_dict_iterate() {
        let dict = ModuleDict::new()
            .insert("a", Linear::new(10, 10))
            .insert("b", Linear::new(10, 10));

        let mut names = Vec::new();
        for (name, _module) in dict.iter() {
            names.push(name);
        }
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_module_dict_remove() {
        let mut dict = ModuleDict::new()
            .insert("keep", Linear::new(10, 5))
            .insert("remove", Linear::new(5, 3));

        assert_eq!(dict.len(), 2);

        let removed = dict.remove("remove");
        assert!(removed.is_some());
        assert_eq!(dict.len(), 1);
        assert!(!dict.contains("remove"));
        assert!(dict.contains("keep"));

        // Keys should also be updated
        let keys: Vec<&str> = dict.keys().collect();
        assert_eq!(keys, vec!["keep"]);
    }

    #[test]
    fn test_module_dict_train_eval() {
        let mut dict = ModuleDict::new()
            .insert("layer", Linear::new(10, 5))
            .insert("dropout", Dropout::new(0.5));

        assert!(dict.training());

        dict.eval();
        assert!(!dict.training());

        dict.train();
        assert!(dict.training());
    }

    #[test]
    fn test_module_dict_replace() {
        let dict = ModuleDict::new()
            .insert("layer", Linear::new(10, 5))
            .insert("layer", Linear::new(10, 10)); // Replace

        // Should still have only one key
        assert_eq!(dict.len(), 1);

        let layer = dict.get("layer").expect("layer should exist after replace");
        let x = Tensor::ones(&[2, 10]);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), &[2, 10]); // From the second Linear
    }

    #[test]
    fn test_sequential_default() {
        let model: Sequential = Sequential::default();
        assert!(model.is_empty());
    }

    #[test]
    fn test_sequential_add_boxed() {
        let linear: Box<dyn Module> = Box::new(Linear::new(10, 5));
        let model = Sequential::new().add_boxed(linear);
        assert_eq!(model.len(), 1);
    }

    #[test]
    fn test_module_list_default() {
        let list: ModuleList = ModuleList::default();
        assert!(list.is_empty());
    }

    #[test]
    fn test_module_list_add_boxed() {
        let linear: Box<dyn Module> = Box::new(Linear::new(10, 5));
        let list = ModuleList::new().add_boxed(linear);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_module_list_iter() {
        let list = ModuleList::new().add(Linear::new(10, 5)).add(ReLU::new());
        let count = list.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_module_dict_default() {
        let dict: ModuleDict = ModuleDict::default();
        assert!(dict.is_empty());
    }

    #[test]
    fn test_module_dict_insert_boxed() {
        let linear: Box<dyn Module> = Box::new(Linear::new(10, 5));
        let dict = ModuleDict::new().insert_boxed("layer", linear);
        assert!(dict.contains("layer"));
    }

    #[test]
    fn test_module_dict_keys_iter() {
        let dict = ModuleDict::new()
            .insert("a", Linear::new(10, 5))
            .insert("b", ReLU::new());
        let keys: Vec<_> = dict.keys().collect();
        assert_eq!(keys.len(), 2);
        // Use iter() to verify both modules are accessible
        let modules: Vec<_> = dict.iter().collect();
        assert_eq!(modules.len(), 2);
    }

    // ==========================================================================
    // Additional Coverage Tests
    // ==========================================================================

    #[test]
    fn test_sequential_debug() {
        let model = Sequential::new().add(Linear::new(10, 5)).add(ReLU::new());
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("Sequential"));
        assert!(debug_str.contains("num_modules"));
    }

    #[test]
    fn test_sequential_parameters_mut() {
        let mut model = Sequential::new()
            .add(Linear::new(10, 8))
            .add(Linear::new(8, 5));

        let params = model.parameters_mut();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_module_list_debug() {
        let list = ModuleList::new().add(Linear::new(10, 5)).add(ReLU::new());
        let debug_str = format!("{:?}", list);
        assert!(debug_str.contains("ModuleList"));
        assert!(debug_str.contains("num_modules"));
    }

    #[test]
    fn test_module_list_get_mut() {
        let mut list = ModuleList::new()
            .add(Linear::new(10, 5))
            .add(Linear::new(5, 3));

        // Get mutable reference
        let layer = list.get_mut(0);
        assert!(layer.is_some());

        // Index out of bounds
        let missing = list.get_mut(10);
        assert!(missing.is_none());
    }

    #[test]
    fn test_module_list_get_missing() {
        let list = ModuleList::new().add(Linear::new(10, 5));

        // Out of bounds
        let missing = list.get(5);
        assert!(missing.is_none());
    }

    #[test]
    fn test_module_list_train_eval() {
        let mut list = ModuleList::new()
            .add(Linear::new(10, 5))
            .add(Dropout::new(0.5));

        assert!(list.training);

        list.eval();
        assert!(!list.training);

        list.train();
        assert!(list.training);
    }

    #[test]
    fn test_module_list_parameters_mut() {
        let mut list = ModuleList::new()
            .add(Linear::new(10, 8))
            .add(Linear::new(8, 5));

        let params = list.parameters_mut();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_module_dict_debug() {
        let dict = ModuleDict::new()
            .insert("layer1", Linear::new(10, 5))
            .insert("layer2", ReLU::new());
        let debug_str = format!("{:?}", dict);
        assert!(debug_str.contains("ModuleDict"));
        assert!(debug_str.contains("keys"));
    }

    #[test]
    fn test_module_dict_get_mut() {
        let mut dict = ModuleDict::new().insert("layer", Linear::new(10, 5));

        // Get mutable reference
        let layer = dict.get_mut("layer");
        assert!(layer.is_some());

        // Key doesn't exist
        let missing = dict.get_mut("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_module_dict_get_missing() {
        let dict = ModuleDict::new().insert("layer", Linear::new(10, 5));

        let missing = dict.get("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_module_dict_parameters_mut() {
        let mut dict = ModuleDict::new()
            .insert("enc", Linear::new(10, 8))
            .insert("dec", Linear::new(8, 5));

        let params = dict.parameters_mut();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_module_dict_remove_nonexistent() {
        let mut dict = ModuleDict::new().insert("keep", Linear::new(10, 5));

        let removed = dict.remove("nonexistent");
        assert!(removed.is_none());
        assert_eq!(dict.len(), 1); // Should be unchanged
    }

    #[test]
    fn test_module_dict_insert_boxed_replace() {
        let linear1: Box<dyn Module> = Box::new(Linear::new(10, 5));
        let linear2: Box<dyn Module> = Box::new(Linear::new(10, 10));

        let dict = ModuleDict::new()
            .insert_boxed("layer", linear1)
            .insert_boxed("layer", linear2); // Replace

        assert_eq!(dict.len(), 1);
        let keys: Vec<_> = dict.keys().collect();
        assert_eq!(keys, vec!["layer"]);

        // Verify it's the second layer
        let layer = dict.get("layer").unwrap();
        let x = Tensor::ones(&[1, 10]);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), &[1, 10]); // Output from second Linear
    }

    #[test]
    fn test_sequential_empty_forward() {
        let model = Sequential::new();
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = model.forward(&x);

        // Empty sequential should return input unchanged
        assert_eq!(y.data(), x.data());
        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_module_list_empty() {
        let list = ModuleList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);

        let params = list.parameters();
        assert!(params.is_empty());
    }

    #[test]
    fn test_sequential_train_propagates() {
        let mut model = Sequential::new()
            .add(Dropout::new(0.5))
            .add(Dropout::new(0.3));

        // Start in training mode
        assert!(model.training());

        // Switch to eval
        model.eval();
        assert!(!model.training());

        // Test forward still works in eval mode
        let x = Tensor::ones(&[4, 10]);
        let _ = model.forward(&x);
    }

    #[test]
    fn test_module_list_multiple_get() {
        let list = ModuleList::new()
            .add(Linear::new(10, 8))
            .add(Linear::new(8, 5))
            .add(Linear::new(5, 3));

        // Test getting each module
        for i in 0..3 {
            let module = list.get(i);
            assert!(module.is_some(), "Module {} should exist", i);
        }
    }

    #[test]
    fn test_module_dict_iter_forward() {
        let dict = ModuleDict::new()
            .insert("l1", Linear::new(10, 8))
            .insert("l2", Linear::new(8, 5))
            .insert("l3", Linear::new(5, 3));

        // Use iter to forward through each module in sequence
        let mut x = Tensor::ones(&[2, 10]);
        for (_name, module) in dict.iter() {
            x = module.forward(&x);
        }

        // Final shape should be from l3
        assert_eq!(x.shape(), &[2, 3]);
    }

    #[test]
    fn test_sequential_deep_network() {
        // Build a deeper network
        let model = Sequential::new()
            .add(Linear::new(100, 64))
            .add(ReLU::new())
            .add(Linear::new(64, 32))
            .add(ReLU::new())
            .add(Linear::new(32, 16))
            .add(ReLU::new())
            .add(Linear::new(16, 10));

        assert_eq!(model.len(), 7);

        let x = Tensor::ones(&[8, 100]);
        let y = model.forward(&x);
        assert_eq!(y.shape(), &[8, 10]);
    }
}
