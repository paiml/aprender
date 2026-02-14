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
///         out = layers.get(i).expect("layer index in bounds").forward(&out);
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
/// let encoded = modules.get("encoder").expect("encoder module exists").forward(&x);
/// let decoded = modules.get("decoder").expect("decoder module exists").forward(&encoded);
/// let logits = modules.get("classifier").expect("classifier module exists").forward(&encoded);
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
#[path = "container_tests.rs"]
mod tests;
