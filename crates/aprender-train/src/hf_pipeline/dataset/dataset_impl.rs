//! Dataset struct and implementation

use super::example::Example;

/// Dataset abstraction
pub struct Dataset {
    /// Dataset name/ID
    name: String,
    /// Examples
    examples: Vec<Example>,
    /// Current position for iteration
    position: usize,
}

impl Dataset {
    /// Create new dataset from examples
    #[must_use]
    pub fn new(name: impl Into<String>, examples: Vec<Example>) -> Self {
        Self { name: name.into(), examples, position: 0 }
    }

    /// Create mock dataset for testing
    #[must_use]
    pub fn mock(num_examples: usize, seq_len: usize) -> Self {
        let examples: Vec<Example> = (0..num_examples)
            .map(|i| {
                Example::from_tokens((0..seq_len).map(|j| ((i + j) % 30000) as u32).collect())
                    .with_labels((0..seq_len).map(|j| ((i + j + 1) % 30000) as u32).collect())
            })
            .collect();

        Self::new("mock_dataset", examples)
    }

    /// Get dataset name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get number of examples
    #[must_use]
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get example by index
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Example> {
        self.examples.get(index)
    }

    /// Get all examples
    #[must_use]
    pub fn examples(&self) -> &[Example] {
        &self.examples
    }

    /// Reset iteration position
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Shuffle examples
    pub fn shuffle(&mut self, seed: u64) {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        self.examples.shuffle(&mut rng);
    }

    /// Take a subset of examples
    #[must_use]
    pub fn take(mut self, n: usize) -> Self {
        self.examples.truncate(n);
        self
    }
}

impl Iterator for Dataset {
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.examples.len() {
            let example = self.examples[self.position].clone();
            self.position += 1;
            Some(example)
        } else {
            None
        }
    }
}
