//! Canonical ML dataset loaders
//!
//! Provides convenient one-liner access to well-known ML datasets
//! for tutorials, examples, and benchmarking.
//!
//! # Example
//!
//! ```ignore
//! use alimentar::datasets::{iris, mnist, cifar10};
//!
//! // Load Iris dataset (embedded, no download)
//! let iris = iris()?;
//! println!("Iris: {} samples", iris.len());
//!
//! // Load MNIST (downloads from HuggingFace Hub on first use)
//! let mnist = mnist()?;
//! let (train, test) = mnist.split()?;
//! ```

mod cifar10;
mod cifar100;
mod fashion_mnist;
mod iris;
mod mnist;

pub use cifar10::{cifar10, Cifar10Dataset, CIFAR10_CLASSES};
pub use cifar100::{cifar100, Cifar100Dataset, CIFAR100_COARSE_CLASSES, CIFAR100_FINE_CLASSES};
pub use fashion_mnist::{fashion_mnist, FashionMnistDataset, FASHION_MNIST_CLASSES};
pub use iris::{iris, IrisDataset};
pub use mnist::{mnist, MnistDataset};

use crate::{ArrowDataset, Dataset};

/// A canonical ML dataset with train/test split support
pub trait CanonicalDataset {
    /// Returns the full dataset
    fn data(&self) -> &ArrowDataset;

    /// Returns the number of samples
    fn len(&self) -> usize {
        self.data().len()
    }

    /// Returns true if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of features (excluding label)
    fn num_features(&self) -> usize;

    /// Returns the number of classes (for classification datasets)
    fn num_classes(&self) -> usize;

    /// Returns the feature column names
    fn feature_names(&self) -> &'static [&'static str];

    /// Returns the label/target column name
    fn target_name(&self) -> &'static str;

    /// Returns a description of the dataset
    fn description(&self) -> &'static str;
}

/// Split information for train/test datasets
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    /// Training dataset
    pub train: ArrowDataset,
    /// Test dataset
    pub test: ArrowDataset,
}

impl DatasetSplit {
    /// Create a new dataset split
    pub fn new(train: ArrowDataset, test: ArrowDataset) -> Self {
        Self { train, test }
    }

    /// Get training data
    pub fn train(&self) -> &ArrowDataset {
        &self.train
    }

    /// Get test data
    pub fn test(&self) -> &ArrowDataset {
        &self.test
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_split_new() {
        let train = iris::iris()
            .ok()
            .unwrap_or_else(|| panic!("Should load iris"))
            .data()
            .clone();
        let test = train.clone();

        let split = DatasetSplit::new(train.clone(), test.clone());
        assert_eq!(split.train().len(), train.len());
        assert_eq!(split.test().len(), test.len());
    }

    #[test]
    fn test_dataset_split_debug() {
        let train = iris::iris()
            .ok()
            .unwrap_or_else(|| panic!("Should load iris"))
            .data()
            .clone();
        let test = train.clone();

        let split = DatasetSplit::new(train, test);
        let debug = format!("{:?}", split);
        assert!(debug.contains("DatasetSplit"));
    }

    #[test]
    fn test_dataset_split_clone() {
        let train = iris::iris()
            .ok()
            .unwrap_or_else(|| panic!("Should load iris"))
            .data()
            .clone();
        let test = train.clone();

        let split = DatasetSplit::new(train, test);
        let cloned = split.clone();
        assert_eq!(cloned.train().len(), split.train().len());
    }

    #[test]
    fn test_canonical_dataset_is_empty() {
        let iris = iris::iris()
            .ok()
            .unwrap_or_else(|| panic!("Should load iris"));
        assert!(!iris.is_empty());
    }
}
