//! CIFAR-10 dataset loader
//!
//! Embedded sample (10 per class = 100 total) works offline.
//! Full dataset (60k) available with `hf-hub` feature.

use std::sync::Arc;

use arrow::{
    array::{Float32Array, Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};

use super::{CanonicalDataset, DatasetSplit};
use crate::{
    transform::{Skip, Take, Transform},
    ArrowDataset, Dataset, Result,
};

/// CIFAR-10 class names
pub const CIFAR10_CLASSES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

/// Load CIFAR-10 dataset (embedded 100-sample subset)
///
/// # Errors
///
/// Returns an error if dataset construction fails.
pub fn cifar10() -> Result<Cifar10Dataset> {
    Cifar10Dataset::load()
}

/// CIFAR-10 image classification dataset
#[derive(Debug, Clone)]
pub struct Cifar10Dataset {
    data: ArrowDataset,
}

impl Cifar10Dataset {
    /// Load embedded CIFAR-10 sample
    ///
    /// # Errors
    ///
    /// Returns an error if construction fails.
    pub fn load() -> Result<Self> {
        // Schema: 3072 pixel columns (32x32x3 RGB) + label
        let mut fields: Vec<Field> = (0..3072)
            .map(|i| Field::new(format!("pixel_{i}"), DataType::Float32, false))
            .collect();
        fields.push(Field::new("label", DataType::Int32, false));
        let schema = Arc::new(Schema::new(fields));

        let (pixels, labels) = embedded_cifar10_sample();
        let num_samples = labels.len();

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(3073);
        for pixel_idx in 0..3072 {
            let pixel_data: Vec<f32> = (0..num_samples)
                .map(|s| pixels[s * 3072 + pixel_idx])
                .collect();
            columns.push(Arc::new(Float32Array::from(pixel_data)));
        }
        columns.push(Arc::new(Int32Array::from(labels)));

        let batch = RecordBatch::try_new(schema, columns).map_err(crate::Error::Arrow)?;
        let data = ArrowDataset::from_batch(batch)?;

        Ok(Self { data })
    }

    /// Load full CIFAR-10 from HuggingFace Hub
    #[cfg(feature = "hf-hub")]
    pub fn load_full() -> Result<Self> {
        use crate::hf_hub::HfDataset;
        let hf = HfDataset::builder("uoft-cs/cifar10")
            .split("train")
            .build()?;
        let data = hf.download()?;
        Ok(Self { data })
    }

    /// Get train/test split (80/20)
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset is empty or split fails.
    pub fn split(&self) -> Result<DatasetSplit> {
        let len = self.data.len();
        let train_size = (len * 8) / 10;

        let batch = self
            .data
            .get_batch(0)
            .ok_or_else(|| crate::Error::empty_dataset("CIFAR-10"))?;

        let train_batch = Take::new(train_size).apply(batch.clone())?;
        let test_batch = Skip::new(train_size).apply(batch.clone())?;

        Ok(DatasetSplit::new(
            ArrowDataset::from_batch(train_batch)?,
            ArrowDataset::from_batch(test_batch)?,
        ))
    }

    /// Get class name for a label
    #[must_use]
    pub fn class_name(label: i32) -> Option<&'static str> {
        if label < 0 {
            return None;
        }
        CIFAR10_CLASSES.get(usize::try_from(label).ok()?).copied()
    }
}

impl CanonicalDataset for Cifar10Dataset {
    fn data(&self) -> &ArrowDataset {
        &self.data
    }
    fn num_features(&self) -> usize {
        3072
    }
    fn num_classes(&self) -> usize {
        10
    }
    fn feature_names(&self) -> &'static [&'static str] {
        &[]
    }
    fn target_name(&self) -> &'static str {
        "label"
    }
    fn description(&self) -> &'static str {
        "CIFAR-10 (Krizhevsky 2009). Embedded: 100 samples. Full: 60k (requires hf-hub)."
    }
}

/// Embedded CIFAR-10 sample - 10 per class with simple color patterns
#[allow(clippy::cast_precision_loss)]
fn embedded_cifar10_sample() -> (Vec<f32>, Vec<i32>) {
    let mut pixels = Vec::with_capacity(100 * 3072);
    let mut labels = Vec::with_capacity(100);

    // Simple color patterns per class
    let class_colors: [(f32, f32, f32); 10] = [
        (0.5, 0.7, 0.9), // airplane - sky blue
        (0.3, 0.3, 0.3), // automobile - gray
        (0.6, 0.4, 0.2), // bird - brown
        (0.8, 0.6, 0.4), // cat - orange
        (0.4, 0.3, 0.2), // deer - brown
        (0.7, 0.5, 0.3), // dog - tan
        (0.2, 0.8, 0.2), // frog - green
        (0.5, 0.3, 0.2), // horse - brown
        (0.2, 0.3, 0.5), // ship - navy
        (0.6, 0.2, 0.2), // truck - red
    ];

    for (class_idx, &(r, g, b)) in class_colors.iter().enumerate() {
        for sample in 0..10i16 {
            // Add variation per sample
            let var = f32::from(sample) * 0.02;
            for _ in 0..1024 {
                pixels.push((r + var).min(1.0));
            } // R channel
            for _ in 0..1024 {
                pixels.push((g + var).min(1.0));
            } // G channel
            for _ in 0..1024 {
                pixels.push((b + var).min(1.0));
            } // B channel
              // class_idx is always 0-9, safe truncation
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            labels.push(class_idx as i32);
        }
    }

    (pixels, labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn test_cifar10_load() {
        let dataset = cifar10().unwrap();
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_classes(), 10);
    }

    #[test]
    fn test_cifar10_split() {
        let dataset = cifar10().unwrap();
        let split = dataset.split().unwrap();
        assert_eq!(split.train.len(), 80);
        assert_eq!(split.test.len(), 20);
    }

    #[test]
    fn test_cifar10_class_names() {
        assert_eq!(Cifar10Dataset::class_name(0), Some("airplane"));
        assert_eq!(Cifar10Dataset::class_name(9), Some("truck"));
        assert_eq!(Cifar10Dataset::class_name(10), None);
    }

    #[test]
    fn test_cifar10_class_name_negative() {
        assert_eq!(Cifar10Dataset::class_name(-1), None);
        assert_eq!(Cifar10Dataset::class_name(-100), None);
    }

    #[test]
    fn test_cifar10_all_class_names() {
        for (idx, &expected) in CIFAR10_CLASSES.iter().enumerate() {
            assert_eq!(Cifar10Dataset::class_name(idx as i32), Some(expected));
        }
    }

    #[test]
    fn test_cifar10_num_features() {
        let dataset = cifar10().unwrap();
        assert_eq!(dataset.num_features(), 3072);
    }

    #[test]
    fn test_cifar10_feature_names() {
        let dataset = cifar10().unwrap();
        assert!(dataset.feature_names().is_empty());
    }

    #[test]
    fn test_cifar10_target_name() {
        let dataset = cifar10().unwrap();
        assert_eq!(dataset.target_name(), "label");
    }

    #[test]
    fn test_cifar10_description() {
        let dataset = cifar10().unwrap();
        let desc = dataset.description();
        assert!(desc.contains("CIFAR-10"));
        assert!(desc.contains("100 samples"));
    }

    #[test]
    fn test_cifar10_data_access() {
        let dataset = cifar10().unwrap();
        let data = dataset.data();
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_cifar10_schema_columns() {
        let dataset = cifar10().unwrap();
        let batch = dataset.data().get_batch(0).unwrap();
        assert_eq!(batch.num_columns(), 3073); // 3072 pixels + 1 label
    }

    #[test]
    fn test_cifar10_pixel_values_normalized() {
        let dataset = cifar10().unwrap();
        let batch = dataset.data().get_batch(0).unwrap();
        let pixel_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 0..pixel_col.len() {
            let val = pixel_col.value(i);
            assert!(
                (0.0..=1.0).contains(&val),
                "Pixel value {} out of range",
                val
            );
        }
    }

    #[test]
    fn test_cifar10_labels_in_range() {
        let dataset = cifar10().unwrap();
        let batch = dataset.data().get_batch(0).unwrap();
        let label_col = batch
            .column(3072)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..label_col.len() {
            let label = label_col.value(i);
            assert!((0..10).contains(&label), "Label {} out of range", label);
        }
    }

    #[test]
    fn test_cifar10_clone() {
        let dataset = cifar10().unwrap();
        let cloned = dataset.clone();
        assert_eq!(cloned.len(), dataset.len());
    }

    #[test]
    fn test_cifar10_debug() {
        let dataset = cifar10().unwrap();
        let debug = format!("{:?}", dataset);
        assert!(debug.contains("Cifar10Dataset"));
    }

    #[test]
    fn test_embedded_cifar10_sample() {
        let (pixels, labels) = embedded_cifar10_sample();
        assert_eq!(pixels.len(), 100 * 3072);
        assert_eq!(labels.len(), 100);
    }

    #[test]
    fn test_embedded_cifar10_sample_labels_balanced() {
        let (_, labels) = embedded_cifar10_sample();
        let mut counts = [0i32; 10];
        for label in labels {
            counts[usize::try_from(label).unwrap()] += 1;
        }
        for (i, &count) in counts.iter().enumerate() {
            assert_eq!(count, 10, "Class {} should have 10 samples", i);
        }
    }

    #[test]
    fn test_cifar10_classes_constant() {
        assert_eq!(CIFAR10_CLASSES.len(), 10);
        assert_eq!(CIFAR10_CLASSES[0], "airplane");
        assert_eq!(CIFAR10_CLASSES[9], "truck");
    }
}
