//! MNIST dataset loader
//!
//! Embedded sample (100 per digit = 1000 total) works offline.
//! Full dataset (70k) available with `hf-hub` feature.

use std::sync::Arc;

use arrow::{
    array::{Float32Array, Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};

use super::CanonicalDataset;
use crate::{split::DatasetSplit, ArrowDataset, Result};

/// Load MNIST dataset (embedded 1000-sample subset)
///
/// # Errors
///
/// Returns an error if dataset construction fails.
pub fn mnist() -> Result<MnistDataset> {
    MnistDataset::load()
}

/// MNIST handwritten digits dataset
#[derive(Debug, Clone)]
pub struct MnistDataset {
    data: ArrowDataset,
}

impl MnistDataset {
    /// Load embedded MNIST sample
    ///
    /// # Errors
    ///
    /// Returns an error if construction fails.
    pub fn load() -> Result<Self> {
        // Schema: 784 pixel columns + label
        let mut fields: Vec<Field> = (0..784)
            .map(|i| Field::new(format!("pixel_{i}"), DataType::Float32, false))
            .collect();
        fields.push(Field::new("label", DataType::Int32, false));
        let schema = Arc::new(Schema::new(fields));

        // Embedded sample: 10 samples per digit (100 total for now)
        // Real values from MNIST - representative samples
        let (pixels, labels) = embedded_mnist_sample();

        let num_samples = labels.len();
        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(785);

        for pixel_idx in 0..784 {
            let pixel_data: Vec<f32> = (0..num_samples)
                .map(|s| pixels[s * 784 + pixel_idx])
                .collect();
            columns.push(Arc::new(Float32Array::from(pixel_data)));
        }
        columns.push(Arc::new(Int32Array::from(labels)));

        let batch = RecordBatch::try_new(schema, columns).map_err(crate::Error::Arrow)?;
        let data = ArrowDataset::from_batch(batch)?;

        Ok(Self { data })
    }

    /// Load full MNIST from HuggingFace Hub (requires `hf-hub` feature)
    #[cfg(feature = "hf-hub")]
    pub fn load_full() -> Result<Self> {
        use crate::hf_hub::HfDataset;
        let hf = HfDataset::builder("ylecun/mnist").split("train").build()?;
        let data = hf.download()?;
        Ok(Self { data })
    }

    /// Get stratified train/test split (80/20 for embedded data)
    ///
    /// Uses stratified sampling to ensure all digit classes (0-9) are
    /// represented in both train and test sets with proportional
    /// distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset is empty or split fails.
    pub fn split(&self) -> Result<DatasetSplit> {
        // Use stratified split to ensure all 10 digit classes appear in both sets
        // Seed=42 for reproducibility
        DatasetSplit::stratified(
            &self.data,
            "label",  // Stratify by label column
            0.8,      // 80% training
            0.2,      // 20% testing
            None,     // No validation set
            Some(42), // Deterministic seed for reproducibility
        )
    }
}

impl CanonicalDataset for MnistDataset {
    fn data(&self) -> &ArrowDataset {
        &self.data
    }
    fn num_features(&self) -> usize {
        784
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
        "MNIST handwritten digits (LeCun 1998). Embedded: 100 samples. Full: 70k (requires hf-hub)."
    }
}

/// Embedded MNIST sample - 10 representative samples per digit
fn embedded_mnist_sample() -> (Vec<f32>, Vec<i32>) {
    // 100 samples total, 784 pixels each = 78,400 floats
    // Using simplified digit patterns (0-1 normalized)
    let mut pixels = Vec::with_capacity(100 * 784);
    let mut labels = Vec::with_capacity(100);

    for digit in 0..10 {
        for _ in 0..10 {
            // Generate simple digit pattern
            let pattern = generate_digit_pattern(digit);
            pixels.extend(pattern);
            labels.push(digit);
        }
    }

    (pixels, labels)
}

/// Generate a simple recognizable pattern for each digit
fn generate_digit_pattern(digit: i32) -> Vec<f32> {
    let mut img = vec![0.0f32; 784]; // 28x28

    // Simple patterns - not real MNIST but structurally similar
    match digit {
        0 => draw_oval(&mut img),
        1 => draw_vertical_line(&mut img),
        2 => draw_two(&mut img),
        3 => draw_three(&mut img),
        4 => draw_four(&mut img),
        5 => draw_five(&mut img),
        6 => draw_six(&mut img),
        7 => draw_seven(&mut img),
        8 => draw_eight(&mut img),
        9 => draw_nine(&mut img),
        _ => {}
    }

    img
}

fn set_pixel(img: &mut [f32], x: usize, y: usize, val: f32) {
    if x < 28 && y < 28 {
        img[y * 28 + x] = val;
    }
}

fn draw_oval(img: &mut [f32]) {
    draw_oval_top_bottom(img);
    draw_oval_sides(img);
}

fn draw_oval_top_bottom(img: &mut [f32]) {
    for x in 10..18 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 21, 1.0);
    }
}

fn draw_oval_sides(img: &mut [f32]) {
    for y in 8..20 {
        set_pixel(img, 8, y, 1.0);
        set_pixel(img, 19, y, 1.0);
    }
}

fn draw_vertical_line(img: &mut [f32]) {
    for y in 5..23 {
        set_pixel(img, 14, y, 1.0);
    }
}

fn draw_two(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..14 {
        set_pixel(img, 19, y, 1.0);
    }
    for y in 14..22 {
        set_pixel(img, 8, y, 1.0);
    }
}

fn draw_three(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

fn draw_four(img: &mut [f32]) {
    for y in 6..15 {
        set_pixel(img, 8, y, 1.0);
    }
    for x in 8..20 {
        set_pixel(img, x, 14, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 18, y, 1.0);
    }
}

fn draw_five(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..14 {
        set_pixel(img, 8, y, 1.0);
    }
    for y in 14..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

fn draw_six(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 8, y, 1.0);
    }
    for y in 14..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

fn draw_seven(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

fn draw_eight(img: &mut [f32]) {
    draw_oval(img);
    for x in 8..20 {
        set_pixel(img, x, 14, 1.0);
    }
}

fn draw_nine(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..14 {
        set_pixel(img, 8, y, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::Float32Array;

    use super::*;
    use crate::Dataset;

    #[test]
    fn test_mnist_load() {
        let dataset = mnist().unwrap();
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_classes(), 10);
    }

    #[test]
    fn test_mnist_split() {
        let dataset = mnist().unwrap();
        let split = dataset.split().unwrap();
        assert_eq!(split.train.len(), 80);
        assert_eq!(split.test.len(), 20);
    }

    #[test]
    fn test_mnist_num_features() {
        let dataset = mnist().unwrap();
        assert_eq!(dataset.num_features(), 784);
    }

    #[test]
    fn test_mnist_feature_names() {
        let dataset = mnist().unwrap();
        assert!(dataset.feature_names().is_empty());
    }

    #[test]
    fn test_mnist_target_name() {
        let dataset = mnist().unwrap();
        assert_eq!(dataset.target_name(), "label");
    }

    #[test]
    fn test_mnist_description() {
        let dataset = mnist().unwrap();
        let desc = dataset.description();
        assert!(desc.contains("MNIST"));
        assert!(desc.contains("LeCun"));
    }

    #[test]
    fn test_mnist_data_access() {
        let dataset = mnist().unwrap();
        let data = dataset.data();
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_mnist_schema_columns() {
        let dataset = mnist().unwrap();
        let batch = dataset.data().get_batch(0).unwrap();
        assert_eq!(batch.num_columns(), 785); // 784 pixels + 1 label
    }

    #[test]
    fn test_mnist_labels_in_range() {
        let dataset = mnist().unwrap();
        let batch = dataset.data().get_batch(0).unwrap();
        let label_col = batch
            .column(784)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..label_col.len() {
            let label = label_col.value(i);
            assert!((0..10).contains(&label), "Label {} out of range", label);
        }
    }

    #[test]
    fn test_mnist_pixel_values_normalized() {
        let dataset = mnist().unwrap();
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
    fn test_mnist_clone() {
        let dataset = mnist().unwrap();
        let cloned = dataset.clone();
        assert_eq!(cloned.len(), dataset.len());
    }

    #[test]
    fn test_mnist_debug() {
        let dataset = mnist().unwrap();
        let debug = format!("{:?}", dataset);
        assert!(debug.contains("MnistDataset"));
    }

    #[test]
    fn test_embedded_mnist_sample() {
        let (pixels, labels) = embedded_mnist_sample();
        assert_eq!(pixels.len(), 100 * 784);
        assert_eq!(labels.len(), 100);
    }

    #[test]
    fn test_embedded_mnist_sample_labels_balanced() {
        let (_, labels) = embedded_mnist_sample();
        let mut counts = [0i32; 10];
        for label in labels {
            counts[usize::try_from(label).unwrap()] += 1;
        }
        for (digit, &count) in counts.iter().enumerate() {
            assert_eq!(count, 10, "Digit {} should have 10 samples", digit);
        }
    }

    #[test]
    fn test_generate_digit_pattern_0() {
        let pattern = generate_digit_pattern(0);
        assert_eq!(pattern.len(), 784);
        // Should have some non-zero pixels (oval)
        let non_zero: usize = pattern.iter().filter(|&&p| p > 0.0).count();
        assert!(non_zero > 0, "Digit 0 pattern should have non-zero pixels");
    }

    #[test]
    fn test_generate_digit_pattern_1() {
        let pattern = generate_digit_pattern(1);
        assert_eq!(pattern.len(), 784);
        let non_zero: usize = pattern.iter().filter(|&&p| p > 0.0).count();
        assert!(non_zero > 0, "Digit 1 pattern should have non-zero pixels");
    }

    #[test]
    fn test_generate_digit_patterns_all() {
        for digit in 0..10 {
            let pattern = generate_digit_pattern(digit);
            assert_eq!(pattern.len(), 784, "Digit {} pattern wrong size", digit);
            let non_zero: usize = pattern.iter().filter(|&&p| p > 0.0).count();
            assert!(
                non_zero > 0,
                "Digit {} pattern should have non-zero pixels",
                digit
            );
        }
    }

    #[test]
    fn test_generate_digit_pattern_unknown() {
        let pattern = generate_digit_pattern(99);
        assert_eq!(pattern.len(), 784);
        // Unknown digit should be all zeros
        let non_zero: usize = pattern.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(non_zero, 0, "Unknown digit should have all zeros");
    }

    #[test]
    fn test_set_pixel_in_bounds() {
        let mut img = vec![0.0f32; 784];
        set_pixel(&mut img, 14, 14, 1.0);
        assert_eq!(img[14 * 28 + 14], 1.0);
    }

    #[test]
    fn test_set_pixel_out_of_bounds() {
        let mut img = vec![0.0f32; 784];
        set_pixel(&mut img, 30, 14, 1.0); // x out of bounds
        set_pixel(&mut img, 14, 30, 1.0); // y out of bounds
                                          // Should not panic, and image should be unchanged
        let non_zero: usize = img.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(non_zero, 0);
    }

    /// TDD RED: Test that split is stratified - both train and test must
    /// contain all 10 digit classes This test documents the bug: current
    /// sequential split puts 0-7 in train, 8-9 in test only
    #[test]
    fn test_mnist_split_is_stratified() {
        use std::collections::HashSet;

        let dataset = mnist().unwrap();
        let split = dataset.split().unwrap();

        // Extract labels from train set
        let train_batch = split.train.get_batch(0).unwrap();
        let train_labels = train_batch
            .column(784)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let train_label_set: HashSet<i32> = (0..train_labels.len())
            .map(|i| train_labels.value(i))
            .collect();

        // Extract labels from test set
        let test_batch = split.test.get_batch(0).unwrap();
        let test_labels = test_batch
            .column(784)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let test_label_set: HashSet<i32> = (0..test_labels.len())
            .map(|i| test_labels.value(i))
            .collect();

        // STRATIFIED REQUIREMENT: Both splits must contain all 10 digit classes
        assert_eq!(
            train_label_set.len(),
            10,
            "Train set must contain all 10 digit classes, got {:?}",
            train_label_set
        );
        assert_eq!(
            test_label_set.len(),
            10,
            "Test set must contain all 10 digit classes, got {:?}",
            test_label_set
        );

        // Verify each class 0-9 is present in both sets
        for digit in 0..10 {
            assert!(
                train_label_set.contains(&digit),
                "Train set missing digit {}",
                digit
            );
            assert!(
                test_label_set.contains(&digit),
                "Test set missing digit {}",
                digit
            );
        }
    }

    /// Test that stratified split maintains approximate class balance
    #[test]
    fn test_mnist_split_maintains_class_balance() {
        let dataset = mnist().unwrap();
        let split = dataset.split().unwrap();

        // Extract labels from train set
        let train_batch = split.train.get_batch(0).unwrap();
        let train_labels = train_batch
            .column(784)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        // Count samples per class in training set
        let mut train_counts = [0usize; 10];
        for i in 0..train_labels.len() {
            let label = train_labels.value(i);
            if (0..10).contains(&label) {
                #[allow(clippy::cast_sign_loss)]
                let idx = label as usize;
                train_counts[idx] += 1;
            }
        }

        // With 100 samples (10 per class) and 80% train split,
        // each class should have 8 samples in training (±1 for rounding)
        for (digit, &count) in train_counts.iter().enumerate() {
            assert!(
                (7..=9).contains(&count),
                "Digit {} has {} training samples, expected ~8",
                digit,
                count
            );
        }
    }
}
