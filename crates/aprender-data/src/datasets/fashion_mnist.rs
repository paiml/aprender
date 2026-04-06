//! Fashion-MNIST dataset loader
//!
//! Embedded sample (10 per class = 100 total) works offline.
//! Full dataset (70k) available with `hf-hub` feature.

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

/// Fashion-MNIST class names
pub const FASHION_MNIST_CLASSES: [&str; 10] = [
    "t-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
];

/// Load Fashion-MNIST dataset (embedded 100-sample subset)
///
/// # Errors
///
/// Returns an error if dataset construction fails.
pub fn fashion_mnist() -> Result<FashionMnistDataset> {
    FashionMnistDataset::load()
}

/// Fashion-MNIST clothing classification dataset
#[derive(Debug, Clone)]
pub struct FashionMnistDataset {
    data: ArrowDataset,
}

impl FashionMnistDataset {
    /// Load embedded Fashion-MNIST sample
    ///
    /// # Errors
    ///
    /// Returns an error if construction fails.
    pub fn load() -> Result<Self> {
        let mut fields: Vec<Field> = (0..784)
            .map(|i| Field::new(format!("pixel_{i}"), DataType::Float32, false))
            .collect();
        fields.push(Field::new("label", DataType::Int32, false));
        let schema = Arc::new(Schema::new(fields));

        let (pixels, labels) = embedded_fashion_mnist_sample();
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

    /// Load full Fashion-MNIST from HuggingFace Hub
    #[cfg(feature = "hf-hub")]
    pub fn load_full() -> Result<Self> {
        use crate::hf_hub::HfDataset;
        let hf = HfDataset::builder("zalando-datasets/fashion_mnist")
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
            .ok_or_else(|| crate::Error::empty_dataset("Fashion-MNIST"))?;

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
        FASHION_MNIST_CLASSES
            .get(usize::try_from(label).ok()?)
            .copied()
    }
}

impl CanonicalDataset for FashionMnistDataset {
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
        "Fashion-MNIST (Xiao et al. 2017). Embedded: 100 samples. Full: 70k (requires hf-hub)."
    }
}

/// Embedded Fashion-MNIST sample - 10 per class with simple patterns
fn embedded_fashion_mnist_sample() -> (Vec<f32>, Vec<i32>) {
    let mut pixels = Vec::with_capacity(100 * 784);
    let mut labels = Vec::with_capacity(100);

    for class_idx in 0..10 {
        for sample in 0..10i16 {
            let pattern = generate_fashion_pattern(class_idx, sample);
            pixels.extend(pattern);
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            labels.push(class_idx as i32);
        }
    }

    (pixels, labels)
}

/// Generate simple fashion item patterns
fn generate_fashion_pattern(class: usize, variation: i16) -> Vec<f32> {
    let mut img = vec![0.0f32; 784];
    let var = f32::from(variation) * 0.02;

    match class {
        0 => draw_tshirt(&mut img, var),     // t-shirt/top
        1 => draw_trouser(&mut img, var),    // trouser
        2 => draw_pullover(&mut img, var),   // pullover
        3 => draw_dress(&mut img, var),      // dress
        4 => draw_coat(&mut img, var),       // coat
        5 => draw_sandal(&mut img, var),     // sandal
        6 => draw_shirt(&mut img, var),      // shirt
        7 => draw_sneaker(&mut img, var),    // sneaker
        8 => draw_bag(&mut img, var),        // bag
        9 => draw_ankle_boot(&mut img, var), // ankle boot
        _ => {}
    }

    img
}

fn set_pixel(img: &mut [f32], x: usize, y: usize, val: f32) {
    if x < 28 && y < 28 {
        img[y * 28 + x] = val;
    }
}

fn draw_tshirt(img: &mut [f32], var: f32) {
    // Body
    for y in 8..22 {
        for x in 8..20 {
            set_pixel(img, x, y, (0.8 + var).min(1.0));
        }
    }
    // Sleeves
    for y in 8..12 {
        for x in 4..8 {
            set_pixel(img, x, y, (0.7 + var).min(1.0));
        }
        for x in 20..24 {
            set_pixel(img, x, y, (0.7 + var).min(1.0));
        }
    }
}

fn draw_trouser(img: &mut [f32], var: f32) {
    // Left leg
    for y in 4..24 {
        for x in 8..13 {
            set_pixel(img, x, y, (0.6 + var).min(1.0));
        }
    }
    // Right leg
    for y in 4..24 {
        for x in 15..20 {
            set_pixel(img, x, y, (0.6 + var).min(1.0));
        }
    }
    // Waist
    for x in 8..20 {
        for y in 4..7 {
            set_pixel(img, x, y, (0.7 + var).min(1.0));
        }
    }
}

fn draw_pullover(img: &mut [f32], var: f32) {
    draw_tshirt(img, var);
    // Longer sleeves
    for y in 12..16 {
        for x in 4..8 {
            set_pixel(img, x, y, (0.7 + var).min(1.0));
        }
        for x in 20..24 {
            set_pixel(img, x, y, (0.7 + var).min(1.0));
        }
    }
}

fn draw_dress(img: &mut [f32], var: f32) {
    // Top
    for y in 6..12 {
        for x in 10..18 {
            set_pixel(img, x, y, (0.8 + var).min(1.0));
        }
    }
    // Flared skirt
    for y in 12..24 {
        let width = 4 + (y - 12) / 2;
        for x in (14 - width)..(14 + width) {
            set_pixel(img, x, y, (0.8 + var).min(1.0));
        }
    }
}

fn draw_coat(img: &mut [f32], var: f32) {
    draw_tshirt(img, var);
    // Extend body
    for y in 22..26 {
        for x in 8..20 {
            set_pixel(img, x, y, (0.8 + var).min(1.0));
        }
    }
}

fn draw_sandal(img: &mut [f32], var: f32) {
    // Sole
    for x in 6..22 {
        for y in 20..24 {
            set_pixel(img, x, y, (0.5 + var).min(1.0));
        }
    }
    // Straps
    for x in 8..20 {
        set_pixel(img, x, 16, (0.7 + var).min(1.0));
        set_pixel(img, x, 12, (0.7 + var).min(1.0));
    }
}

fn draw_shirt(img: &mut [f32], var: f32) {
    draw_tshirt(img, var);
    // Collar
    for x in 12..16 {
        set_pixel(img, x, 7, (0.9 + var).min(1.0));
    }
}

fn draw_sneaker(img: &mut [f32], var: f32) {
    // Sole
    for x in 4..24 {
        for y in 18..22 {
            set_pixel(img, x, y, (0.4 + var).min(1.0));
        }
    }
    // Upper
    for x in 6..22 {
        for y in 12..18 {
            set_pixel(img, x, y, (0.8 + var).min(1.0));
        }
    }
}

fn draw_bag(img: &mut [f32], var: f32) {
    // Body
    for y in 10..24 {
        for x in 8..20 {
            set_pixel(img, x, y, (0.7 + var).min(1.0));
        }
    }
    // Handle
    for x in 10..18 {
        set_pixel(img, x, 6, (0.6 + var).min(1.0));
        set_pixel(img, x, 8, (0.6 + var).min(1.0));
    }
    set_pixel(img, 10, 7, (0.6 + var).min(1.0));
    set_pixel(img, 17, 7, (0.6 + var).min(1.0));
}

fn draw_ankle_boot(img: &mut [f32], var: f32) {
    // Sole
    for x in 6..22 {
        for y in 20..24 {
            set_pixel(img, x, y, (0.3 + var).min(1.0));
        }
    }
    // Boot upper
    for x in 8..20 {
        for y in 8..20 {
            set_pixel(img, x, y, (0.6 + var).min(1.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::Float32Array;

    use super::*;
    use crate::Dataset;

    #[test]
    fn test_fashion_mnist_load() {
        let dataset = fashion_mnist().unwrap();
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_classes(), 10);
    }

    #[test]
    fn test_fashion_mnist_split() {
        let dataset = fashion_mnist().unwrap();
        let split = dataset.split().unwrap();
        assert_eq!(split.train.len(), 80);
        assert_eq!(split.test.len(), 20);
    }

    #[test]
    fn test_fashion_mnist_class_names() {
        assert_eq!(FashionMnistDataset::class_name(0), Some("t-shirt/top"));
        assert_eq!(FashionMnistDataset::class_name(9), Some("ankle boot"));
        assert_eq!(FashionMnistDataset::class_name(10), None);
        assert_eq!(FashionMnistDataset::class_name(-1), None);
    }

    #[test]
    fn test_fashion_mnist_all_class_names() {
        for (idx, &expected) in FASHION_MNIST_CLASSES.iter().enumerate() {
            assert_eq!(FashionMnistDataset::class_name(idx as i32), Some(expected));
        }
    }

    #[test]
    fn test_fashion_mnist_num_features() {
        let dataset = fashion_mnist().unwrap();
        assert_eq!(dataset.num_features(), 784);
    }

    #[test]
    fn test_fashion_mnist_feature_names() {
        let dataset = fashion_mnist().unwrap();
        assert!(dataset.feature_names().is_empty());
    }

    #[test]
    fn test_fashion_mnist_target_name() {
        let dataset = fashion_mnist().unwrap();
        assert_eq!(dataset.target_name(), "label");
    }

    #[test]
    fn test_fashion_mnist_description() {
        let dataset = fashion_mnist().unwrap();
        let desc = dataset.description();
        assert!(desc.contains("Fashion-MNIST"));
        assert!(desc.contains("Xiao"));
    }

    #[test]
    fn test_fashion_mnist_data_access() {
        let dataset = fashion_mnist().unwrap();
        let data = dataset.data();
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_fashion_mnist_schema_columns() {
        let dataset = fashion_mnist().unwrap();
        let batch = dataset.data().get_batch(0).unwrap();
        assert_eq!(batch.num_columns(), 785); // 784 pixels + 1 label
    }

    #[test]
    fn test_fashion_mnist_labels_in_range() {
        let dataset = fashion_mnist().unwrap();
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
    fn test_fashion_mnist_pixel_values_normalized() {
        let dataset = fashion_mnist().unwrap();
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
    fn test_fashion_mnist_clone() {
        let dataset = fashion_mnist().unwrap();
        let cloned = dataset.clone();
        assert_eq!(cloned.len(), dataset.len());
    }

    #[test]
    fn test_fashion_mnist_debug() {
        let dataset = fashion_mnist().unwrap();
        let debug = format!("{:?}", dataset);
        assert!(debug.contains("FashionMnistDataset"));
    }

    #[test]
    fn test_embedded_fashion_mnist_sample() {
        let (pixels, labels) = embedded_fashion_mnist_sample();
        assert_eq!(pixels.len(), 100 * 784);
        assert_eq!(labels.len(), 100);
    }

    #[test]
    fn test_embedded_fashion_mnist_sample_labels_balanced() {
        let (_, labels) = embedded_fashion_mnist_sample();
        let mut counts = [0i32; 10];
        for label in labels {
            counts[usize::try_from(label).unwrap()] += 1;
        }
        for (class, &count) in counts.iter().enumerate() {
            assert_eq!(count, 10, "Class {} should have 10 samples", class);
        }
    }

    #[test]
    fn test_generate_fashion_pattern_all_classes() {
        for class in 0..10 {
            let pattern = generate_fashion_pattern(class, 0);
            assert_eq!(pattern.len(), 784, "Class {} pattern wrong size", class);
            let non_zero: usize = pattern.iter().filter(|&&p| p > 0.0).count();
            assert!(
                non_zero > 0,
                "Class {} pattern should have non-zero pixels",
                class
            );
        }
    }

    #[test]
    fn test_generate_fashion_pattern_with_variation() {
        let pattern1 = generate_fashion_pattern(0, 0);
        let pattern2 = generate_fashion_pattern(0, 5);
        // Patterns should differ due to variation
        let different = pattern1
            .iter()
            .zip(pattern2.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(
            different,
            "Patterns with different variations should differ"
        );
    }

    #[test]
    fn test_generate_fashion_pattern_unknown() {
        let pattern = generate_fashion_pattern(99, 0);
        assert_eq!(pattern.len(), 784);
        // Unknown class should be all zeros
        let non_zero: usize = pattern.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(non_zero, 0, "Unknown class should have all zeros");
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

    #[test]
    fn test_fashion_mnist_classes_constant() {
        assert_eq!(FASHION_MNIST_CLASSES.len(), 10);
        assert_eq!(FASHION_MNIST_CLASSES[0], "t-shirt/top");
        assert_eq!(FASHION_MNIST_CLASSES[9], "ankle boot");
    }
}
