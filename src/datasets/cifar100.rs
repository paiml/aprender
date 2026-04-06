//! CIFAR-100 dataset loader
//!
//! Embedded sample (1 per class = 100 total) works offline.
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

/// CIFAR-100 fine class names (100 classes)
pub const CIFAR100_FINE_CLASSES: [&str; 100] = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
];

/// CIFAR-100 coarse class names (20 superclasses)
pub const CIFAR100_COARSE_CLASSES: [&str; 20] = [
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
];

/// Load CIFAR-100 dataset (embedded 100-sample subset)
///
/// # Errors
///
/// Returns an error if dataset construction fails.
pub fn cifar100() -> Result<Cifar100Dataset> {
    Cifar100Dataset::load()
}

/// CIFAR-100 image classification dataset
#[derive(Debug, Clone)]
pub struct Cifar100Dataset {
    data: ArrowDataset,
}

impl Cifar100Dataset {
    /// Load embedded CIFAR-100 sample
    ///
    /// # Errors
    ///
    /// Returns an error if construction fails.
    pub fn load() -> Result<Self> {
        let mut fields: Vec<Field> = (0..3072)
            .map(|i| Field::new(format!("pixel_{i}"), DataType::Float32, false))
            .collect();
        fields.push(Field::new("fine_label", DataType::Int32, false));
        fields.push(Field::new("coarse_label", DataType::Int32, false));
        let schema = Arc::new(Schema::new(fields));

        let (pixels, fine_labels, coarse_labels) = embedded_cifar100_sample();
        let num_samples = fine_labels.len();

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(3074);
        for pixel_idx in 0..3072 {
            let pixel_data: Vec<f32> = (0..num_samples)
                .map(|s| pixels[s * 3072 + pixel_idx])
                .collect();
            columns.push(Arc::new(Float32Array::from(pixel_data)));
        }
        columns.push(Arc::new(Int32Array::from(fine_labels)));
        columns.push(Arc::new(Int32Array::from(coarse_labels)));

        let batch = RecordBatch::try_new(schema, columns).map_err(crate::Error::Arrow)?;
        let data = ArrowDataset::from_batch(batch)?;

        Ok(Self { data })
    }

    /// Load full CIFAR-100 from HuggingFace Hub
    #[cfg(feature = "hf-hub")]
    pub fn load_full() -> Result<Self> {
        use crate::hf_hub::HfDataset;
        let hf = HfDataset::builder("uoft-cs/cifar100")
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
            .ok_or_else(|| crate::Error::empty_dataset("CIFAR-100"))?;

        let train_batch = Take::new(train_size).apply(batch.clone())?;
        let test_batch = Skip::new(train_size).apply(batch.clone())?;

        Ok(DatasetSplit::new(
            ArrowDataset::from_batch(train_batch)?,
            ArrowDataset::from_batch(test_batch)?,
        ))
    }

    /// Get fine class name for a label (100 classes)
    #[must_use]
    pub fn fine_class_name(label: i32) -> Option<&'static str> {
        if label < 0 {
            return None;
        }
        CIFAR100_FINE_CLASSES
            .get(usize::try_from(label).ok()?)
            .copied()
    }

    /// Get coarse class name for a label (20 superclasses)
    #[must_use]
    pub fn coarse_class_name(label: i32) -> Option<&'static str> {
        if label < 0 {
            return None;
        }
        CIFAR100_COARSE_CLASSES
            .get(usize::try_from(label).ok()?)
            .copied()
    }
}

impl CanonicalDataset for Cifar100Dataset {
    fn data(&self) -> &ArrowDataset {
        &self.data
    }
    fn num_features(&self) -> usize {
        3072
    }
    fn num_classes(&self) -> usize {
        100
    }
    fn feature_names(&self) -> &'static [&'static str] {
        &[]
    }
    fn target_name(&self) -> &'static str {
        "fine_label"
    }
    fn description(&self) -> &'static str {
        "CIFAR-100 (Krizhevsky 2009). 100 fine classes, 20 coarse. Embedded: 100. Full: 60k."
    }
}

/// Fine-to-coarse label mapping
const FINE_TO_COARSE: [usize; 100] = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15,
    3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4,
    2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
];

/// Embedded CIFAR-100 sample - 1 per fine class
#[allow(clippy::cast_precision_loss)]
fn embedded_cifar100_sample() -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let mut pixels = Vec::with_capacity(100 * 3072);
    let mut fine_labels = Vec::with_capacity(100);
    let mut coarse_labels = Vec::with_capacity(100);

    // Generate unique color for each of 100 classes
    for (class_idx, &coarse_idx) in FINE_TO_COARSE.iter().enumerate() {
        // Deterministic color based on class index (values 0-99, safe precision)
        let r = ((class_idx * 37) % 100) as f32 / 100.0;
        let g = ((class_idx * 59) % 100) as f32 / 100.0;
        let b = ((class_idx * 73) % 100) as f32 / 100.0;

        // Fill RGB channels
        for _ in 0..1024 {
            pixels.push(r);
        }
        for _ in 0..1024 {
            pixels.push(g);
        }
        for _ in 0..1024 {
            pixels.push(b);
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        {
            fine_labels.push(class_idx as i32);
            coarse_labels.push(coarse_idx as i32);
        }
    }

    (pixels, fine_labels, coarse_labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn test_cifar100_load() {
        let dataset = cifar100().unwrap();
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_classes(), 100);
    }

    #[test]
    fn test_cifar100_split() {
        let dataset = cifar100().unwrap();
        let split = dataset.split().unwrap();
        assert_eq!(split.train.len(), 80);
        assert_eq!(split.test.len(), 20);
    }

    #[test]
    fn test_cifar100_fine_class_names() {
        assert_eq!(Cifar100Dataset::fine_class_name(0), Some("apple"));
        assert_eq!(Cifar100Dataset::fine_class_name(99), Some("worm"));
        assert_eq!(Cifar100Dataset::fine_class_name(100), None);
        assert_eq!(Cifar100Dataset::fine_class_name(-1), None);
    }

    #[test]
    fn test_cifar100_coarse_class_names() {
        assert_eq!(
            Cifar100Dataset::coarse_class_name(0),
            Some("aquatic_mammals")
        );
        assert_eq!(Cifar100Dataset::coarse_class_name(19), Some("vehicles_2"));
        assert_eq!(Cifar100Dataset::coarse_class_name(20), None);
    }

    #[test]
    fn test_cifar100_has_both_labels() {
        let dataset = cifar100().unwrap();
        let schema = dataset.data().schema();
        assert!(schema.field_with_name("fine_label").is_ok());
        assert!(schema.field_with_name("coarse_label").is_ok());
    }

    #[test]
    fn test_cifar100_coarse_class_name_negative() {
        assert_eq!(Cifar100Dataset::coarse_class_name(-1), None);
        assert_eq!(Cifar100Dataset::coarse_class_name(-100), None);
    }

    #[test]
    fn test_cifar100_num_features() {
        let dataset = cifar100().unwrap();
        assert_eq!(dataset.num_features(), 3072);
    }

    #[test]
    fn test_cifar100_feature_names() {
        let dataset = cifar100().unwrap();
        assert!(dataset.feature_names().is_empty());
    }

    #[test]
    fn test_cifar100_target_name() {
        let dataset = cifar100().unwrap();
        assert_eq!(dataset.target_name(), "fine_label");
    }

    #[test]
    fn test_cifar100_description() {
        let dataset = cifar100().unwrap();
        let desc = dataset.description();
        assert!(desc.contains("CIFAR-100"));
        assert!(desc.contains("100 fine classes"));
    }

    #[test]
    fn test_cifar100_data_access() {
        let dataset = cifar100().unwrap();
        let data = dataset.data();
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_cifar100_schema_columns() {
        let dataset = cifar100().unwrap();
        let batch = dataset.data().get_batch(0).unwrap();
        assert_eq!(batch.num_columns(), 3074); // 3072 pixels + 2 labels
    }

    #[test]
    fn test_cifar100_fine_labels_in_range() {
        let dataset = cifar100().unwrap();
        let batch = dataset.data().get_batch(0).unwrap();
        let label_col = batch
            .column(3072)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..label_col.len() {
            let label = label_col.value(i);
            assert!(
                (0..100).contains(&label),
                "Fine label {} out of range",
                label
            );
        }
    }

    #[test]
    fn test_cifar100_coarse_labels_in_range() {
        let dataset = cifar100().unwrap();
        let batch = dataset.data().get_batch(0).unwrap();
        let label_col = batch
            .column(3073)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..label_col.len() {
            let label = label_col.value(i);
            assert!(
                (0..20).contains(&label),
                "Coarse label {} out of range",
                label
            );
        }
    }

    #[test]
    fn test_cifar100_clone() {
        let dataset = cifar100().unwrap();
        let cloned = dataset.clone();
        assert_eq!(cloned.len(), dataset.len());
    }

    #[test]
    fn test_cifar100_debug() {
        let dataset = cifar100().unwrap();
        let debug = format!("{:?}", dataset);
        assert!(debug.contains("Cifar100Dataset"));
    }

    #[test]
    fn test_cifar100_fine_classes_constant() {
        assert_eq!(CIFAR100_FINE_CLASSES.len(), 100);
        assert_eq!(CIFAR100_FINE_CLASSES[0], "apple");
        assert_eq!(CIFAR100_FINE_CLASSES[99], "worm");
    }

    #[test]
    fn test_cifar100_coarse_classes_constant() {
        assert_eq!(CIFAR100_COARSE_CLASSES.len(), 20);
        assert_eq!(CIFAR100_COARSE_CLASSES[0], "aquatic_mammals");
        assert_eq!(CIFAR100_COARSE_CLASSES[19], "vehicles_2");
    }

    #[test]
    fn test_fine_to_coarse_mapping_valid() {
        for &coarse_idx in &FINE_TO_COARSE {
            assert!(coarse_idx < 20, "Coarse index {} out of range", coarse_idx);
        }
    }

    #[test]
    fn test_embedded_cifar100_sample() {
        let (pixels, fine_labels, coarse_labels) = embedded_cifar100_sample();
        assert_eq!(pixels.len(), 100 * 3072);
        assert_eq!(fine_labels.len(), 100);
        assert_eq!(coarse_labels.len(), 100);
    }

    #[test]
    fn test_embedded_cifar100_sample_labels_valid() {
        let (_, fine_labels, coarse_labels) = embedded_cifar100_sample();
        for (i, &fine) in fine_labels.iter().enumerate() {
            assert!(
                (0..100).contains(&fine),
                "Fine label {} at {} out of range",
                fine,
                i
            );
        }
        for (i, &coarse) in coarse_labels.iter().enumerate() {
            assert!(
                (0..20).contains(&coarse),
                "Coarse label {} at {} out of range",
                coarse,
                i
            );
        }
    }
}
