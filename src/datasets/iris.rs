//! Iris dataset loader
//!
//! The classic Iris flower dataset by Ronald Fisher (1936).
//! Contains 150 samples of 3 iris species with 4 features each.
//!
//! # Example
//!
//! ```
//! use alimentar::datasets::{iris, CanonicalDataset};
//!
//! let dataset = iris().unwrap();
//! assert_eq!(dataset.len(), 150);
//! assert_eq!(dataset.num_classes(), 3);
//! ```

use std::sync::Arc;

use arrow::{
    array::{Array, Float64Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};

use super::CanonicalDataset;
use crate::{ArrowDataset, Dataset, Result};

/// Load the Iris dataset
///
/// Returns a dataset with 150 samples and 5 columns:
/// - sepal_length (f64)
/// - sepal_width (f64)
/// - petal_length (f64)
/// - petal_width (f64)
/// - species (string: "setosa", "versicolor", "virginica")
///
/// # Errors
///
/// Returns an error if the dataset cannot be constructed (should never happen
/// for embedded data).
///
/// # Example
///
/// ```
/// use alimentar::datasets::{iris, CanonicalDataset};
///
/// let dataset = iris().unwrap();
/// println!(
///     "Iris dataset: {} samples, {} features",
///     dataset.len(),
///     dataset.num_features()
/// );
/// ```
pub fn iris() -> Result<IrisDataset> {
    IrisDataset::load()
}

/// The Iris flower dataset
///
/// A classic dataset for classification containing measurements of 150 iris
/// flowers from 3 species (setosa, versicolor, virginica).
#[derive(Debug, Clone)]
pub struct IrisDataset {
    data: ArrowDataset,
}

impl IrisDataset {
    /// Load the embedded Iris dataset
    ///
    /// # Errors
    ///
    /// Returns an error if dataset construction fails.
    pub fn load() -> Result<Self> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("sepal_length", DataType::Float64, false),
            Field::new("sepal_width", DataType::Float64, false),
            Field::new("petal_length", DataType::Float64, false),
            Field::new("petal_width", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));

        // Embedded Iris data (150 samples)
        // Data from UCI ML Repository / scikit-learn
        let (sepal_length, sepal_width, petal_length, petal_width, species) = iris_data();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(sepal_length)),
                Arc::new(Float64Array::from(sepal_width)),
                Arc::new(Float64Array::from(petal_length)),
                Arc::new(Float64Array::from(petal_width)),
                Arc::new(StringArray::from(species)),
            ],
        )
        .map_err(crate::Error::Arrow)?;

        let data = ArrowDataset::from_batch(batch)?;

        Ok(Self { data })
    }

    /// Get the underlying Arrow dataset
    #[must_use]
    pub fn into_inner(self) -> ArrowDataset {
        self.data
    }

    /// Get feature columns as a new dataset (excludes species)
    ///
    /// # Errors
    ///
    /// Returns an error if transform fails.
    pub fn features(&self) -> Result<ArrowDataset> {
        use crate::transform::{Select, Transform};
        let select = Select::new(vec![
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]);
        let batch = select.apply(
            self.data
                .get_batch(0)
                .ok_or_else(|| crate::Error::empty_dataset("Iris dataset is empty"))?
                .clone(),
        )?;
        ArrowDataset::from_batch(batch)
    }

    /// Get species labels as string array
    #[must_use]
    pub fn labels(&self) -> Vec<String> {
        if let Some(batch) = self.data.get_batch(0) {
            if let Some(col) = batch.column_by_name("species") {
                if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                    return (0..arr.len()).map(|i| arr.value(i).to_string()).collect();
                }
            }
        }
        Vec::new()
    }

    /// Get species labels as numeric (0=setosa, 1=versicolor, 2=virginica)
    #[must_use]
    pub fn labels_numeric(&self) -> Vec<i32> {
        self.labels()
            .iter()
            .map(|s| match s.as_str() {
                "setosa" => 0,
                "versicolor" => 1,
                "virginica" => 2,
                _ => -1,
            })
            .collect()
    }
}

impl CanonicalDataset for IrisDataset {
    fn data(&self) -> &ArrowDataset {
        &self.data
    }

    fn num_features(&self) -> usize {
        4
    }

    fn num_classes(&self) -> usize {
        3
    }

    fn feature_names(&self) -> &'static [&'static str] {
        &["sepal_length", "sepal_width", "petal_length", "petal_width"]
    }

    fn target_name(&self) -> &'static str {
        "species"
    }

    fn description(&self) -> &'static str {
        "Iris flower dataset (Fisher, 1936). 150 samples of 3 iris species \
         (setosa, versicolor, virginica) with 4 features: sepal length/width \
         and petal length/width in centimeters."
    }
}

/// Returns the embedded Iris dataset values
#[allow(clippy::type_complexity, clippy::similar_names)]
fn iris_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<&'static str>) {
    // Iris setosa (50 samples)
    let setosa_sl = vec![
        5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1,
        5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5.0,
        5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0,
    ];
    let setosa_sw = vec![
        3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5,
        3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2,
        3.5, 3.6, 3.0, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3,
    ];
    let setosa_pl = vec![
        1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4,
        1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2,
        1.3, 1.4, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4,
    ];
    let setosa_pw = vec![
        0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3,
        0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2,
        0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2,
    ];

    // Iris versicolor (50 samples)
    let versicolor_sl = vec![
        7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8,
        6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0,
        6.7, 6.3, 5.6, 5.5, 5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7,
    ];
    let versicolor_sw = vec![
        3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7,
        2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3.0, 2.8, 3.0, 2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3.0, 3.4,
        3.1, 2.3, 3.0, 2.5, 2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8,
    ];
    let versicolor_pl = vec![
        4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1,
        4.5, 3.9, 4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5,
        4.7, 4.4, 4.1, 4.0, 4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1,
    ];
    let versicolor_pw = vec![
        1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0,
        1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5, 1.6,
        1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3,
    ];

    // Iris virginica (50 samples)
    let virginica_sl = vec![
        6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7,
        7.7, 6.0, 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2, 7.4, 7.9, 6.4, 6.3, 6.1, 7.7,
        6.3, 6.4, 6.0, 6.9, 6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9,
    ];
    let virginica_sw = vec![
        3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3.0, 2.5, 2.8, 3.2, 3.0, 3.8,
        2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3.0, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0,
        3.4, 3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2, 3.3, 3.0, 2.5, 3.0, 3.4, 3.0,
    ];
    let virginica_pl = vec![
        6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7,
        6.9, 5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1,
        5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5.0, 5.2, 5.4, 5.1,
    ];
    let virginica_pw = vec![
        2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2,
        2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3,
        2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3, 1.8,
    ];

    // Combine all data
    let mut sepal_length = setosa_sl;
    sepal_length.extend(versicolor_sl);
    sepal_length.extend(virginica_sl);

    let mut sepal_width = setosa_sw;
    sepal_width.extend(versicolor_sw);
    sepal_width.extend(virginica_sw);

    let mut petal_length = setosa_pl;
    petal_length.extend(versicolor_pl);
    petal_length.extend(virginica_pl);

    let mut petal_width = setosa_pw;
    petal_width.extend(versicolor_pw);
    petal_width.extend(virginica_pw);

    let species: Vec<&'static str> = std::iter::repeat("setosa")
        .take(50)
        .chain(std::iter::repeat("versicolor").take(50))
        .chain(std::iter::repeat("virginica").take(50))
        .collect();

    (
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        species,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn test_iris_load() {
        let dataset = iris().ok();
        assert!(dataset.is_some());
        let dataset = dataset.unwrap_or_else(|| panic!("Failed to load iris"));
        assert_eq!(dataset.len(), 150);
    }

    #[test]
    fn test_iris_features() {
        let dataset = iris().unwrap_or_else(|e| panic!("Failed: {e}"));
        assert_eq!(dataset.num_features(), 4);
        assert_eq!(dataset.num_classes(), 3);
    }

    #[test]
    fn test_iris_labels() {
        let dataset = iris().unwrap_or_else(|e| panic!("Failed: {e}"));
        let labels = dataset.labels();
        assert_eq!(labels.len(), 150);

        // Check distribution: 50 of each species
        let setosa_count = labels.iter().filter(|s| *s == "setosa").count();
        let versicolor_count = labels.iter().filter(|s| *s == "versicolor").count();
        let virginica_count = labels.iter().filter(|s| *s == "virginica").count();

        assert_eq!(setosa_count, 50);
        assert_eq!(versicolor_count, 50);
        assert_eq!(virginica_count, 50);
    }

    #[test]
    fn test_iris_labels_numeric() {
        let dataset = iris().unwrap_or_else(|e| panic!("Failed: {e}"));
        let labels = dataset.labels_numeric();
        assert_eq!(labels.len(), 150);

        // First 50 should be 0 (setosa)
        assert!(labels[0..50].iter().all(|&x| x == 0));
        // Next 50 should be 1 (versicolor)
        assert!(labels[50..100].iter().all(|&x| x == 1));
        // Last 50 should be 2 (virginica)
        assert!(labels[100..150].iter().all(|&x| x == 2));
    }

    #[test]
    fn test_iris_schema() {
        let dataset = iris().unwrap_or_else(|e| panic!("Failed: {e}"));
        let schema = dataset.data().schema();

        assert_eq!(schema.fields().len(), 5);
        assert!(schema.field_with_name("sepal_length").is_ok());
        assert!(schema.field_with_name("sepal_width").is_ok());
        assert!(schema.field_with_name("petal_length").is_ok());
        assert!(schema.field_with_name("petal_width").is_ok());
        assert!(schema.field_with_name("species").is_ok());
    }

    #[test]
    fn test_iris_feature_extraction() {
        let dataset = iris().unwrap_or_else(|e| panic!("Failed: {e}"));
        let features = dataset.features();
        assert!(features.is_ok());

        let features = features.unwrap_or_else(|e| panic!("Failed: {e}"));
        assert_eq!(features.schema().fields().len(), 4);
        assert!(features.schema().field_with_name("species").is_err());
    }

    #[test]
    fn test_iris_description() {
        let dataset = iris().unwrap_or_else(|e| panic!("Failed: {e}"));
        assert!(dataset.description().contains("Fisher"));
        assert!(dataset.description().contains("150"));
    }

    #[test]
    fn test_iris_canonical_trait() {
        let dataset = iris().unwrap_or_else(|e| panic!("Failed: {e}"));

        assert_eq!(dataset.feature_names().len(), 4);
        assert_eq!(dataset.target_name(), "species");
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_iris_into_inner() {
        let dataset = iris().unwrap();
        let inner = dataset.into_inner();
        assert_eq!(inner.len(), 150);
    }

    #[test]
    fn test_iris_clone() {
        let dataset = iris().unwrap();
        let cloned = dataset.clone();
        assert_eq!(cloned.len(), dataset.len());
    }

    #[test]
    fn test_iris_debug() {
        let dataset = iris().unwrap();
        let debug = format!("{:?}", dataset);
        assert!(debug.contains("IrisDataset"));
    }

    #[test]
    fn test_iris_data_access() {
        let dataset = iris().unwrap();
        let data = dataset.data();
        assert_eq!(data.len(), 150);
    }

    #[test]
    fn test_iris_data_function() {
        let (sl, sw, pl, pw, species) = iris_data();
        assert_eq!(sl.len(), 150);
        assert_eq!(sw.len(), 150);
        assert_eq!(pl.len(), 150);
        assert_eq!(pw.len(), 150);
        assert_eq!(species.len(), 150);
    }

    #[test]
    fn test_iris_data_species_distribution() {
        let (_, _, _, _, species) = iris_data();
        let setosa_count = species.iter().filter(|&&s| s == "setosa").count();
        let versicolor_count = species.iter().filter(|&&s| s == "versicolor").count();
        let virginica_count = species.iter().filter(|&&s| s == "virginica").count();
        assert_eq!(setosa_count, 50);
        assert_eq!(versicolor_count, 50);
        assert_eq!(virginica_count, 50);
    }

    #[test]
    fn test_iris_sepal_length_range() {
        let (sepal_length, _, _, _, _) = iris_data();
        for &val in &sepal_length {
            assert!(
                (4.0..=8.0).contains(&val),
                "Sepal length {} out of typical range",
                val
            );
        }
    }

    #[test]
    fn test_iris_sepal_width_range() {
        let (_, sepal_width, _, _, _) = iris_data();
        for &val in &sepal_width {
            assert!(
                (2.0..=5.0).contains(&val),
                "Sepal width {} out of typical range",
                val
            );
        }
    }

    #[test]
    fn test_iris_feature_names_content() {
        let dataset = iris().unwrap();
        let names = dataset.feature_names();
        assert!(names.contains(&"sepal_length"));
        assert!(names.contains(&"sepal_width"));
        assert!(names.contains(&"petal_length"));
        assert!(names.contains(&"petal_width"));
    }
}
