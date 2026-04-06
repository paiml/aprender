//! Column selection and manipulation transforms.

use std::sync::Arc;

use arrow::{
    array::RecordBatch,
    datatypes::{Field, Schema},
};

use super::Transform;
use crate::error::{Error, Result};

/// A transform that selects specific columns from a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Select;
///
/// let select = Select::new(vec!["id", "name"]);
/// ```
#[derive(Debug, Clone)]
pub struct Select {
    columns: Vec<String>,
}

impl Select {
    /// Creates a new Select transform for the given column names.
    pub fn new<S: Into<String>>(columns: impl IntoIterator<Item = S>) -> Self {
        Self {
            columns: columns.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns the columns to be selected.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }
}

impl Transform for Select {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();
        let mut fields = Vec::with_capacity(self.columns.len());
        let mut arrays = Vec::with_capacity(self.columns.len());

        for col_name in &self.columns {
            let (idx, field) = schema
                .column_with_name(col_name)
                .ok_or_else(|| Error::column_not_found(col_name))?;

            fields.push(field.clone());
            arrays.push(Arc::clone(batch.column(idx)));
        }

        let new_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(new_schema, arrays).map_err(Error::Arrow)
    }
}

/// A transform that renames columns in a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Rename;
/// use std::collections::HashMap;
///
/// let mut mapping = HashMap::new();
/// mapping.insert("old_name".to_string(), "new_name".to_string());
/// let rename = Rename::new(mapping);
/// ```
#[derive(Debug, Clone)]
pub struct Rename {
    mapping: std::collections::HashMap<String, String>,
}

impl Rename {
    /// Creates a new Rename transform with the given column mappings.
    pub fn new(mapping: std::collections::HashMap<String, String>) -> Self {
        Self { mapping }
    }

    /// Creates a Rename transform from pairs of (old_name, new_name).
    pub fn from_pairs<S: Into<String>>(pairs: impl IntoIterator<Item = (S, S)>) -> Self {
        let mapping = pairs
            .into_iter()
            .map(|(old, new)| (old.into(), new.into()))
            .collect();
        Self { mapping }
    }
}

impl Transform for Rename {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();
        let new_fields: Vec<Field> = schema
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                match self.mapping.get(name) {
                    Some(new_name) => {
                        Field::new(new_name, field.data_type().clone(), field.is_nullable())
                    }
                    None => field.as_ref().clone(),
                }
            })
            .collect();

        let new_schema = Arc::new(Schema::new(new_fields));
        RecordBatch::try_new(new_schema, batch.columns().to_vec()).map_err(Error::Arrow)
    }
}

/// A transform that drops (removes) specified columns from a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Drop;
///
/// let drop = Drop::new(vec!["temp_column", "debug_info"]);
/// ```
#[derive(Debug, Clone)]
pub struct Drop {
    columns: Vec<String>,
}

impl Drop {
    /// Creates a new Drop transform for the given column names.
    pub fn new<S: Into<String>>(columns: impl IntoIterator<Item = S>) -> Self {
        Self {
            columns: columns.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns the columns to be dropped.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }
}

impl Transform for Drop {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();
        let drop_set: std::collections::HashSet<&str> =
            self.columns.iter().map(String::as_str).collect();

        let mut fields = Vec::new();
        let mut arrays = Vec::new();

        for (idx, field) in schema.fields().iter().enumerate() {
            if !drop_set.contains(field.name().as_str()) {
                fields.push(field.as_ref().clone());
                arrays.push(Arc::clone(batch.column(idx)));
            }
        }

        if fields.is_empty() {
            return Err(Error::transform("Cannot drop all columns from batch"));
        }

        let new_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(new_schema, arrays).map_err(Error::Arrow)
    }
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::DataType,
    };

    use super::*;

    fn create_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let name_array = StringArray::from(vec!["a", "b", "c", "d", "e"]);
        let value_array = Int32Array::from(vec![10, 20, 30, 40, 50]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(name_array),
                Arc::new(value_array),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"))
    }

    #[test]
    fn test_select_transform() {
        let batch = create_test_batch();
        let transform = Select::new(vec!["id", "value"]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(0).name(), "id");
        assert_eq!(result.schema().field(1).name(), "value");
    }

    #[test]
    fn test_select_column_not_found() {
        let batch = create_test_batch();
        let transform = Select::new(vec!["nonexistent"]);

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_columns_getter() {
        let select = Select::new(vec!["a", "b"]);
        assert_eq!(select.columns(), &["a", "b"]);
    }

    #[test]
    fn test_select_preserves_column_order() {
        let batch = create_test_batch();
        // Select in reverse order
        let select = Select::new(vec!["value", "name", "id"]);
        let result = select.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.schema().field(0).name(), "value");
        assert_eq!(result.schema().field(1).name(), "name");
        assert_eq!(result.schema().field(2).name(), "id");
    }

    #[test]
    fn test_rename_transform() {
        let batch = create_test_batch();
        let transform = Rename::from_pairs([("id", "identifier"), ("name", "label")]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.schema().field(0).name(), "identifier");
        assert_eq!(result.schema().field(1).name(), "label");
        assert_eq!(result.schema().field(2).name(), "value"); // Unchanged
    }

    #[test]
    fn test_rename_multiple_columns() {
        let batch = create_test_batch();
        let transform = Rename::from_pairs([("id", "identifier"), ("name", "label")]);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert!(result.schema().field_with_name("identifier").is_ok());
        assert!(result.schema().field_with_name("label").is_ok());
    }

    #[test]
    fn test_rename_nonexistent_column_is_ok() {
        let batch = create_test_batch();
        let transform = Rename::from_pairs([("nonexistent", "new_name")]);
        let result = transform.apply(batch.clone());
        // Renaming a nonexistent column should succeed (no-op)
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_rename_debug() {
        let rename = Rename::from_pairs([("old", "new")]);
        let debug_str = format!("{:?}", rename);
        assert!(debug_str.contains("Rename"));
    }

    #[test]
    fn test_rename_nonexistent_column() {
        let batch = create_test_batch();
        let rename = Rename::from_pairs([("nonexistent", "new_name")]);
        let result = rename.apply(batch);
        // Renaming nonexistent column should succeed (no-op)
        assert!(result.is_ok());
    }

    #[test]
    fn test_drop_transform() {
        let batch = create_test_batch();
        let transform = Drop::new(vec!["name"]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(0).name(), "id");
        assert_eq!(result.schema().field(1).name(), "value");
    }

    #[test]
    fn test_drop_multiple_columns() {
        let batch = create_test_batch();
        let transform = Drop::new(vec!["id", "name"]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.schema().field(0).name(), "value");
    }

    #[test]
    fn test_drop_all_columns_error() {
        let batch = create_test_batch();
        let transform = Drop::new(vec!["id", "name", "value"]);

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_nonexistent_column_is_ok() {
        let batch = create_test_batch();
        let transform = Drop::new(vec!["nonexistent"]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 3); // All columns remain
    }

    #[test]
    fn test_drop_columns_getter() {
        let transform = Drop::new(vec!["a", "b"]);
        assert_eq!(transform.columns(), &["a", "b"]);
    }

    #[test]
    fn test_select_debug() {
        let select = Select::new(vec!["id", "name"]);
        let debug_str = format!("{:?}", select);
        assert!(debug_str.contains("Select"));
    }

    #[test]
    fn test_drop_debug() {
        let drop_t = Drop::new(vec!["col"]);
        let debug_str = format!("{:?}", drop_t);
        assert!(debug_str.contains("Drop"));
    }

    #[test]
    fn test_drop_nonexistent_columns_unchanged() {
        let batch = create_test_batch();
        let original_cols = batch.num_columns();
        let drop = Drop::new(["nonexistent_column", "also_nonexistent"]);
        let result = drop.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        // Dropping nonexistent columns should return unchanged batch
        assert_eq!(result.num_columns(), original_cols);
    }
}
