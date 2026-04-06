//! RO-Crate entity types and structures.

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

/// RO-Crate entity types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    /// Root dataset
    Dataset,
    /// File entity
    File,
    /// Person
    Person,
    /// Organization
    Organization,
    /// Software application
    SoftwareApplication,
    /// Creative work (paper, etc.)
    CreativeWork,
    /// Action (workflow execution, etc.)
    CreateAction,
    /// Custom type
    Custom(String),
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset => write!(f, "Dataset"),
            Self::File => write!(f, "File"),
            Self::Person => write!(f, "Person"),
            Self::Organization => write!(f, "Organization"),
            Self::SoftwareApplication => write!(f, "SoftwareApplication"),
            Self::CreativeWork => write!(f, "CreativeWork"),
            Self::CreateAction => write!(f, "CreateAction"),
            Self::Custom(t) => write!(f, "{t}"),
        }
    }
}

/// An entity in the RO-Crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoCrateEntity {
    /// Entity ID (path or URL)
    #[serde(rename = "@id")]
    pub id: String,
    /// Entity type
    #[serde(rename = "@type")]
    pub type_field: String,
    /// Additional properties
    #[serde(flatten)]
    pub properties: HashMap<String, serde_json::Value>,
}

impl RoCrateEntity {
    /// Create a new entity
    pub fn new(id: impl Into<String>, entity_type: EntityType) -> Self {
        Self { id: id.into(), type_field: entity_type.to_string(), properties: HashMap::new() }
    }

    /// Add a property
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Add a name property
    pub fn with_name(self, name: impl Into<String>) -> Self {
        self.with_property("name", name.into())
    }

    /// Add a description
    pub fn with_description(self, description: impl Into<String>) -> Self {
        self.with_property("description", description.into())
    }

    /// Add a reference to another entity
    pub fn with_reference(self, key: impl Into<String>, ref_id: impl Into<String>) -> Self {
        self.with_property(key, json!({ "@id": ref_id.into() }))
    }

    /// Add multiple references
    pub fn with_references(
        self,
        key: impl Into<String>,
        ref_ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let refs: Vec<serde_json::Value> =
            ref_ids.into_iter().map(|id| json!({ "@id": id.into() })).collect();
        self.with_property(key, refs)
    }

    /// Create the root dataset entity
    pub fn root_dataset() -> Self {
        Self::new("./", EntityType::Dataset)
    }

    /// Create a file entity
    pub fn file(path: impl Into<String>) -> Self {
        Self::new(path, EntityType::File)
    }

    /// Create a person entity
    pub fn person(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::new(id, EntityType::Person).with_name(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_display_dataset() {
        assert_eq!(EntityType::Dataset.to_string(), "Dataset");
    }

    #[test]
    fn test_entity_type_display_file() {
        assert_eq!(EntityType::File.to_string(), "File");
    }

    #[test]
    fn test_entity_type_display_person() {
        assert_eq!(EntityType::Person.to_string(), "Person");
    }

    #[test]
    fn test_entity_type_display_organization() {
        assert_eq!(EntityType::Organization.to_string(), "Organization");
    }

    #[test]
    fn test_entity_type_display_software_application() {
        assert_eq!(EntityType::SoftwareApplication.to_string(), "SoftwareApplication");
    }

    #[test]
    fn test_entity_type_display_creative_work() {
        assert_eq!(EntityType::CreativeWork.to_string(), "CreativeWork");
    }

    #[test]
    fn test_entity_type_display_create_action() {
        assert_eq!(EntityType::CreateAction.to_string(), "CreateAction");
    }

    #[test]
    fn test_entity_type_display_custom() {
        let custom = EntityType::Custom("MyType".to_string());
        assert_eq!(custom.to_string(), "MyType");
    }

    #[test]
    fn test_entity_type_clone() {
        let et = EntityType::Dataset;
        let cloned = et.clone();
        assert_eq!(et, cloned);
    }

    #[test]
    fn test_entity_type_eq() {
        assert_eq!(EntityType::File, EntityType::File);
        assert_ne!(EntityType::File, EntityType::Person);
    }

    #[test]
    fn test_ro_crate_entity_new() {
        let entity = RoCrateEntity::new("test-id", EntityType::Dataset);
        assert_eq!(entity.id, "test-id");
        assert_eq!(entity.type_field, "Dataset");
        assert!(entity.properties.is_empty());
    }

    #[test]
    fn test_ro_crate_entity_with_property() {
        let entity = RoCrateEntity::new("test", EntityType::File).with_property("size", 1024);
        assert_eq!(entity.properties.get("size"), Some(&json!(1024)));
    }

    #[test]
    fn test_ro_crate_entity_with_name() {
        let entity = RoCrateEntity::new("test", EntityType::Dataset).with_name("My Dataset");
        assert_eq!(entity.properties.get("name"), Some(&json!("My Dataset")));
    }

    #[test]
    fn test_ro_crate_entity_with_description() {
        let entity =
            RoCrateEntity::new("test", EntityType::Dataset).with_description("A test dataset");
        assert_eq!(entity.properties.get("description"), Some(&json!("A test dataset")));
    }

    #[test]
    fn test_ro_crate_entity_with_reference() {
        let entity =
            RoCrateEntity::new("test", EntityType::File).with_reference("author", "#person1");
        let expected = json!({ "@id": "#person1" });
        assert_eq!(entity.properties.get("author"), Some(&expected));
    }

    #[test]
    fn test_ro_crate_entity_with_references() {
        let entity = RoCrateEntity::new("test", EntityType::Dataset)
            .with_references("hasPart", vec!["file1.txt", "file2.txt"]);
        let parts = entity.properties.get("hasPart").expect("key should exist");
        assert!(parts.is_array());
        let arr = parts.as_array().expect("operation should succeed");
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_ro_crate_entity_root_dataset() {
        let entity = RoCrateEntity::root_dataset();
        assert_eq!(entity.id, "./");
        assert_eq!(entity.type_field, "Dataset");
    }

    #[test]
    fn test_ro_crate_entity_file() {
        let entity = RoCrateEntity::file("data/model.safetensors");
        assert_eq!(entity.id, "data/model.safetensors");
        assert_eq!(entity.type_field, "File");
    }

    #[test]
    fn test_ro_crate_entity_person() {
        let entity = RoCrateEntity::person("#alice", "Alice Smith");
        assert_eq!(entity.id, "#alice");
        assert_eq!(entity.type_field, "Person");
        assert_eq!(entity.properties.get("name"), Some(&json!("Alice Smith")));
    }

    #[test]
    fn test_ro_crate_entity_clone() {
        let entity = RoCrateEntity::new("test", EntityType::File).with_name("test.txt");
        let cloned = entity.clone();
        assert_eq!(entity.id, cloned.id);
        assert_eq!(entity.type_field, cloned.type_field);
    }

    #[test]
    fn test_ro_crate_entity_serde() {
        let entity = RoCrateEntity::new("test", EntityType::Dataset)
            .with_name("Test Dataset")
            .with_description("A test");

        let json = serde_json::to_string(&entity).expect("JSON serialization should succeed");
        let deserialized: RoCrateEntity =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(entity.id, deserialized.id);
        assert_eq!(entity.type_field, deserialized.type_field);
    }

    #[test]
    fn test_entity_type_serde() {
        let et = EntityType::SoftwareApplication;
        let json = serde_json::to_string(&et).expect("JSON serialization should succeed");
        let deserialized: EntityType =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(et, deserialized);
    }

    #[test]
    fn test_entity_type_debug() {
        assert_eq!(format!("{:?}", EntityType::Dataset), "Dataset");
        assert_eq!(format!("{:?}", EntityType::Custom("Foo".to_string())), "Custom(\"Foo\")");
    }

    #[test]
    fn test_ro_crate_entity_chained_methods() {
        let entity = RoCrateEntity::root_dataset()
            .with_name("My Research Crate")
            .with_description("Contains ML artifacts")
            .with_reference("author", "#researcher")
            .with_property("datePublished", "2024-01-15");

        assert_eq!(entity.properties.len(), 4);
        assert_eq!(entity.properties.get("name"), Some(&json!("My Research Crate")));
    }
}
