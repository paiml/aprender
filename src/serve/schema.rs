//! Content Schema - Schema definitions for content validation and UI generation
//!
//! Provides a flexible schema system for defining content structure,
//! validation rules, and UI hints.

use serde::{Deserialize, Serialize};

use crate::serve::content::ContentTypeId;

/// Schema definition for content validation and UI generation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContentSchema {
    /// Schema version for compatibility checking
    pub version: String,

    /// Content type identifier
    pub content_type: ContentTypeId,

    /// Field definitions with types and constraints
    #[serde(default)]
    pub fields: Vec<FieldDefinition>,

    /// Required fields for validation
    #[serde(default)]
    pub required: Vec<String>,

    /// Custom validators (regex patterns, range checks, etc.)
    #[serde(default)]
    pub validators: Vec<ValidatorDefinition>,
}

impl ContentSchema {
    /// Create a new schema
    pub fn new(content_type: ContentTypeId, version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            content_type,
            fields: Vec::new(),
            required: Vec::new(),
            validators: Vec::new(),
        }
    }

    /// Add a field definition
    pub fn with_field(mut self, field: FieldDefinition) -> Self {
        self.fields.push(field);
        self
    }

    /// Add multiple field definitions
    pub fn with_fields(mut self, fields: Vec<FieldDefinition>) -> Self {
        self.fields.extend(fields);
        self
    }

    /// Mark a field as required
    pub fn with_required(mut self, field_name: impl Into<String>) -> Self {
        self.required.push(field_name.into());
        self
    }

    /// Add a validator
    pub fn with_validator(mut self, validator: ValidatorDefinition) -> Self {
        self.validators.push(validator);
        self
    }

    /// Get a field by name
    pub fn get_field(&self, name: &str) -> Option<&FieldDefinition> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Check if a field is required
    pub fn is_required(&self, name: &str) -> bool {
        self.required.contains(&name.to_string())
    }
}

/// Field definition with type and constraints
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldDefinition {
    /// Field name
    pub name: String,

    /// Field type
    pub field_type: FieldType,

    /// Human-readable description
    #[serde(default)]
    pub description: Option<String>,

    /// Default value (as JSON)
    #[serde(default)]
    pub default: Option<serde_json::Value>,

    /// Validation constraints
    #[serde(default)]
    pub constraints: Vec<Constraint>,

    /// Whether this field is nullable
    #[serde(default)]
    pub nullable: bool,
}

impl FieldDefinition {
    /// Create a new field definition
    pub fn new(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            description: None,
            default: None,
            constraints: Vec::new(),
            nullable: false,
        }
    }

    /// Add a description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set default value
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.default = Some(default);
        self
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set nullable
    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }
}

/// Field type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum FieldType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Floating point type
    Float,
    /// Boolean type
    Boolean,
    /// Date/time type
    DateTime,
    /// Binary data type
    Binary,
    /// Array type with item type
    Array {
        /// Type of array items
        item_type: Box<Self>,
    },
    /// Object type with nested schema
    Object {
        /// Nested schema
        schema: Box<ContentSchema>,
    },
    /// Reference to another content type
    Reference {
        /// Referenced content type
        content_type: ContentTypeId,
    },
}

impl FieldType {
    /// Create a string type
    pub fn string() -> Self {
        Self::String
    }

    /// Create an integer type
    pub fn integer() -> Self {
        Self::Integer
    }

    /// Create a float type
    pub fn float() -> Self {
        Self::Float
    }

    /// Create a boolean type
    pub fn boolean() -> Self {
        Self::Boolean
    }

    /// Create a datetime type
    pub fn datetime() -> Self {
        Self::DateTime
    }

    /// Create a binary type
    pub fn binary() -> Self {
        Self::Binary
    }

    /// Create an array type
    pub fn array(item_type: Self) -> Self {
        Self::Array {
            item_type: Box::new(item_type),
        }
    }

    /// Create an object type
    pub fn object(schema: ContentSchema) -> Self {
        Self::Object {
            schema: Box::new(schema),
        }
    }

    /// Create a reference type
    pub fn reference(content_type: ContentTypeId) -> Self {
        Self::Reference { content_type }
    }
}

/// Constraint definition for field validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Constraint {
    /// Minimum value (for numbers)
    Min {
        /// Minimum value
        value: f64,
    },
    /// Maximum value (for numbers)
    Max {
        /// Maximum value
        value: f64,
    },
    /// Minimum length (for strings/arrays)
    MinLength {
        /// Minimum length
        value: usize,
    },
    /// Maximum length (for strings/arrays)
    MaxLength {
        /// Maximum length
        value: usize,
    },
    /// Regex pattern (for strings)
    Pattern {
        /// Regex pattern
        pattern: String,
    },
    /// Enum of allowed values
    Enum {
        /// Allowed values
        values: Vec<serde_json::Value>,
    },
    /// Custom constraint
    Custom {
        /// Constraint name
        name: String,
        /// Constraint parameters
        params: serde_json::Value,
    },
}

impl Constraint {
    /// Create a minimum value constraint
    pub fn min(value: f64) -> Self {
        Self::Min { value }
    }

    /// Create a maximum value constraint
    pub fn max(value: f64) -> Self {
        Self::Max { value }
    }

    /// Create a minimum length constraint
    pub fn min_length(value: usize) -> Self {
        Self::MinLength { value }
    }

    /// Create a maximum length constraint
    pub fn max_length(value: usize) -> Self {
        Self::MaxLength { value }
    }

    /// Create a pattern constraint
    pub fn pattern(pattern: impl Into<String>) -> Self {
        Self::Pattern {
            pattern: pattern.into(),
        }
    }

    /// Create an enum constraint
    pub fn enum_values(values: Vec<serde_json::Value>) -> Self {
        Self::Enum { values }
    }

    /// Create a custom constraint
    pub fn custom(name: impl Into<String>, params: serde_json::Value) -> Self {
        Self::Custom {
            name: name.into(),
            params,
        }
    }
}

/// Validator definition for custom validation logic
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidatorDefinition {
    /// Validator type (e.g., "custom", "regex", "range")
    pub validator_type: String,

    /// Validator name for identification
    pub name: String,

    /// Error message on validation failure
    pub message: String,

    /// Validation expression or configuration
    pub check: String,
}

impl ValidatorDefinition {
    /// Create a new validator definition
    pub fn new(
        validator_type: impl Into<String>,
        name: impl Into<String>,
        message: impl Into<String>,
        check: impl Into<String>,
    ) -> Self {
        Self {
            validator_type: validator_type.into(),
            name: name.into(),
            message: message.into(),
            check: check.into(),
        }
    }

    /// Create a custom validator
    pub fn custom(
        name: impl Into<String>,
        message: impl Into<String>,
        check: impl Into<String>,
    ) -> Self {
        Self::new("custom", name, message, check)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let schema = ContentSchema::new(ContentTypeId::dataset(), "1.0")
            .with_field(FieldDefinition::new("name", FieldType::String))
            .with_field(FieldDefinition::new("age", FieldType::Integer))
            .with_required("name");

        assert_eq!(schema.version, "1.0");
        assert_eq!(schema.fields.len(), 2);
        assert!(schema.is_required("name"));
        assert!(!schema.is_required("age"));
    }

    #[test]
    fn test_field_definition() {
        let field = FieldDefinition::new("email", FieldType::String)
            .with_description("User email address")
            .with_constraint(Constraint::pattern(r"^[\w.-]+@[\w.-]+\.\w+$"))
            .with_constraint(Constraint::max_length(255));

        assert_eq!(field.name, "email");
        assert_eq!(field.constraints.len(), 2);
    }

    #[test]
    fn test_field_types() {
        assert_eq!(FieldType::string(), FieldType::String);
        assert_eq!(FieldType::integer(), FieldType::Integer);

        let array_type = FieldType::array(FieldType::String);
        match array_type {
            FieldType::Array { item_type } => {
                assert_eq!(*item_type, FieldType::String);
            }
            _ => panic!("Expected Array type"),
        }
    }

    #[test]
    fn test_constraints() {
        let min = Constraint::min(0.0);
        let max = Constraint::max(100.0);
        let pattern = Constraint::pattern(r"^\d+$");
        let enum_vals =
            Constraint::enum_values(vec![serde_json::json!("a"), serde_json::json!("b")]);

        assert!(matches!(min, Constraint::Min { value } if value == 0.0));
        assert!(matches!(max, Constraint::Max { value } if value == 100.0));
        assert!(matches!(pattern, Constraint::Pattern { .. }));
        assert!(matches!(enum_vals, Constraint::Enum { .. }));
    }

    #[test]
    fn test_validator_definition() {
        let validator = ValidatorDefinition::custom(
            "valid_email",
            "Email must be valid",
            "email matches /^[\\w.-]+@[\\w.-]+\\.\\w+$/",
        );

        assert_eq!(validator.validator_type, "custom");
        assert_eq!(validator.name, "valid_email");
    }

    #[test]
    fn test_nested_schema() {
        let address_schema = ContentSchema::new(ContentTypeId::new("address"), "1.0")
            .with_field(FieldDefinition::new("street", FieldType::String))
            .with_field(FieldDefinition::new("city", FieldType::String));

        let user_schema = ContentSchema::new(ContentTypeId::new("user"), "1.0")
            .with_field(FieldDefinition::new("name", FieldType::String))
            .with_field(FieldDefinition::new(
                "address",
                FieldType::object(address_schema),
            ));

        assert_eq!(user_schema.fields.len(), 2);
        match &user_schema.fields[1].field_type {
            FieldType::Object { schema } => {
                assert_eq!(schema.fields.len(), 2);
            }
            _ => panic!("Expected Object type"),
        }
    }

    #[test]
    fn test_get_field() {
        let schema = ContentSchema::new(ContentTypeId::dataset(), "1.0")
            .with_field(FieldDefinition::new("id", FieldType::Integer))
            .with_field(FieldDefinition::new("name", FieldType::String));

        assert!(schema.get_field("id").is_some());
        assert!(schema.get_field("name").is_some());
        assert!(schema.get_field("nonexistent").is_none());
    }

    #[test]
    fn test_schema_with_fields() {
        let fields = vec![
            FieldDefinition::new("a", FieldType::String),
            FieldDefinition::new("b", FieldType::Integer),
        ];
        let schema = ContentSchema::new(ContentTypeId::dataset(), "1.0").with_fields(fields);
        assert_eq!(schema.fields.len(), 2);
    }

    #[test]
    fn test_schema_with_validator() {
        let validator = ValidatorDefinition::new("custom", "test", "must be valid", "true");
        let schema = ContentSchema::new(ContentTypeId::dataset(), "1.0").with_validator(validator);
        assert_eq!(schema.validators.len(), 1);
    }

    #[test]
    fn test_field_with_default() {
        let field =
            FieldDefinition::new("count", FieldType::Integer).with_default(serde_json::json!(0));
        assert_eq!(field.default, Some(serde_json::json!(0)));
    }

    #[test]
    fn test_field_nullable() {
        let field = FieldDefinition::new("optional", FieldType::String).nullable();
        assert!(field.nullable);
    }

    #[test]
    fn test_field_type_boolean() {
        assert_eq!(FieldType::boolean(), FieldType::Boolean);
    }

    #[test]
    fn test_field_type_float() {
        assert_eq!(FieldType::float(), FieldType::Float);
    }

    #[test]
    fn test_field_type_datetime() {
        assert_eq!(FieldType::datetime(), FieldType::DateTime);
    }

    #[test]
    fn test_field_type_binary() {
        assert_eq!(FieldType::binary(), FieldType::Binary);
    }

    #[test]
    fn test_constraint_min_length() {
        let c = Constraint::min_length(5);
        assert!(matches!(c, Constraint::MinLength { value } if value == 5));
    }

    #[test]
    fn test_validator_new() {
        let v = ValidatorDefinition::new("regex", "email_check", "invalid email", r"^.+@.+$");
        assert_eq!(v.validator_type, "regex");
        assert_eq!(v.name, "email_check");
        assert_eq!(v.message, "invalid email");
    }

    #[test]
    fn test_field_type_reference() {
        let ref_type = FieldType::reference(ContentTypeId::dataset());
        match ref_type {
            FieldType::Reference { content_type } => {
                assert_eq!(content_type.as_str(), "alimentar.dataset");
            }
            _ => panic!("Expected Reference type"),
        }
    }

    #[test]
    fn test_constraint_custom() {
        let c = Constraint::custom("unique", serde_json::json!({"scope": "global"}));
        match c {
            Constraint::Custom { name, params } => {
                assert_eq!(name, "unique");
                assert!(params.get("scope").is_some());
            }
            _ => panic!("Expected Custom constraint"),
        }
    }

    #[test]
    fn test_schema_serialization() {
        let schema = ContentSchema::new(ContentTypeId::dataset(), "1.0")
            .with_field(FieldDefinition::new("id", FieldType::Integer));

        let json = serde_json::to_string(&schema);
        assert!(json.is_ok());

        let parsed: Result<ContentSchema, _> =
            serde_json::from_str(&json.ok().unwrap_or_else(|| panic!("Should serialize")));
        assert!(parsed.is_ok());
    }
}
