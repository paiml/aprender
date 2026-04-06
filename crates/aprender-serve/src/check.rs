
/// Server availability check - validates server URL and health endpoint response
///
/// Per spec Section 4.2: Verify server is reachable before benchmark
#[derive(Debug)]
pub struct ServerAvailabilityCheck {
    /// Server URL to check
    url: String,
    /// Health endpoint path (e.g., "/health")
    health_path: String,
    /// Cached health check result (status code, None if not yet checked)
    health_status: Option<u16>,
}

impl ServerAvailabilityCheck {
    /// Create a new server availability check
    #[must_use]
    pub fn new(url: String, health_path: String) -> Self {
        Self {
            url,
            health_path,
            health_status: None,
        }
    }

    /// Create with llama.cpp defaults (port 8082, /health)
    #[must_use]
    pub fn llama_cpp(port: u16) -> Self {
        Self::new(format!("http://127.0.0.1:{port}"), "/health".to_string())
    }

    /// Create with Ollama defaults (port 11434, /api/tags)
    #[must_use]
    pub fn ollama(port: u16) -> Self {
        Self::new(format!("http://127.0.0.1:{port}"), "/api/tags".to_string())
    }

    /// Set the health check result (called after HTTP request)
    pub fn set_health_status(&mut self, status: u16) {
        self.health_status = Some(status);
    }

    /// Get the full health URL
    #[must_use]
    pub fn health_url(&self) -> String {
        format!("{}{}", self.url, self.health_path)
    }

    /// Check if URL is well-formed
    fn validate_url(&self) -> PreflightResult<()> {
        if self.url.is_empty() {
            return Err(PreflightError::ConfigError {
                reason: "Server URL cannot be empty".to_string(),
            });
        }
        if !self.url.starts_with("http://") && !self.url.starts_with("https://") {
            return Err(PreflightError::ConfigError {
                reason: format!(
                    "Server URL must start with http:// or https://, got: {}",
                    self.url
                ),
            });
        }
        Ok(())
    }
}

impl PreflightCheck for ServerAvailabilityCheck {
    fn name(&self) -> &'static str {
        "server_availability_check"
    }

    fn description(&self) -> &'static str {
        "Validates server is reachable at the configured URL"
    }

    fn validate(&self) -> PreflightResult<()> {
        // First validate URL format
        self.validate_url()?;

        // Check if health status has been set
        match self.health_status {
            Some(status) if status >= 200 && status < 300 => Ok(()),
            Some(status) => Err(PreflightError::HealthCheckFailed {
                url: self.health_url(),
                status,
            }),
            None => Err(PreflightError::ConfigError {
                reason: "Health check not performed - call set_health_status() first".to_string(),
            }),
        }
    }
}

/// Model availability check - validates requested model exists
///
/// Per spec Section 4.2: Verify model is loaded before benchmark
#[derive(Debug)]
pub struct ModelAvailabilityCheck {
    /// Model name that is requested
    requested_model: String,
    /// List of available models (populated after query)
    available_models: Vec<String>,
}

impl ModelAvailabilityCheck {
    /// Create a new model availability check
    #[must_use]
    pub fn new(requested_model: String) -> Self {
        Self {
            requested_model,
            available_models: Vec::new(),
        }
    }

    /// Set the list of available models (called after querying server)
    pub fn set_available_models(&mut self, models: Vec<String>) {
        self.available_models = models;
    }

    /// Get the requested model name
    #[must_use]
    pub fn requested_model(&self) -> &str {
        &self.requested_model
    }
}

impl PreflightCheck for ModelAvailabilityCheck {
    fn name(&self) -> &'static str {
        "model_availability_check"
    }

    fn description(&self) -> &'static str {
        "Validates requested model is available on the server"
    }

    fn validate(&self) -> PreflightResult<()> {
        if self.requested_model.is_empty() {
            return Err(PreflightError::ConfigError {
                reason: "Model name cannot be empty".to_string(),
            });
        }

        if self.available_models.is_empty() {
            return Err(PreflightError::ConfigError {
                reason: "Available models list not set - call set_available_models() first"
                    .to_string(),
            });
        }

        // Check for exact match or partial match (model:tag format)
        let found = self.available_models.iter().any(|m| {
            m == &self.requested_model
                || m.starts_with(&format!("{}:", self.requested_model))
                || self.requested_model.starts_with(&format!("{m}:"))
        });

        if found {
            Ok(())
        } else {
            Err(PreflightError::ModelNotFound {
                requested: self.requested_model.clone(),
                available: self.available_models.clone(),
            })
        }
    }
}

/// Response schema check - validates JSON response has required fields
///
/// Per spec Section 4.3: Verify response format matches expected schema
#[derive(Debug)]
pub struct ResponseSchemaCheck {
    /// Required field names that must be present
    required_fields: Vec<String>,
    /// Optional field type constraints (field_name -> expected_type)
    field_types: std::collections::HashMap<String, String>,
}

impl ResponseSchemaCheck {
    /// Create a new response schema check with required fields
    #[must_use]
    pub fn new(required_fields: Vec<String>) -> Self {
        Self {
            required_fields,
            field_types: std::collections::HashMap::new(),
        }
    }

    /// Create schema check for llama.cpp /completion response
    #[must_use]
    pub fn llama_cpp_completion() -> Self {
        let mut check = Self::new(vec![
            "content".to_string(),
            "tokens_predicted".to_string(),
            "timings".to_string(),
        ]);
        check
            .field_types
            .insert("tokens_predicted".to_string(), "number".to_string());
        check
            .field_types
            .insert("content".to_string(), "string".to_string());
        check
    }

    /// Create schema check for Ollama /api/generate response
    #[must_use]
    pub fn ollama_generate() -> Self {
        let mut check = Self::new(vec!["response".to_string(), "done".to_string()]);
        check
            .field_types
            .insert("response".to_string(), "string".to_string());
        check
            .field_types
            .insert("done".to_string(), "boolean".to_string());
        check
    }

    /// Add a type constraint for a field
    #[must_use]
    pub fn with_type_constraint(mut self, field: String, expected_type: String) -> Self {
        self.field_types.insert(field, expected_type);
        self
    }

    /// Validate a JSON value against this schema
    ///
    /// # Errors
    /// Returns `PreflightError` if:
    /// - The JSON is not an object
    /// - A required field is missing
    /// - A field has an unexpected type
    pub fn validate_json(&self, json: &serde_json::Value) -> PreflightResult<()> {
        let obj = json
            .as_object()
            .ok_or_else(|| PreflightError::ResponseParseError {
                reason: "Expected JSON object at root".to_string(),
            })?;

        // Check required fields exist
        for field in &self.required_fields {
            if !obj.contains_key(field) {
                return Err(PreflightError::SchemaMismatch {
                    missing_field: field.clone(),
                });
            }
        }

        // Check field types
        for (field, expected_type) in &self.field_types {
            if let Some(value) = obj.get(field) {
                let actual_type = match value {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "boolean",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Object(_) => "object",
                };

                if actual_type != expected_type {
                    return Err(PreflightError::FieldTypeMismatch {
                        field: field.clone(),
                        expected: expected_type.clone(),
                        actual: actual_type.to_string(),
                    });
                }
            }
        }

        Ok(())
    }
}

impl PreflightCheck for ResponseSchemaCheck {
    fn name(&self) -> &'static str {
        "response_schema_check"
    }

    fn description(&self) -> &'static str {
        "Validates response JSON matches expected schema"
    }

    fn validate(&self) -> PreflightResult<()> {
        // Standalone validation - just checks configuration is valid
        if self.required_fields.is_empty() {
            return Err(PreflightError::ConfigError {
                reason: "At least one required field must be specified".to_string(),
            });
        }
        Ok(())
    }
}

/// Preflight validation runner - executes all checks in sequence
///
/// Per Jidoka principle: Stop immediately on first failure
#[derive(Debug, Default)]
pub struct PreflightRunner {
    /// Checks to execute in order
    checks: Vec<Box<dyn PreflightCheck>>,
    /// Names of passed checks (populated during run)
    passed: Vec<String>,
}

impl PreflightRunner {
    /// Create a new preflight runner
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a check to the runner
    pub fn add_check(&mut self, check: Box<dyn PreflightCheck>) {
        self.checks.push(check);
    }

    /// Run all checks, stopping on first failure (Jidoka)
    ///
    /// Returns list of passed check names on success
    ///
    /// # Errors
    /// Returns the `PreflightError` from the first check that fails.
    /// Per Jidoka principle, execution stops immediately on first failure.
    pub fn run(&mut self) -> PreflightResult<Vec<String>> {
        self.passed.clear();

        for check in &self.checks {
            check.validate()?;
            self.passed.push(check.name().to_string());
        }

        Ok(self.passed.clone())
    }

    /// Get passed checks (after run)
    #[must_use]
    pub fn passed_checks(&self) -> &[String] {
        &self.passed
    }
}
