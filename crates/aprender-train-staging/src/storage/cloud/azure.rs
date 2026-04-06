//! Azure Blob Storage configuration

use serde::{Deserialize, Serialize};

/// Azure Blob Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureConfig {
    /// Storage account name
    pub account: String,
    /// Container name
    pub container: String,
    /// Blob prefix
    pub prefix: String,
    /// Connection string (if not using managed identity)
    pub connection_string: Option<String>,
}

impl AzureConfig {
    /// Create a new Azure configuration
    pub fn new(account: &str, container: &str) -> Self {
        Self {
            account: account.to_string(),
            container: container.to_string(),
            prefix: String::new(),
            connection_string: None,
        }
    }

    /// Set blob prefix
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    /// Set connection string
    pub fn with_connection_string(mut self, conn_str: &str) -> Self {
        self.connection_string = Some(conn_str.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azure_config_new() {
        let config = AzureConfig::new("myaccount", "mycontainer");
        assert_eq!(config.account, "myaccount");
        assert_eq!(config.container, "mycontainer");
    }

    #[test]
    fn test_azure_config_with_prefix() {
        let config = AzureConfig::new("account", "container").with_prefix("models/");
        assert_eq!(config.prefix, "models/");
    }

    #[test]
    fn test_azure_config_with_connection_string() {
        let config = AzureConfig::new("account", "container")
            .with_connection_string("DefaultEndpointsProtocol=https;...");
        assert!(config.connection_string.is_some());
    }

    #[test]
    fn test_azure_config_serde() {
        let config = AzureConfig::new("account", "container")
            .with_prefix("models/")
            .with_connection_string("conn");

        let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
        let parsed: AzureConfig =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(config.account, parsed.account);
        assert_eq!(config.container, parsed.container);
        assert_eq!(config.prefix, parsed.prefix);
    }
}
