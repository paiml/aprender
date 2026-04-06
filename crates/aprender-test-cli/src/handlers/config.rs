//! Config command handler

use crate::config::CliConfig;
use crate::ConfigArgs;

/// Execute the config command
pub fn execute_config(config: &CliConfig, args: &ConfigArgs) {
    if args.show {
        print_current_config(config);
    }

    if let Some(ref setting) = args.set {
        set_config_value(setting);
    }

    if args.reset {
        print_default_config();
    }
}

/// Print the current configuration
pub fn print_current_config(config: &CliConfig) {
    println!("Current configuration:");
    println!("  Verbosity: {:?}", config.verbosity);
    println!("  Color: {:?}", config.color);
    println!("  Parallel jobs: {}", config.effective_jobs());
    println!("  Fail fast: {}", config.fail_fast);
    println!("  Coverage: {}", config.coverage);
    println!("  Output dir: {}", config.output_dir);
}

/// Set a configuration value (key=value format)
pub fn set_config_value(setting: &str) {
    if let Some((key, value)) = setting.split_once('=') {
        println!("Setting {key} = {value}");
        // Config persistence would require a config file (e.g., .probar.toml)
        // For now, settings are applied via CLI flags only
        println!(
            "Note: Settings are applied via CLI flags. Use environment variables for persistence."
        );
    } else {
        eprintln!("Invalid setting format. Use: key=value");
    }
}

/// Print default configuration values
pub fn print_default_config() {
    println!("Configuration reset to defaults:");
    let default = CliConfig::new();
    println!("  Verbosity: {:?}", default.verbosity);
    println!("  Color: {:?}", default.color);
    println!("  Parallel jobs: {}", default.effective_jobs());
    println!("  Fail fast: {}", default.fail_fast);
    println!("  Coverage: {}", default.coverage);
}

/// Parse a config setting string into key-value pair
#[must_use]
pub fn parse_setting(setting: &str) -> Option<(&str, &str)> {
    setting.split_once('=')
}

/// Validate a config key
#[must_use]
pub fn is_valid_config_key(key: &str) -> bool {
    matches!(
        key,
        "verbosity" | "color" | "parallel_jobs" | "fail_fast" | "coverage" | "output_dir"
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_setting_valid() {
        let result = parse_setting("key=value");
        assert_eq!(result, Some(("key", "value")));
    }

    #[test]
    fn test_parse_setting_with_equals_in_value() {
        let result = parse_setting("key=value=with=equals");
        assert_eq!(result, Some(("key", "value=with=equals")));
    }

    #[test]
    fn test_parse_setting_invalid() {
        let result = parse_setting("no_equals_sign");
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_setting_empty_value() {
        let result = parse_setting("key=");
        assert_eq!(result, Some(("key", "")));
    }

    #[test]
    fn test_is_valid_config_key_verbosity() {
        assert!(is_valid_config_key("verbosity"));
    }

    #[test]
    fn test_is_valid_config_key_color() {
        assert!(is_valid_config_key("color"));
    }

    #[test]
    fn test_is_valid_config_key_parallel_jobs() {
        assert!(is_valid_config_key("parallel_jobs"));
    }

    #[test]
    fn test_is_valid_config_key_fail_fast() {
        assert!(is_valid_config_key("fail_fast"));
    }

    #[test]
    fn test_is_valid_config_key_coverage() {
        assert!(is_valid_config_key("coverage"));
    }

    #[test]
    fn test_is_valid_config_key_output_dir() {
        assert!(is_valid_config_key("output_dir"));
    }

    #[test]
    fn test_is_valid_config_key_invalid() {
        assert!(!is_valid_config_key("invalid_key"));
        assert!(!is_valid_config_key(""));
        assert!(!is_valid_config_key("random"));
    }

    #[test]
    fn test_execute_config_show() {
        let config = CliConfig::default();
        let args = ConfigArgs {
            show: true,
            set: None,
            reset: false,
        };
        // Should not panic
        execute_config(&config, &args);
    }

    #[test]
    fn test_execute_config_set() {
        let config = CliConfig::default();
        let args = ConfigArgs {
            show: false,
            set: Some("verbosity=debug".to_string()),
            reset: false,
        };
        // Should not panic
        execute_config(&config, &args);
    }

    #[test]
    fn test_execute_config_reset() {
        let config = CliConfig::default();
        let args = ConfigArgs {
            show: false,
            set: None,
            reset: true,
        };
        // Should not panic
        execute_config(&config, &args);
    }

    #[test]
    fn test_execute_config_invalid_set() {
        let config = CliConfig::default();
        let args = ConfigArgs {
            show: false,
            set: Some("no_equals".to_string()),
            reset: false,
        };
        // Should not panic, just print error
        execute_config(&config, &args);
    }

    #[test]
    fn test_print_current_config() {
        let config = CliConfig::default();
        // Should not panic
        print_current_config(&config);
    }

    #[test]
    fn test_print_default_config() {
        // Should not panic
        print_default_config();
    }

    #[test]
    fn test_set_config_value_valid() {
        // Should not panic
        set_config_value("key=value");
    }

    #[test]
    fn test_set_config_value_invalid() {
        // Should not panic, just print error
        set_config_value("invalid");
    }
}
