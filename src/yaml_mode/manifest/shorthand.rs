//! Human-readable value shorthand for YAML configs.
//!
//! Supports SI suffix notation for integer fields:
//! - `32K` → 32,768 (kibi, powers of 2)
//! - `1M` → 1,048,576 (mebi)
//! - `10B` → 10,000,000,000 (giga, base-10 for token counts)
//!
//! Also supports:
//! - Plain integers: `1024`
//! - Underscore notation: `32_768` (YAML native)
//! - Scientific notation strings: `"1e6"` → 1,000,000

use serde::{Deserialize, Deserializer};

/// Parse a human-readable size string into a usize.
///
/// Supports:
/// - Plain numbers: "1024", "32768"
/// - SI suffixes (binary): "32K" (32*1024), "1M" (1*1024²), "1G" (1*1024³)
/// - SI suffixes (decimal): "10B" (10*10⁹), "1T" (1*10¹²)
/// - Scientific notation: "1e6", "3.2e4"
///
/// Note: K/M use binary (powers of 2) since they're used for model dimensions.
/// B/T use decimal since they're used for token/parameter counts where "10B" means 10 billion.
pub fn parse_human_usize(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty string".into());
    }

    // Try scientific notation first (e.g., "1e6", "3.2e4")
    if s.contains('e') || s.contains('E') {
        return s
            .parse::<f64>()
            .map(|v| v as usize)
            .map_err(|e| format!("invalid scientific notation '{s}': {e}"));
    }

    // Check for SI suffix
    let (num_str, multiplier) = match s.as_bytes().last() {
        Some(b'K' | b'k') => (&s[..s.len() - 1], 1024_usize),
        Some(b'M' | b'm') => (&s[..s.len() - 1], 1024 * 1024),
        Some(b'G' | b'g') => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        Some(b'B' | b'b') => (&s[..s.len() - 1], 1_000_000_000_usize),
        Some(b'T' | b't') => (&s[..s.len() - 1], 1_000_000_000_000_usize),
        _ => (s, 1),
    };

    // Parse the numeric part (allow float for "1.5K" etc.)
    if num_str.contains('.') {
        let v: f64 = num_str.parse().map_err(|e| format!("invalid number '{num_str}': {e}"))?;
        Ok((v * multiplier as f64) as usize)
    } else {
        let v: usize = num_str.parse().map_err(|e| format!("invalid number '{num_str}': {e}"))?;
        v.checked_mul(multiplier).ok_or_else(|| format!("overflow: {v} * {multiplier}"))
    }
}

/// Deserialize an `Option<usize>` that accepts both numbers and human-readable strings.
///
/// # Examples (YAML)
/// ```yaml
/// vocab_size: 32K       # → Some(32768)
/// vocab_size: 32768     # → Some(32768)
/// vocab_size: "1e5"     # → Some(100000)
/// ```
pub fn deserialize_human_usize_opt<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum NumOrStr {
        Num(usize),
        Float(f64),
        Str(String),
    }

    let opt: Option<NumOrStr> = Option::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some(NumOrStr::Num(n)) => Ok(Some(n)),
        Some(NumOrStr::Float(f)) => Ok(Some(f as usize)),
        Some(NumOrStr::Str(s)) => parse_human_usize(&s).map(Some).map_err(serde::de::Error::custom),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_numbers() {
        assert_eq!(parse_human_usize("1024").expect("valid"), 1024);
        assert_eq!(parse_human_usize("0").expect("valid"), 0);
        assert_eq!(parse_human_usize("32768").expect("valid"), 32768);
    }

    #[test]
    fn test_si_suffix_binary() {
        assert_eq!(parse_human_usize("32K").expect("valid"), 32 * 1024);
        assert_eq!(parse_human_usize("1M").expect("valid"), 1024 * 1024);
        assert_eq!(parse_human_usize("1G").expect("valid"), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_si_suffix_lowercase() {
        assert_eq!(parse_human_usize("32k").expect("valid"), 32 * 1024);
        assert_eq!(parse_human_usize("1m").expect("valid"), 1024 * 1024);
    }

    #[test]
    fn test_si_suffix_decimal() {
        assert_eq!(parse_human_usize("10B").expect("valid"), 10_000_000_000);
        assert_eq!(parse_human_usize("1T").expect("valid"), 1_000_000_000_000);
    }

    #[test]
    fn test_scientific_notation() {
        assert_eq!(parse_human_usize("1e6").expect("valid"), 1_000_000);
        assert_eq!(parse_human_usize("3.2e4").expect("valid"), 32000);
        assert_eq!(parse_human_usize("1E5").expect("valid"), 100_000);
    }

    #[test]
    fn test_fractional_suffix() {
        assert_eq!(parse_human_usize("1.5K").expect("valid"), 1536); // 1.5 * 1024
        assert_eq!(parse_human_usize("0.5M").expect("valid"), 524_288); // 0.5 * 1M
    }

    #[test]
    fn test_empty_string_errors() {
        assert!(parse_human_usize("").is_err());
    }

    #[test]
    fn test_invalid_string_errors() {
        assert!(parse_human_usize("abc").is_err());
        assert!(parse_human_usize("K").is_err());
    }

    #[test]
    fn test_whitespace_trimmed() {
        assert_eq!(parse_human_usize("  32K  ").expect("valid"), 32 * 1024);
    }

    #[test]
    fn test_serde_deserialize_number() {
        #[derive(Deserialize)]
        struct Config {
            #[serde(
                default,
                skip_serializing_if = "Option::is_none",
                deserialize_with = "deserialize_human_usize_opt"
            )]
            vocab_size: Option<usize>,
        }

        let yaml = "vocab_size: 32768";
        let config: Config = serde_yaml::from_str(yaml).expect("should parse");
        assert_eq!(config.vocab_size, Some(32768));
    }

    #[test]
    fn test_serde_deserialize_string_suffix() {
        #[derive(Deserialize)]
        struct Config {
            #[serde(
                default,
                skip_serializing_if = "Option::is_none",
                deserialize_with = "deserialize_human_usize_opt"
            )]
            vocab_size: Option<usize>,
        }

        let yaml = "vocab_size: \"32K\"";
        let config: Config = serde_yaml::from_str(yaml).expect("should parse");
        assert_eq!(config.vocab_size, Some(32 * 1024));
    }

    #[test]
    fn test_serde_deserialize_none() {
        #[derive(Deserialize)]
        struct Config {
            #[serde(
                default,
                skip_serializing_if = "Option::is_none",
                deserialize_with = "deserialize_human_usize_opt"
            )]
            vocab_size: Option<usize>,
        }

        let yaml = "other: 123";
        let config: Config = serde_yaml::from_str(yaml).expect("should parse");
        assert_eq!(config.vocab_size, None);
    }

    #[test]
    fn test_serde_deserialize_scientific() {
        #[derive(Deserialize)]
        struct Config {
            #[serde(
                default,
                skip_serializing_if = "Option::is_none",
                deserialize_with = "deserialize_human_usize_opt"
            )]
            count: Option<usize>,
        }

        let yaml = "count: \"1e6\"";
        let config: Config = serde_yaml::from_str(yaml).expect("should parse");
        assert_eq!(config.count, Some(1_000_000));
    }
}
