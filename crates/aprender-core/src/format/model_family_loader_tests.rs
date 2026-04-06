
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_scalar_types() {
        assert!(matches!(parse_scalar("42"), YamlValue::Int(42)));
        assert!(matches!(parse_scalar("3.14"), YamlValue::Float(_)));
        assert!(matches!(parse_scalar("true"), YamlValue::Bool(true)));
        assert!(matches!(parse_scalar("false"), YamlValue::Bool(false)));
        assert!(matches!(parse_scalar("null"), YamlValue::Null));
        assert!(matches!(parse_scalar("\"hello\""), YamlValue::String(_)));
        assert!(matches!(parse_scalar("hello"), YamlValue::String(_)));
    }

    #[test]
    fn test_parse_simple_yaml() {
        let yaml = r#"
family: qwen2
display_name: "Qwen2"
vendor: Alibaba
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        assert_eq!(result.get_str("family"), Some("qwen2"));
        assert_eq!(result.get_str("display_name"), Some("Qwen2"));
        assert_eq!(result.get_str("vendor"), Some("Alibaba"));
    }

    #[test]
    fn test_parse_nested_yaml() {
        let yaml = r#"
constraints:
  attention_type: gqa
  has_bias: true
  activation: silu
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let constraints = result.get("constraints").expect("constraints");
        assert_eq!(constraints.get_str("attention_type"), Some("gqa"));
        assert_eq!(constraints.get_bool("has_bias"), Some(true));
    }

    #[test]
    fn test_parse_sequence_yaml() {
        let yaml = r#"
architectures:
  - Qwen2ForCausalLM
  - Qwen2ForSequenceClassification
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let arch = result
            .get("architectures")
            .and_then(YamlValue::as_sequence)
            .expect("architectures");
        assert_eq!(arch.len(), 2);
        assert_eq!(arch[0].as_str(), Some("Qwen2ForCausalLM"));
    }

    #[test]
    fn test_load_qwen2_yaml() {
        let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
        let yaml_path = contracts_dir.join("model-families/qwen2.yaml");
        if yaml_path.exists() {
            let config = load_family_yaml(&yaml_path).expect("load qwen2 yaml");
            assert_eq!(config.family, "qwen2");
            assert_eq!(config.vendor, "Alibaba");
            assert!(!config.size_variants.is_empty());
            assert!(config.size_variants.contains_key("0.5b"));

            let half_b = &config.size_variants["0.5b"];
            assert_eq!(half_b.hidden_dim, 896);
            assert_eq!(half_b.num_layers, 24);
        }
    }

    #[test]
    fn test_load_family_registry() {
        let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
        if contracts_dir.join("model-families").exists() {
            let registry = load_family_registry(&contracts_dir).expect("load registry");
            assert!(
                !registry.is_empty(),
                "Registry should have at least one family"
            );

            let names = registry.family_names();
            assert!(names.contains(&"qwen2"), "Registry should contain qwen2");
        }
    }

    #[test]
    fn test_parse_null_values() {
        let yaml = r#"
tensor_template:
  embedding: model.embed_tokens.weight
  per_layer:
    q_proj: "model.layers.{n}.self_attn.q_proj.weight"
    q_proj_bias: null
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let template = parse_tensor_template(result.get("tensor_template").expect("template"))
            .expect("parse template");
        assert!(template.per_layer.get("q_proj").expect("q_proj").is_some());
        assert!(template
            .per_layer
            .get("q_proj_bias")
            .expect("q_proj_bias")
            .is_none());
    }

    #[test]
    fn test_parse_scalar_quoted_single() {
        assert!(matches!(parse_scalar("'hello'"), YamlValue::String(_)));
        if let YamlValue::String(s) = parse_scalar("'hello'") {
            assert_eq!(s, "hello");
        }
    }

    #[test]
    fn test_parse_scalar_tilde_null() {
        assert!(matches!(parse_scalar("~"), YamlValue::Null));
    }

    #[test]
    fn test_parse_scalar_yes_no() {
        assert!(matches!(parse_scalar("yes"), YamlValue::Bool(true)));
        assert!(matches!(parse_scalar("no"), YamlValue::Bool(false)));
    }

    #[test]
    fn test_yaml_value_as_i64() {
        let v = YamlValue::Int(42);
        assert_eq!(v.as_i64(), Some(42));
        let v = YamlValue::Float(3.9);
        assert_eq!(v.as_i64(), Some(3));
        let v = YamlValue::String("nope".to_string());
        assert_eq!(v.as_i64(), None);
    }

    #[test]
    fn test_yaml_value_as_f64() {
        let v = YamlValue::Float(3.14);
        assert!((v.as_f64().expect("f64") - 3.14).abs() < f64::EPSILON);
        let v = YamlValue::Int(42);
        assert!((v.as_f64().expect("f64") - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_yaml_value_as_bool() {
        let v = YamlValue::Bool(true);
        assert_eq!(v.as_bool(), Some(true));
        let v = YamlValue::String("nope".to_string());
        assert_eq!(v.as_bool(), None);
    }

    #[test]
    fn test_yaml_value_as_usize() {
        let v = YamlValue::Int(10);
        assert_eq!(v.as_usize(), Some(10));
        let v = YamlValue::Int(-1);
        assert_eq!(v.as_usize(), None);
    }

    #[test]
    fn test_yaml_value_as_sequence() {
        let v = YamlValue::Sequence(vec![YamlValue::Int(1)]);
        assert!(v.as_sequence().is_some());
        let v = YamlValue::Int(1);
        assert!(v.as_sequence().is_none());
    }

    #[test]
    fn test_yaml_value_as_mapping() {
        let v = YamlValue::Mapping(vec![("key".to_string(), YamlValue::Int(1))]);
        assert!(v.as_mapping().is_some());
        let v = YamlValue::Int(1);
        assert!(v.as_mapping().is_none());
    }

    #[test]
    fn test_yaml_value_get_on_non_mapping() {
        let v = YamlValue::Int(42);
        assert!(v.get("key").is_none());
        assert!(v.get_str("key").is_none());
        assert!(v.get_usize("key").is_none());
        assert!(v.get_f64("key").is_none());
        assert!(v.get_bool("key").is_none());
    }

    #[test]
    fn test_yaml_value_as_str_on_non_string() {
        let v = YamlValue::Int(42);
        assert!(v.as_str().is_none());
    }

    #[test]
    fn test_parse_empty_yaml() {
        let yaml = "";
        let result = parse_yaml(yaml).expect("parse yaml");
        assert!(result.get("anything").is_none());
    }

    #[test]
    fn test_parse_yaml_comments_only() {
        let yaml = "# just a comment\n# another comment\n";
        let result = parse_yaml(yaml).expect("parse yaml");
        assert!(result.get("anything").is_none());
    }

    #[test]
    fn test_default_contracts_dir() {
        let root = Path::new("/tmp/project");
        let dir = default_contracts_dir(root);
        assert_eq!(dir, Path::new("/tmp/project/contracts"));
    }

    #[test]
    fn test_load_family_registry_missing_dir() {
        let result = load_family_registry(Path::new("/nonexistent/path"));
        assert!(result.is_ok());
        assert!(result.expect("registry").is_empty());
    }

    #[test]
    fn test_parse_family_yaml_missing_family() {
        let yaml = r#"
display_name: "Test"
vendor: Test
architectures:
  - TestArch
hf_pattern: "test/*"
size_variants:
  small:
    parameters: "1B"
    hidden_dim: 768
    num_layers: 12
    num_heads: 12
    intermediate_dim: 3072
    vocab_size: 30000
constraints:
  attention_type: mha
tensor_template:
  embedding: "model.embed.weight"
"#;
        let result = parse_family_yaml(yaml, Path::new("test.yaml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_shape_template_flat() {
        let yaml = r#"
shape_template:
  embedding: "[vocab_size, hidden_dim]"
  lm_head: "[vocab_size, hidden_dim]"
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let shapes = parse_shape_template(result.get("shape_template").expect("shape_template"));
        assert_eq!(
            shapes.shapes.get("embedding"),
            Some(&"[vocab_size, hidden_dim]".to_string())
        );
        assert_eq!(
            shapes.shapes.get("lm_head"),
            Some(&"[vocab_size, hidden_dim]".to_string())
        );
    }

    #[test]
    fn test_parse_chat_template() {
        let yaml = r#"
chat_template:
  format: chatml
  template: "test template"
  bos_token: "<s>"
  eos_token: "</s>"
  special_tokens:
    pad: "<pad>"
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let ct = parse_chat_template(result.get("chat_template").expect("chat_template"))
            .expect("parse chat template");
        assert_eq!(ct.format, "chatml");
        assert_eq!(ct.bos_token, "<s>");
        assert_eq!(ct.special_tokens.get("pad"), Some(&"<pad>".to_string()));
    }

    #[test]
    fn test_parse_certification() {
        let yaml = r#"
certification:
  playbook_path: "path/to/playbook.yaml"
  csv_family_key: "test"
  size_categories:
    small: tiny
    large: xlarge
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let cert = parse_certification(result.get("certification").expect("certification"))
            .expect("parse certification");
        assert_eq!(cert.playbook_path, "path/to/playbook.yaml");
        assert_eq!(cert.csv_family_key, "test");
        assert_eq!(cert.size_categories.get("small"), Some(&"tiny".to_string()));
    }
}
