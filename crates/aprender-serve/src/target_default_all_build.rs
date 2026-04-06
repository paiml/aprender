
impl BuildManifest {
    /// Create default manifest with all targets
    #[must_use]
    pub fn default_all_targets() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_hash: None,
            build_time: String::new(),
            targets: vec![
                BuildTarget {
                    name: "linux-x86_64".to_string(),
                    triple: "x86_64-unknown-linux-musl".to_string(),
                    deploy_target: DeployTarget::Docker,
                    features: vec!["server".to_string()],
                },
                BuildTarget {
                    name: "linux-arm64".to_string(),
                    triple: "aarch64-unknown-linux-musl".to_string(),
                    deploy_target: DeployTarget::Lambda,
                    features: vec!["lambda".to_string()],
                },
                BuildTarget {
                    name: "wasm".to_string(),
                    triple: "wasm32-unknown-unknown".to_string(),
                    deploy_target: DeployTarget::Wasm,
                    features: vec![],
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // RED PHASE: Failing tests for multi-target support
    // Per EXTREME TDD: Write tests FIRST
    // ==========================================================================

    // --------------------------------------------------------------------------
    // Test: DeployTarget detection and capabilities
    // --------------------------------------------------------------------------

    #[test]
    fn test_deploy_target_detect_native() {
        // In test environment without Lambda/Docker env vars, should detect native
        // Note: This test may behave differently in CI with Docker
        let target = DeployTarget::detect();
        // Should be Native, Lambda, or Docker depending on environment
        assert!(
            matches!(
                target,
                DeployTarget::Native | DeployTarget::Lambda | DeployTarget::Docker
            ),
            "Should detect a valid non-WASM target"
        );
    }

    #[test]
    fn test_deploy_target_names() {
        assert_eq!(DeployTarget::Native.name(), "native");
        assert_eq!(DeployTarget::Lambda.name(), "lambda");
        assert_eq!(DeployTarget::Docker.name(), "docker");
        assert_eq!(DeployTarget::Wasm.name(), "wasm");
    }

    #[test]
    fn test_deploy_target_display() {
        assert_eq!(format!("{}", DeployTarget::Native), "native");
        assert_eq!(format!("{}", DeployTarget::Wasm), "wasm");
    }

    #[test]
    fn test_native_capabilities() {
        let caps = DeployTarget::Native.capabilities();
        assert!(caps.supports_simd);
        assert!(caps.supports_gpu);
        assert!(caps.supports_threads);
        assert!(caps.supports_filesystem);
        assert!(caps.supports_async_io);
        assert_eq!(caps.max_memory_mb, 0); // Unlimited
    }

    #[test]
    fn test_lambda_capabilities() {
        let caps = DeployTarget::Lambda.capabilities();
        assert!(caps.supports_simd);
        assert!(!caps.supports_gpu); // No GPU in Lambda
        assert!(caps.supports_threads);
        assert!(!caps.supports_filesystem); // /tmp only
        assert!(caps.supports_async_io);
        assert_eq!(caps.max_memory_mb, 10240);
    }

    #[test]
    fn test_wasm_capabilities() {
        let caps = DeployTarget::Wasm.capabilities();
        assert!(!caps.supports_simd); // Limited SIMD
        assert!(!caps.supports_gpu);
        assert!(!caps.supports_threads); // No rayon
        assert!(!caps.supports_filesystem); // Must use include_bytes!
        assert!(!caps.supports_async_io);
        assert_eq!(caps.max_memory_mb, 128);
    }

    #[test]
    fn test_deploy_target_supports_feature() {
        assert!(DeployTarget::Native.supports(TargetFeature::Simd));
        assert!(DeployTarget::Native.supports(TargetFeature::Gpu));

        assert!(DeployTarget::Lambda.supports(TargetFeature::Simd));
        assert!(!DeployTarget::Lambda.supports(TargetFeature::Gpu));

        assert!(!DeployTarget::Wasm.supports(TargetFeature::Threads));
        assert!(!DeployTarget::Wasm.supports(TargetFeature::Filesystem));
    }

    #[test]
    fn test_capabilities_has_all() {
        let native_caps = DeployTarget::Native.capabilities();
        assert!(native_caps.has_all(&[TargetFeature::Simd, TargetFeature::Gpu]));
        assert!(native_caps.has_all(&[TargetFeature::Threads, TargetFeature::Filesystem]));

        let wasm_caps = DeployTarget::Wasm.capabilities();
        assert!(!wasm_caps.has_all(&[TargetFeature::Simd]));
        assert!(!wasm_caps.has_all(&[TargetFeature::Threads]));
        assert!(wasm_caps.has_all(&[])); // Empty requirements
    }

    // --------------------------------------------------------------------------
    // Test: Docker configuration
    // --------------------------------------------------------------------------

    #[test]
    fn test_docker_config_default() {
        let config = DockerConfig::default();
        assert_eq!(config.builder_image, "rust:1.83");
        assert!(config.runtime_image.contains("distroless"));
        assert_eq!(config.target_triple, "x86_64-unknown-linux-musl");
        assert!(config.strip_binary);
        assert_eq!(config.expose_port, 8080);
    }

    #[test]
    fn test_docker_config_arm64() {
        let config = DockerConfig::arm64();
        assert!(config.target_triple.contains("aarch64"));
    }

    #[test]
    fn test_docker_config_scratch() {
        let config = DockerConfig::scratch();
        assert_eq!(config.runtime_image, "scratch");
    }

    #[test]
    fn test_docker_generate_dockerfile() {
        let config = DockerConfig::default();
        let dockerfile = config.generate_dockerfile();

        assert!(dockerfile.contains("FROM rust:1.83 AS builder"));
        assert!(dockerfile.contains("distroless"));
        assert!(dockerfile.contains("x86_64-unknown-linux-musl"));
        assert!(dockerfile.contains("EXPOSE 8080"));
        assert!(dockerfile.contains("strip"));
    }

    #[test]
    fn test_docker_estimated_size() {
        let distroless = DockerConfig::default();
        assert!(distroless.estimated_size_mb() < 20); // ~7MB

        let scratch = DockerConfig::scratch();
        assert!(scratch.estimated_size_mb() < 10); // ~5MB (binary only)
    }

    // --------------------------------------------------------------------------
    // Test: WASM configuration
    // --------------------------------------------------------------------------

    #[test]
    fn test_wasm_config_default() {
        let config = WasmConfig::default();
        assert_eq!(config.target, WasmTarget::Web);
        assert_eq!(config.out_dir, "pkg");
        assert!(!config.enable_simd);
    }

    #[test]
    fn test_wasm_target_flags() {
        assert_eq!(WasmTarget::Web.flag(), "web");
        assert_eq!(WasmTarget::Bundler.flag(), "bundler");
        assert_eq!(WasmTarget::NodeJs.flag(), "nodejs");
    }

    #[test]
    fn test_wasm_build_command() {
        let config = WasmConfig::default();
        let cmd = config.build_command();

        assert!(cmd.contains("wasm-pack build"));
        assert!(cmd.contains("--target web"));
        assert!(cmd.contains("--release"));
        assert!(cmd.contains("--out-dir pkg"));
    }

    #[test]
    fn test_wasm_build_command_with_simd() {
        let config = WasmConfig {
            enable_simd: true,
            ..WasmConfig::default()
        };
        let cmd = config.build_command();

        assert!(cmd.contains("simd128"));
    }

    #[test]
    fn test_wasm_cloudflare_worker_template() {
        let config = WasmConfig::default();
        let template = config.cloudflare_worker_template();

        assert!(template.contains("import init"));
        assert!(template.contains("predict"));
        assert!(template.contains("async fetch"));
        assert!(template.contains("application/json"));
    }

    // --------------------------------------------------------------------------
    // Test: Build manifest
    // --------------------------------------------------------------------------

    #[test]
    fn test_build_manifest_default() {
        let manifest = BuildManifest::default_all_targets();

        assert!(!manifest.version.is_empty());
        assert_eq!(manifest.targets.len(), 3);

        // Check target names
        let names: Vec<_> = manifest.targets.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"linux-x86_64"));
        assert!(names.contains(&"linux-arm64"));
        assert!(names.contains(&"wasm"));
    }

    #[test]
    fn test_build_manifest_targets() {
        let manifest = BuildManifest::default_all_targets();

        let x86_target = manifest
            .targets
            .iter()
            .find(|t| t.name == "linux-x86_64")
            .expect("test");
        assert_eq!(x86_target.deploy_target, DeployTarget::Docker);

        let arm_target = manifest
            .targets
            .iter()
            .find(|t| t.name == "linux-arm64")
            .expect("test");
        assert_eq!(arm_target.deploy_target, DeployTarget::Lambda);

        let wasm_target = manifest
            .targets
            .iter()
            .find(|t| t.name == "wasm")
            .expect("test");
        assert_eq!(wasm_target.deploy_target, DeployTarget::Wasm);
    }

    // --------------------------------------------------------------------------
    // Test: Serialization
    // --------------------------------------------------------------------------

    #[test]
    fn test_deploy_target_serialization() {
        let target = DeployTarget::Lambda;
        let json = serde_json::to_string(&target).expect("test");
        assert!(json.contains("Lambda"));

        let deserialized: DeployTarget = serde_json::from_str(&json).expect("test");
        assert_eq!(deserialized, DeployTarget::Lambda);
    }

    #[test]
    fn test_docker_config_serialization() {
        let config = DockerConfig::default();
        let json = serde_json::to_string(&config).expect("test");

        assert!(json.contains("rust:1.83"));
        assert!(json.contains("distroless"));

        let deserialized: DockerConfig = serde_json::from_str(&json).expect("test");
        assert_eq!(deserialized.expose_port, 8080);
    }

    #[test]
    fn test_wasm_config_serialization() {
        let config = WasmConfig::default();
        let json = serde_json::to_string(&config).expect("test");

        assert!(json.contains("Web"));
        assert!(json.contains("pkg"));

        let deserialized: WasmConfig = serde_json::from_str(&json).expect("test");
        assert_eq!(deserialized.target, WasmTarget::Web);
    }
}
