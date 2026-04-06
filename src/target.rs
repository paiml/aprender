//! Multi-Target Deployment Support
//!
//! Per `docs/specifications/serve-deploy-apr.md` Section 6 and §11.4:
//! Support for Lambda, Docker, WASM Edge, and bare metal deployments.
//!
//! ## Supported Targets
//!
//! | Target | Features | Use Case |
//! |--------|----------|----------|
//! | Native | Full (SIMD, GPU, threads) | Bare metal, EC2 |
//! | Lambda | SIMD, no GPU | AWS Lambda ARM64 |
//! | Docker | Full | Container deployments |
//! | WASM | CPU-only, no threads | Cloudflare Workers |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use realizar::target::{DeployTarget, TargetCapabilities};
//!
//! let target = DeployTarget::detect();
//! let caps = target.capabilities();
//!
//! if caps.supports_simd {
//!     // Use SIMD-accelerated inference
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Deployment target enumeration
///
/// Per spec §6: Multi-target deployment support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeployTarget {
    /// Native binary (bare metal, EC2, etc.)
    Native,
    /// AWS Lambda (ARM64 Graviton or `x86_64`)
    Lambda,
    /// Docker container
    Docker,
    /// WebAssembly (Cloudflare Workers, browser)
    Wasm,
}

impl DeployTarget {
    /// Detect current deployment target at runtime
    ///
    /// Detection heuristics:
    /// - WASM: `cfg!(target_arch = "wasm32")`
    /// - Lambda: `AWS_LAMBDA_FUNCTION_NAME` env var
    /// - Docker: `/.dockerenv` file exists
    /// - Native: fallback
    #[must_use]
    pub fn detect() -> Self {
        // WASM detection at compile time
        #[cfg(target_arch = "wasm32")]
        {
            return Self::Wasm;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Lambda detection via environment
            if std::env::var("AWS_LAMBDA_FUNCTION_NAME").is_ok() {
                return Self::Lambda;
            }

            // Docker detection via /.dockerenv
            if std::path::Path::new("/.dockerenv").exists() {
                return Self::Docker;
            }

            // Default to native
            Self::Native
        }
    }

    /// Get capabilities for this target
    #[must_use]
    pub const fn capabilities(&self) -> TargetCapabilities {
        match self {
            // Native and Docker have full capabilities
            Self::Native | Self::Docker => TargetCapabilities {
                supports_simd: true,
                supports_gpu: true,
                supports_threads: true,
                supports_filesystem: true,
                supports_async_io: true,
                max_memory_mb: 0, // Unlimited (container limit applies for Docker)
            },
            Self::Lambda => TargetCapabilities {
                supports_simd: true,
                supports_gpu: false,
                supports_threads: true,
                supports_filesystem: false, // /tmp only
                supports_async_io: true,
                max_memory_mb: 10240, // Max Lambda memory
            },
            Self::Wasm => TargetCapabilities {
                supports_simd: false, // WASM SIMD limited
                supports_gpu: false,
                supports_threads: false,    // No rayon
                supports_filesystem: false, // Must use include_bytes!
                supports_async_io: false,   // Limited
                max_memory_mb: 128,         // Typical worker limit
            },
        }
    }

    /// Get target name as string
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Lambda => "lambda",
            Self::Docker => "docker",
            Self::Wasm => "wasm",
        }
    }

    /// Check if target supports a specific feature
    #[must_use]
    pub const fn supports(&self, feature: TargetFeature) -> bool {
        let caps = self.capabilities();
        match feature {
            TargetFeature::Simd => caps.supports_simd,
            TargetFeature::Gpu => caps.supports_gpu,
            TargetFeature::Threads => caps.supports_threads,
            TargetFeature::Filesystem => caps.supports_filesystem,
            TargetFeature::AsyncIo => caps.supports_async_io,
        }
    }
}

impl std::fmt::Display for DeployTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Target capabilities
///
/// Per spec §6.3: WASM Limitations table
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)] // Capability flags are naturally boolean
pub struct TargetCapabilities {
    /// SIMD acceleration available (AVX2, NEON, etc.)
    pub supports_simd: bool,
    /// GPU acceleration available (wgpu)
    pub supports_gpu: bool,
    /// Multi-threading available (rayon)
    pub supports_threads: bool,
    /// Filesystem access available
    pub supports_filesystem: bool,
    /// Async I/O available
    pub supports_async_io: bool,
    /// Maximum memory in MB (0 = unlimited)
    pub max_memory_mb: u32,
}

impl TargetCapabilities {
    /// Check if all required features are available
    #[must_use]
    pub const fn has_all(&self, required: &[TargetFeature]) -> bool {
        let mut i = 0;
        while i < required.len() {
            let has = match required[i] {
                TargetFeature::Simd => self.supports_simd,
                TargetFeature::Gpu => self.supports_gpu,
                TargetFeature::Threads => self.supports_threads,
                TargetFeature::Filesystem => self.supports_filesystem,
                TargetFeature::AsyncIo => self.supports_async_io,
            };
            if !has {
                return false;
            }
            i += 1;
        }
        true
    }
}

/// Target feature flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetFeature {
    /// SIMD acceleration
    Simd,
    /// GPU acceleration
    Gpu,
    /// Multi-threading
    Threads,
    /// Filesystem access
    Filesystem,
    /// Async I/O
    AsyncIo,
}

/// Docker build configuration
///
/// Per spec §6.2: Multi-stage Dockerfile patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerConfig {
    /// Base image for builder stage
    pub builder_image: String,
    /// Base image for runtime stage
    pub runtime_image: String,
    /// Target triple for cross-compilation
    pub target_triple: String,
    /// Whether to strip binary
    pub strip_binary: bool,
    /// Exposed port
    pub expose_port: u16,
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            builder_image: "rust:1.83".to_string(),
            runtime_image: "gcr.io/distroless/static-debian12:latest".to_string(),
            target_triple: "x86_64-unknown-linux-musl".to_string(),
            strip_binary: true,
            expose_port: 8080,
        }
    }
}

impl DockerConfig {
    /// Create config for ARM64 (Graviton)
    #[must_use]
    pub fn arm64() -> Self {
        Self {
            target_triple: "aarch64-unknown-linux-musl".to_string(),
            ..Self::default()
        }
    }

    /// Create config for minimal scratch image
    #[must_use]
    pub fn scratch() -> Self {
        Self {
            runtime_image: "scratch".to_string(),
            ..Self::default()
        }
    }

    /// Generate Dockerfile content
    #[must_use]
    pub fn generate_dockerfile(&self) -> String {
        format!(
            r#"# Auto-generated by realizar target module
# Per docs/specifications/serve-deploy-apr.md Section 6.2

# Stage 1: Build
FROM {builder} AS builder
WORKDIR /build

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {{}}" > src/main.rs
RUN cargo build --release --target {target}
RUN rm -rf src

# Build actual binary
COPY src ./src
RUN cargo build --release --target {target}
{strip}

# Stage 2: Runtime
FROM {runtime}
COPY --from=builder /build/target/{target}/release/realizar /serve
EXPOSE {port}
ENTRYPOINT ["/serve"]
"#,
            builder = self.builder_image,
            runtime = self.runtime_image,
            target = self.target_triple,
            strip = if self.strip_binary {
                format!(
                    "RUN strip target/{}/release/realizar || true",
                    self.target_triple
                )
            } else {
                String::new()
            },
            port = self.expose_port
        )
    }

    /// Estimated image size in MB
    #[must_use]
    pub fn estimated_size_mb(&self) -> u32 {
        let base_size = if self.runtime_image.contains("scratch") {
            0
        } else if self.runtime_image.contains("distroless/static") {
            2
        } else if self.runtime_image.contains("distroless/cc") {
            20
        } else {
            50 // Generic estimate
        };

        // Binary size estimate: ~10MB for release build
        let binary_size = if self.strip_binary { 5 } else { 10 };

        base_size + binary_size
    }
}

/// WASM build configuration
///
/// Per spec §6.3: WASM Edge deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    /// wasm-pack target (web, bundler, nodejs)
    pub target: WasmTarget,
    /// Output directory
    pub out_dir: String,
    /// Enable WASM SIMD (experimental)
    pub enable_simd: bool,
}

/// WASM build targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmTarget {
    /// Browser via ES modules
    Web,
    /// Bundler (webpack, etc.)
    Bundler,
    /// Node.js
    NodeJs,
}

impl WasmTarget {
    /// Get wasm-pack target flag
    #[must_use]
    pub const fn flag(&self) -> &'static str {
        match self {
            Self::Web => "web",
            Self::Bundler => "bundler",
            Self::NodeJs => "nodejs",
        }
    }
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            target: WasmTarget::Web,
            out_dir: "pkg".to_string(),
            enable_simd: false,
        }
    }
}

impl WasmConfig {
    /// Generate build command
    #[must_use]
    pub fn build_command(&self) -> String {
        let mut cmd = format!(
            "wasm-pack build --target {} --release --out-dir {}",
            self.target.flag(),
            self.out_dir
        );

        if self.enable_simd {
            cmd.push_str(" -- -C target-feature=+simd128");
        }

        cmd
    }

    /// Generate Cloudflare Worker template
    #[must_use]
    pub fn cloudflare_worker_template(&self) -> String {
        format!(
            r"// Auto-generated Cloudflare Worker
// Per docs/specifications/serve-deploy-apr.md Section 6.3

import init, {{ predict }} from './{}/realizar.js';

let initialized = false;

export default {{
  async fetch(request, env) {{
    if (!initialized) {{
      await init();
      initialized = true;
    }}

    if (request.method !== 'POST') {{
      return new Response('Method not allowed', {{ status: 405 }});
    }}

    try {{
      const body = await request.json();
      const result = predict(body.features);

      return new Response(JSON.stringify(result), {{
        headers: {{ 'Content-Type': 'application/json' }}
      }});
    }} catch (e) {{
      return new Response(JSON.stringify({{ error: e.message }}), {{
        status: 500,
        headers: {{ 'Content-Type': 'application/json' }}
      }});
    }}
  }}
}};
",
            self.out_dir
        )
    }
}

/// Build manifest for CI/CD
///
/// Per spec §11.4: Multi-target build support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildManifest {
    /// Version from Cargo.toml
    pub version: String,
    /// Git commit hash
    pub git_hash: Option<String>,
    /// Build timestamp
    pub build_time: String,
    /// Targets to build
    pub targets: Vec<BuildTarget>,
}

/// Individual build target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTarget {
    /// Target name
    pub name: String,
    /// Rust target triple
    pub triple: String,
    /// Deploy target type
    pub deploy_target: DeployTarget,
    /// Build features to enable
    pub features: Vec<String>,
}

include!("target_default_all_build.rs");
