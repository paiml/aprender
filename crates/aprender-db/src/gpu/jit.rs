//! JIT WGSL Compiler for Kernel Fusion (CORE-003)
//!
//! Toyota Way: Muda elimination (waste of intermediate memory writes)
//!
//! This module provides runtime compilation of fused WGSL kernels.
//! Phase 1 MVP: Simplified approach with template-based code generation.
//!
//! Example: Filter + SUM fusion
//! - Non-fused: Filter → intermediate buffer → SUM (2 GPU passes, 1 memory write)
//! - Fused: Filter + SUM in single pass (1 GPU pass, 0 intermediate writes)
//!
//! References:
//! - Wu et al. (2012): Kernel fusion execution model
//! - Neumann (2011): JIT compilation for queries
//! - MonetDB/X100 (2005): Vectorized query execution

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Shader compilation cache for JIT-compiled kernels
///
/// Caches compiled shaders by query signature to avoid recompilation.
/// Thread-safe via Mutex for concurrent query execution.
pub struct ShaderCache {
    cache: Mutex<HashMap<String, Arc<wgpu::ShaderModule>>>,
}

impl ShaderCache {
    /// Create a new shader cache
    #[must_use]
    pub fn new() -> Self {
        Self { cache: Mutex::new(HashMap::new()) }
    }

    /// Get cached shader or insert new one
    ///
    /// # Arguments
    /// * `key` - Query signature (e.g., `"filter_gt_1000_sum"`)
    /// * `device` - GPU device for shader compilation
    /// * `shader_source` - WGSL shader source code
    ///
    /// # Returns
    /// Arc reference to compiled shader module (either cached or newly compiled)
    ///
    /// # Panics
    /// Panics if the cache mutex is poisoned (should never happen in normal operation)
    pub fn get_or_insert(
        &self,
        key: &str,
        device: &wgpu::Device,
        shader_source: &str,
    ) -> Arc<wgpu::ShaderModule> {
        let mut cache = self
            .cache
            .lock()
            .expect("Shader cache mutex poisoned (should never happen in normal operation)");

        if !cache.contains_key(key) {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(key),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
            cache.insert(key.to_string(), Arc::new(shader));
        }

        // Clone the Arc (cheap), not the ShaderModule
        Arc::clone(cache.get(key).expect("Shader must exist in cache after insertion"))
    }

    /// Get cache statistics
    ///
    /// # Panics
    /// Panics if the cache mutex is poisoned (should never happen in normal operation)
    #[must_use]
    pub fn stats(&self) -> (usize, usize) {
        let cache = self
            .cache
            .lock()
            .expect("Shader cache mutex poisoned (should never happen in normal operation)");
        (cache.len(), cache.capacity())
    }
}

impl Default for ShaderCache {
    fn default() -> Self {
        Self::new()
    }
}

/// JIT WGSL compiler for kernel fusion
///
/// Phase 1 MVP: Template-based code generation for common patterns.
/// Future: Full SQL AST → WGSL compilation in Phase 2.
pub struct JitCompiler {
    cache: ShaderCache,
}

impl JitCompiler {
    /// Create a new JIT compiler with shader cache
    #[must_use]
    pub fn new() -> Self {
        Self { cache: ShaderCache::new() }
    }

    /// Generate fused filter+sum kernel
    ///
    /// Fuses WHERE clause with SUM aggregation in single GPU pass.
    ///
    /// # Arguments
    /// * `filter_threshold` - Threshold value for filter (e.g., WHERE value > 1000)
    /// * `filter_op` - Filter operator ("gt", "lt", "eq", "gte", "lte")
    ///
    /// # Returns
    /// WGSL shader source code for fused kernel
    ///
    /// # Example
    /// ```ignore
    /// let shader = compiler.generate_fused_filter_sum(1000, "gt");
    /// // Generates: WHERE value > 1000, SUM(value) in single pass
    /// ```
    #[must_use]
    pub fn generate_fused_filter_sum(&self, filter_threshold: i32, filter_op: &str) -> String {
        // Convert operator to WGSL
        let wgsl_op = match filter_op {
            "lt" => "<",
            "eq" => "==",
            "gte" => ">=",
            "lte" => "<=",
            "ne" => "!=",
            _ => ">", // Default to greater-than (handles "gt" and unknown ops)
        };

        format!(
            r"
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<i32>>;

var<workgroup> shared_data: array<i32, 256>;

@compute @workgroup_size(256)
fn fused_filter_sum(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {{
    let tid = local_id.x;
    let gid = global_id.x;
    let input_size = arrayLength(&input);

    // Fused filter + load: Apply filter predicate during load
    // Eliminates intermediate buffer write (Muda elimination)
    var value: i32 = 0;
    if (gid < input_size) {{
        let data = input[gid];
        // Filter: WHERE value {wgsl_op} {filter_threshold}
        if (data {wgsl_op} {filter_threshold}) {{
            value = data;
        }}
    }}
    shared_data[tid] = value;
    workgroupBarrier();

    // Parallel reduction (same as unfused SUM kernel)
    var stride = 128u;
    while (stride > 0u) {{
        if (tid < stride && gid + stride < input_size) {{
            shared_data[tid] += shared_data[tid + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    // Write result
    if (tid == 0u) {{
        atomicAdd(&output[0], shared_data[0]);
    }}
}}
"
        )
    }

    /// Compile and cache fused filter+sum kernel
    ///
    /// # Arguments
    /// * `device` - GPU device for compilation
    /// * `filter_threshold` - Filter threshold value
    /// * `filter_op` - Filter operator
    ///
    /// # Returns
    /// Arc reference to compiled shader module (cached for reuse)
    pub fn compile_fused_filter_sum(
        &self,
        device: &wgpu::Device,
        filter_threshold: i32,
        filter_op: &str,
    ) -> Arc<wgpu::ShaderModule> {
        // Generate cache key from query signature
        let cache_key = format!("filter_{filter_op}_{filter_threshold}_sum");

        // Generate WGSL source
        let shader_source = self.generate_fused_filter_sum(filter_threshold, filter_op);

        // Get from cache or compile
        self.cache.get_or_insert(&cache_key, device, &shader_source)
    }

    /// Get cache statistics (size, capacity)
    #[must_use]
    pub fn cache_stats(&self) -> (usize, usize) {
        self.cache.stats()
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_cache_new() {
        let cache = ShaderCache::new();
        let (size, _capacity) = cache.stats();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_jit_compiler_new() {
        let compiler = JitCompiler::new();
        let (size, _capacity) = compiler.cache_stats();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_generate_fused_filter_sum() {
        let compiler = JitCompiler::new();

        // Test greater-than filter
        let greater_than = compiler.generate_fused_filter_sum(1000, "gt");
        assert!(greater_than.contains("if (data > 1000)"));
        assert!(greater_than.contains("fused_filter_sum"));

        // Test less-than filter
        let less_than = compiler.generate_fused_filter_sum(500, "lt");
        assert!(less_than.contains("if (data < 500)"));

        // Test equals filter
        let equals = compiler.generate_fused_filter_sum(42, "eq");
        assert!(equals.contains("if (data == 42)"));
    }

    #[test]
    fn test_shader_source_contains_fusion() {
        let compiler = JitCompiler::new();
        let shader = compiler.generate_fused_filter_sum(100, "gte");

        // Verify it contains key fusion components
        assert!(shader.contains("@workgroup_size(256)"));
        assert!(shader.contains("var<workgroup> shared_data"));
        assert!(shader.contains("atomicAdd"));
        assert!(shader.contains("workgroupBarrier"));

        // Verify filter is inline (fused)
        assert!(shader.contains("if (data >= 100)"));
    }

    #[test]
    fn test_all_filter_operators() {
        let compiler = JitCompiler::new();

        // Test all supported operators
        let gte_shader = compiler.generate_fused_filter_sum(10, "gte");
        assert!(gte_shader.contains("if (data >= 10)"));

        let lte_shader = compiler.generate_fused_filter_sum(20, "lte");
        assert!(lte_shader.contains("if (data <= 20)"));

        let ne_shader = compiler.generate_fused_filter_sum(30, "ne");
        assert!(ne_shader.contains("if (data != 30)"));

        // Test unknown operator defaults to >
        let unknown_shader = compiler.generate_fused_filter_sum(40, "unknown");
        assert!(unknown_shader.contains("if (data > 40)"));
    }

    #[test]
    fn test_shader_cache_default() {
        let cache = ShaderCache::default();
        let (size, _capacity) = cache.stats();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_jit_compiler_default() {
        let compiler = JitCompiler::default();
        let (size, _capacity) = compiler.cache_stats();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_cache_key_generation() {
        let compiler = JitCompiler::new();

        // Different thresholds should generate different shaders
        let shader1 = compiler.generate_fused_filter_sum(100, "gt");
        let shader2 = compiler.generate_fused_filter_sum(200, "gt");
        assert_ne!(shader1, shader2);

        // Different operators should generate different shaders
        let shader3 = compiler.generate_fused_filter_sum(100, "lt");
        assert_ne!(shader1, shader3);
    }

    #[test]
    fn test_wgsl_syntax_valid() {
        let compiler = JitCompiler::new();
        let shader = compiler.generate_fused_filter_sum(999, "eq");

        // Verify critical WGSL syntax elements
        assert!(shader.contains("@group(0) @binding(0)"));
        assert!(shader.contains("@group(0) @binding(1)"));
        assert!(shader.contains("@compute @workgroup_size(256)"));
        assert!(shader.contains("@builtin(global_invocation_id)"));
        assert!(shader.contains("@builtin(local_invocation_id)"));
        assert!(shader.contains("var<workgroup>"));
        assert!(shader.contains("var<storage, read>"));
        assert!(shader.contains("var<storage, read_write>"));
        assert!(shader.contains("array<atomic<i32>>"));
    }

    #[test]
    fn test_parallel_reduction_logic() {
        let compiler = JitCompiler::new();
        let shader = compiler.generate_fused_filter_sum(500, "gt");

        // Verify parallel reduction pattern (Harris 2007)
        assert!(shader.contains("var stride = 128u;"));
        assert!(shader.contains("while (stride > 0u)"));
        assert!(shader.contains("stride = stride / 2u;"));
        assert!(shader.contains("if (tid == 0u)"));
    }

    #[test]
    fn test_muda_elimination_comment() {
        let compiler = JitCompiler::new();
        let shader = compiler.generate_fused_filter_sum(100, "gt");

        // Verify Toyota Way: Muda elimination comment exists
        assert!(shader.contains("Eliminates intermediate buffer write"));
    }
}
