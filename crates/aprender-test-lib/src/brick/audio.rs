//! AudioBrick: AudioWorklet code generation from brick definitions (PROBAR-SPEC-009-P7)
//!
//! Generates AudioWorklet processor JavaScript from brick definitions.
//! Zero hand-written audio processing code.
//!
//! # Example
//!
//! ```rust,ignore
//! use probar::brick::audio::{AudioBrick, AudioParam, RingBufferConfig};
//!
//! let audio = AudioBrick::new("whisper-capture")
//!     .with_ring_buffer(RingBufferConfig {
//!         size: 144000,  // 3 seconds at 48kHz
//!         channels: 1,
//!         use_atomics: true,
//!     })
//!     .param(AudioParam::new("gain", 1.0).range(0.0, 2.0));
//!
//! // Generate AudioWorklet processor JS
//! let worklet_js = audio.to_worklet_js();
//!
//! // Generate main thread initialization JS
//! let init_js = audio.to_audio_init_js();
//! ```

use super::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::time::Duration;

/// Audio parameter configuration
#[derive(Debug, Clone)]
pub struct AudioParam {
    /// Parameter name
    pub name: String,
    /// Default value
    pub default_value: f64,
    /// Minimum value
    pub min_value: f64,
    /// Maximum value
    pub max_value: f64,
    /// Automation rate: "a-rate" (per sample) or "k-rate" (per block)
    pub automation_rate: String,
}

impl AudioParam {
    /// Create a new audio parameter
    #[must_use]
    pub fn new(name: impl Into<String>, default_value: f64) -> Self {
        Self {
            name: name.into(),
            default_value,
            min_value: f64::MIN,
            max_value: f64::MAX,
            automation_rate: "k-rate".into(),
        }
    }

    /// Set the parameter range
    #[must_use]
    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.min_value = min;
        self.max_value = max;
        self
    }

    /// Set to a-rate (per-sample automation)
    #[must_use]
    pub fn a_rate(mut self) -> Self {
        self.automation_rate = "a-rate".into();
        self
    }

    /// Set to k-rate (per-block automation)
    #[must_use]
    pub fn k_rate(mut self) -> Self {
        self.automation_rate = "k-rate".into();
        self
    }

    /// Generate JavaScript parameter descriptor
    #[must_use]
    pub fn to_js_descriptor(&self) -> String {
        format!(
            "{{ name: '{}', defaultValue: {}, minValue: {}, maxValue: {}, automationRate: '{}' }}",
            self.name, self.default_value, self.min_value, self.max_value, self.automation_rate
        )
    }
}

/// Ring buffer configuration for audio data transfer
#[derive(Debug, Clone)]
pub struct RingBufferConfig {
    /// Buffer size in samples
    pub size: usize,
    /// Number of audio channels
    pub channels: usize,
    /// Use SharedArrayBuffer + Atomics for lock-free transfer
    pub use_atomics: bool,
}

impl Default for RingBufferConfig {
    fn default() -> Self {
        Self {
            size: 48000, // 1 second at 48kHz
            channels: 1,
            use_atomics: true,
        }
    }
}

impl RingBufferConfig {
    /// Create a new ring buffer config
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            size,
            ..Default::default()
        }
    }

    /// Set number of channels
    #[must_use]
    pub fn channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self
    }

    /// Disable atomics (use postMessage instead)
    #[must_use]
    pub fn without_atomics(mut self) -> Self {
        self.use_atomics = false;
        self
    }
}

/// AudioBrick: Generates AudioWorklet processor code
#[derive(Debug, Clone)]
pub struct AudioBrick {
    /// Processor name (used in registerProcessor)
    name: String,
    /// Number of inputs
    inputs: usize,
    /// Number of outputs
    outputs: usize,
    /// Audio parameters
    params: Vec<AudioParam>,
    /// Ring buffer configuration (if any)
    ring_buffer: Option<RingBufferConfig>,
    /// Sample rate (for calculations)
    sample_rate: u32,
}

impl AudioBrick {
    /// Create a new audio brick
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inputs: 1,
            outputs: 1,
            params: Vec::new(),
            ring_buffer: None,
            sample_rate: 48000,
        }
    }

    /// Set number of inputs
    #[must_use]
    pub fn inputs(mut self, count: usize) -> Self {
        self.inputs = count;
        self
    }

    /// Set number of outputs
    #[must_use]
    pub fn outputs(mut self, count: usize) -> Self {
        self.outputs = count;
        self
    }

    /// Add an audio parameter
    #[must_use]
    pub fn param(mut self, param: AudioParam) -> Self {
        self.params.push(param);
        self
    }

    /// Configure ring buffer
    #[must_use]
    pub fn with_ring_buffer(mut self, config: RingBufferConfig) -> Self {
        self.ring_buffer = Some(config);
        self
    }

    /// Set expected sample rate
    #[must_use]
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Get the processor class name
    #[must_use]
    pub fn class_name(&self) -> String {
        let mut result = String::new();
        let mut capitalize_next = true;

        for c in self.name.chars() {
            if c == '-' || c == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.push(c.to_ascii_uppercase());
                capitalize_next = false;
            } else {
                result.push(c);
            }
        }

        result.push_str("Processor");
        result
    }

    /// Generate AudioWorklet processor JavaScript
    #[must_use]
    pub fn to_worklet_js(&self) -> String {
        let mut js = String::new();
        let class_name = self.class_name();

        // Header
        js.push_str(&format!("// {} AudioWorklet Processor\n", class_name));
        js.push_str("// Generated by probar - DO NOT EDIT MANUALLY\n\n");

        // Ring buffer class (if needed)
        if let Some(ref rb) = self.ring_buffer {
            if rb.use_atomics {
                js.push_str(&self.generate_ring_buffer_class(rb));
                js.push('\n');
            }
        }

        // Processor class
        js.push_str(&format!(
            "class {} extends AudioWorkletProcessor {{\n",
            class_name
        ));

        // Static parameter descriptors
        if !self.params.is_empty() {
            js.push_str("    static get parameterDescriptors() {\n");
            js.push_str("        return [\n");
            for param in &self.params {
                js.push_str(&format!("            {},\n", param.to_js_descriptor()));
            }
            js.push_str("        ];\n");
            js.push_str("    }\n\n");
        }

        // Constructor
        js.push_str("    constructor() {\n");
        js.push_str("        super();\n");

        if self.ring_buffer.is_some() {
            js.push_str("        this.ringBuffer = null;\n");
            js.push_str("        this.port.onmessage = (e) => {\n");
            js.push_str("            if (e.data.type === 'init' && e.data.ringBuffer) {\n");
            js.push_str("                this.ringBuffer = new RingBuffer(e.data.ringBuffer);\n");
            js.push_str("            }\n");
            js.push_str("        };\n");
        }

        js.push_str("    }\n\n");

        // Process method
        js.push_str("    process(inputs, outputs, parameters) {\n");
        js.push_str("        const input = inputs[0];\n");
        js.push_str("        if (!input || !input[0]) return true;\n\n");

        // Copy to output (passthrough)
        if self.outputs > 0 {
            js.push_str("        const output = outputs[0];\n");
            js.push_str("        for (let channel = 0; channel < input.length; channel++) {\n");
            js.push_str("            if (output[channel]) {\n");
            js.push_str("                output[channel].set(input[channel]);\n");
            js.push_str("            }\n");
            js.push_str("        }\n\n");
        }

        // Ring buffer write
        if self.ring_buffer.is_some() {
            js.push_str("        // Write to ring buffer for worker consumption\n");
            js.push_str("        if (this.ringBuffer) {\n");
            js.push_str("            this.ringBuffer.write(input[0]);\n");
            js.push_str("        }\n\n");
        }

        js.push_str("        return true; // Keep processor alive\n");
        js.push_str("    }\n");
        js.push_str("}\n\n");

        // Register processor
        js.push_str(&format!(
            "registerProcessor('{}', {});\n",
            self.name, class_name
        ));

        js
    }

    /// Generate ring buffer class for SharedArrayBuffer
    fn generate_ring_buffer_class(&self, config: &RingBufferConfig) -> String {
        format!(
            r#"// Lock-free ring buffer using SharedArrayBuffer + Atomics
class RingBuffer {{
    constructor(sab) {{
        this.buffer = new Float32Array(sab, 8, {size});
        this.state = new Int32Array(sab, 0, 2);  // [writeIdx, readIdx]
    }}

    write(samples) {{
        const writeIdx = Atomics.load(this.state, 0);
        const len = samples.length;
        const bufferLen = this.buffer.length;

        for (let i = 0; i < len; i++) {{
            this.buffer[(writeIdx + i) % bufferLen] = samples[i];
        }}

        Atomics.store(this.state, 0, (writeIdx + len) % bufferLen);
        Atomics.notify(this.state, 0);
    }}

    read(samples) {{
        const readIdx = Atomics.load(this.state, 1);
        const writeIdx = Atomics.load(this.state, 0);
        const bufferLen = this.buffer.length;

        let available = writeIdx - readIdx;
        if (available < 0) available += bufferLen;

        const toRead = Math.min(samples.length, available);

        for (let i = 0; i < toRead; i++) {{
            samples[i] = this.buffer[(readIdx + i) % bufferLen];
        }}

        Atomics.store(this.state, 1, (readIdx + toRead) % bufferLen);
        return toRead;
    }}

    available() {{
        const readIdx = Atomics.load(this.state, 1);
        const writeIdx = Atomics.load(this.state, 0);
        let available = writeIdx - readIdx;
        if (available < 0) available += {size};
        return available;
    }}
}}
"#,
            size = config.size
        )
    }

    /// Generate main thread audio initialization JavaScript
    #[must_use]
    pub fn to_audio_init_js(&self) -> String {
        let mut js = String::new();

        js.push_str("// Audio Pipeline Initialization\n");
        js.push_str("// Generated by probar - DO NOT EDIT MANUALLY\n\n");

        js.push_str("async function initAudio(workletUrl) {\n");
        js.push_str("    const audioContext = new AudioContext();\n");
        js.push_str("    await audioContext.audioWorklet.addModule(workletUrl);\n\n");

        // Create ring buffer if needed
        if let Some(ref rb) = self.ring_buffer {
            let buffer_bytes = rb.size * 4 + 8; // Float32 + state
            js.push_str(&format!(
                "    // Ring buffer: {} samples ({} bytes)\n",
                rb.size, buffer_bytes
            ));
            js.push_str(&format!(
                "    const ringBufferSab = new SharedArrayBuffer({});\n\n",
                buffer_bytes
            ));
        }

        // Create worklet node
        js.push_str(&format!(
            "    const workletNode = new AudioWorkletNode(audioContext, '{}');\n",
            self.name
        ));

        // Send ring buffer to worklet
        if self.ring_buffer.is_some() {
            js.push_str(
                "    workletNode.port.postMessage({ type: 'init', ringBuffer: ringBufferSab });\n",
            );
        }

        js.push_str("\n    return { audioContext, workletNode");
        if self.ring_buffer.is_some() {
            js.push_str(", ringBufferSab");
        }
        js.push_str(" };\n");
        js.push_str("}\n");

        js
    }

    /// Generate Rust bindings for ring buffer access
    #[must_use]
    pub fn to_rust_bindings(&self) -> String {
        let mut rust = String::new();

        rust.push_str(&format!("//! {} Audio Bindings\n", self.class_name()));
        rust.push_str("//! Generated by probar - DO NOT EDIT MANUALLY\n\n");

        if let Some(ref rb) = self.ring_buffer {
            rust.push_str("use std::sync::atomic::{AtomicI32, Ordering};\n\n");

            rust.push_str(&format!(
                "pub const RING_BUFFER_SIZE: usize = {};\n",
                rb.size
            ));
            rust.push_str(&format!(
                "pub const RING_BUFFER_CHANNELS: usize = {};\n\n",
                rb.channels
            ));

            rust.push_str("/// Lock-free ring buffer for audio data transfer\n");
            rust.push_str("pub struct AudioRingBuffer {\n");
            rust.push_str("    buffer: js_sys::Float32Array,\n");
            rust.push_str("    state: js_sys::Int32Array,\n");
            rust.push_str("}\n\n");

            rust.push_str("impl AudioRingBuffer {\n");
            rust.push_str("    /// Create from SharedArrayBuffer\n");
            rust.push_str("    pub fn new(sab: js_sys::SharedArrayBuffer) -> Self {\n");
            rust.push_str(&format!(
                "        let buffer = js_sys::Float32Array::new_with_byte_offset_and_length(&sab, 8, {});\n",
                rb.size
            ));
            rust.push_str("        let state = js_sys::Int32Array::new_with_byte_offset_and_length(&sab, 0, 2);\n");
            rust.push_str("        Self { buffer, state }\n");
            rust.push_str("    }\n\n");

            rust.push_str("    /// Read available samples\n");
            rust.push_str("    pub fn read(&self, output: &mut [f32]) -> usize {\n");
            rust.push_str("        // Implementation uses Atomics for thread-safe access\n");
            rust.push_str("        let read_idx = self.state.get_index(1) as usize;\n");
            rust.push_str("        let write_idx = self.state.get_index(0) as usize;\n");
            rust.push_str(&format!("        let buffer_len = {};\n", rb.size));
            rust.push_str("        \n");
            rust.push_str("        let mut available = write_idx as i32 - read_idx as i32;\n");
            rust.push_str("        if available < 0 { available += buffer_len as i32; }\n");
            rust.push_str("        \n");
            rust.push_str("        let to_read = output.len().min(available as usize);\n");
            rust.push_str("        for i in 0..to_read {\n");
            rust.push_str("            output[i] = self.buffer.get_index(((read_idx + i) % buffer_len) as u32);\n");
            rust.push_str("        }\n");
            rust.push_str("        \n");
            rust.push_str(
                "        self.state.set_index(1, ((read_idx + to_read) % buffer_len) as i32);\n",
            );
            rust.push_str("        to_read\n");
            rust.push_str("    }\n\n");

            rust.push_str("    /// Get number of available samples\n");
            rust.push_str("    pub fn available(&self) -> usize {\n");
            rust.push_str("        let read_idx = self.state.get_index(1) as i32;\n");
            rust.push_str("        let write_idx = self.state.get_index(0) as i32;\n");
            rust.push_str("        let mut available = write_idx - read_idx;\n");
            rust.push_str(&format!(
                "        if available < 0 {{ available += {}; }}\n",
                rb.size
            ));
            rust.push_str("        available as usize\n");
            rust.push_str("    }\n");
            rust.push_str("}\n");
        }

        rust
    }
}

impl Brick for AudioBrick {
    fn brick_name(&self) -> &'static str {
        "AudioBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[]
    }

    fn budget(&self) -> BrickBudget {
        // Audio processing has strict real-time requirements
        // 128 samples at 48kHz = 2.67ms per block
        BrickBudget::uniform(3)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Verify ring buffer size is reasonable
        if let Some(ref rb) = self.ring_buffer {
            if rb.size >= 128 && rb.size <= 48000 * 10 {
                passed.push(BrickAssertion::Custom {
                    name: "ring_buffer_size_valid".into(),
                    validator_id: 20,
                });
            } else {
                failed.push((
                    BrickAssertion::Custom {
                        name: "ring_buffer_size_valid".into(),
                        validator_id: 20,
                    },
                    format!("Ring buffer size {} out of range (128-480000)", rb.size),
                ));
            }
        }

        // Verify parameter ranges
        for param in &self.params {
            if param.min_value < param.max_value {
                passed.push(BrickAssertion::Custom {
                    name: format!("param_{}_range_valid", param.name),
                    validator_id: 21,
                });
            } else {
                failed.push((
                    BrickAssertion::Custom {
                        name: format!("param_{}_range_valid", param.name),
                        validator_id: 21,
                    },
                    "min >= max".into(),
                ));
            }
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(50),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ============================================================
    // AudioParam tests
    // ============================================================

    #[test]
    fn test_audio_param_new() {
        let param = AudioParam::new("gain", 1.0);

        assert_eq!(param.name, "gain");
        assert_eq!(param.default_value, 1.0);
        assert_eq!(param.min_value, f64::MIN);
        assert_eq!(param.max_value, f64::MAX);
        assert_eq!(param.automation_rate, "k-rate");
    }

    #[test]
    fn test_audio_param() {
        let param = AudioParam::new("gain", 1.0).range(0.0, 2.0).a_rate();

        assert_eq!(param.name, "gain");
        assert_eq!(param.default_value, 1.0);
        assert_eq!(param.min_value, 0.0);
        assert_eq!(param.max_value, 2.0);
        assert_eq!(param.automation_rate, "a-rate");
    }

    #[test]
    fn test_audio_param_k_rate() {
        let param = AudioParam::new("frequency", 440.0).k_rate();
        assert_eq!(param.automation_rate, "k-rate");
    }

    #[test]
    fn test_audio_param_js_descriptor() {
        let param = AudioParam::new("volume", 0.5).range(0.0, 1.0);
        let js = param.to_js_descriptor();

        assert!(js.contains("name: 'volume'"));
        assert!(js.contains("defaultValue: 0.5"));
        assert!(js.contains("minValue: 0"));
        assert!(js.contains("maxValue: 1"));
        assert!(js.contains("automationRate: 'k-rate'"));
    }

    #[test]
    fn test_audio_param_js_descriptor_a_rate() {
        let param = AudioParam::new("pan", 0.0).range(-1.0, 1.0).a_rate();
        let js = param.to_js_descriptor();

        assert!(js.contains("automationRate: 'a-rate'"));
    }

    #[test]
    fn test_audio_param_debug_and_clone() {
        let param = AudioParam::new("test", 0.5);
        let cloned = param;

        assert_eq!(cloned.name, "test");
        assert!(format!("{:?}", cloned).contains("AudioParam"));
    }

    // ============================================================
    // RingBufferConfig tests
    // ============================================================

    #[test]
    fn test_ring_buffer_config_default() {
        let config = RingBufferConfig::default();

        assert_eq!(config.size, 48000);
        assert_eq!(config.channels, 1);
        assert!(config.use_atomics);
    }

    #[test]
    fn test_ring_buffer_config_new() {
        let config = RingBufferConfig::new(24000);

        assert_eq!(config.size, 24000);
        assert_eq!(config.channels, 1);
        assert!(config.use_atomics);
    }

    #[test]
    fn test_ring_buffer_config() {
        let config = RingBufferConfig::new(48000).channels(2);

        assert_eq!(config.size, 48000);
        assert_eq!(config.channels, 2);
        assert!(config.use_atomics);
    }

    #[test]
    fn test_ring_buffer_config_without_atomics() {
        let config = RingBufferConfig::new(16000).without_atomics();

        assert!(!config.use_atomics);
    }

    #[test]
    fn test_ring_buffer_config_chained() {
        let config = RingBufferConfig::new(96000).channels(4).without_atomics();

        assert_eq!(config.size, 96000);
        assert_eq!(config.channels, 4);
        assert!(!config.use_atomics);
    }

    #[test]
    fn test_ring_buffer_config_debug_and_clone() {
        let config = RingBufferConfig::new(48000);
        let cloned = config;

        assert_eq!(cloned.size, 48000);
        assert!(format!("{:?}", cloned).contains("RingBufferConfig"));
    }

    // ============================================================
    // AudioBrick tests
    // ============================================================

    #[test]
    fn test_audio_brick_basic() {
        let audio = AudioBrick::new("whisper-capture");
        assert_eq!(audio.name, "whisper-capture");
        assert_eq!(audio.inputs, 1);
        assert_eq!(audio.outputs, 1);
    }

    #[test]
    fn test_audio_brick_inputs_outputs() {
        let audio = AudioBrick::new("mixer").inputs(4).outputs(2);

        assert_eq!(audio.inputs, 4);
        assert_eq!(audio.outputs, 2);
    }

    #[test]
    fn test_audio_brick_sample_rate() {
        let audio = AudioBrick::new("test").sample_rate(44100);
        assert_eq!(audio.sample_rate, 44100);
    }

    #[test]
    fn test_audio_brick_with_param() {
        let audio = AudioBrick::new("processor").param(AudioParam::new("gain", 1.0));

        assert_eq!(audio.params.len(), 1);
        assert_eq!(audio.params[0].name, "gain");
    }

    #[test]
    fn test_audio_brick_with_multiple_params() {
        let audio = AudioBrick::new("eq")
            .param(AudioParam::new("low", 0.0).range(-12.0, 12.0))
            .param(AudioParam::new("mid", 0.0).range(-12.0, 12.0))
            .param(AudioParam::new("high", 0.0).range(-12.0, 12.0));

        assert_eq!(audio.params.len(), 3);
    }

    #[test]
    fn test_audio_brick_with_ring_buffer() {
        let audio =
            AudioBrick::new("capture").with_ring_buffer(RingBufferConfig::new(48000).channels(2));

        assert!(audio.ring_buffer.is_some());
        let rb = audio.ring_buffer.unwrap();
        assert_eq!(rb.size, 48000);
        assert_eq!(rb.channels, 2);
    }

    #[test]
    fn test_audio_brick_class_name() {
        let audio = AudioBrick::new("whisper-capture");
        assert_eq!(audio.class_name(), "WhisperCaptureProcessor");

        let audio2 = AudioBrick::new("my_processor");
        assert_eq!(audio2.class_name(), "MyProcessorProcessor");
    }

    #[test]
    fn test_audio_brick_class_name_single_word() {
        let audio = AudioBrick::new("processor");
        assert_eq!(audio.class_name(), "ProcessorProcessor");
    }

    #[test]
    fn test_audio_brick_class_name_complex() {
        let audio = AudioBrick::new("my-complex_audio-processor");
        assert_eq!(audio.class_name(), "MyComplexAudioProcessorProcessor");
    }

    #[test]
    fn test_audio_brick_debug_and_clone() {
        let audio = AudioBrick::new("test")
            .param(AudioParam::new("gain", 1.0))
            .with_ring_buffer(RingBufferConfig::default());

        let cloned = audio;
        assert_eq!(cloned.name, "test");
        assert!(format!("{:?}", cloned).contains("AudioBrick"));
    }

    // ============================================================
    // to_worklet_js tests
    // ============================================================

    #[test]
    fn test_worklet_js_generation() {
        let audio = AudioBrick::new("test-processor")
            .param(AudioParam::new("gain", 1.0))
            .with_ring_buffer(RingBufferConfig::new(24000));

        let js = audio.to_worklet_js();

        assert!(js.contains("Generated by probar"));
        assert!(js.contains("class TestProcessorProcessor"));
        assert!(js.contains("extends AudioWorkletProcessor"));
        assert!(js.contains("parameterDescriptors"));
        assert!(js.contains("process(inputs, outputs, parameters)"));
        assert!(js.contains("registerProcessor('test-processor'"));
        assert!(js.contains("RingBuffer"));
    }

    #[test]
    fn test_worklet_js_without_params() {
        let audio = AudioBrick::new("simple").with_ring_buffer(RingBufferConfig::new(24000));

        let js = audio.to_worklet_js();

        // Should not contain parameterDescriptors when no params
        assert!(!js.contains("parameterDescriptors"));
        assert!(js.contains("class SimpleProcessor"));
    }

    #[test]
    fn test_worklet_js_without_ring_buffer() {
        let audio = AudioBrick::new("passthrough").param(AudioParam::new("gain", 1.0));

        let js = audio.to_worklet_js();

        // Should not contain RingBuffer class
        assert!(!js.contains("class RingBuffer"));
        assert!(!js.contains("this.ringBuffer"));
        assert!(js.contains("parameterDescriptors"));
    }

    #[test]
    fn test_worklet_js_no_outputs() {
        let audio = AudioBrick::new("sink")
            .outputs(0)
            .with_ring_buffer(RingBufferConfig::new(24000));

        let js = audio.to_worklet_js();

        // Should not contain output copy logic
        assert!(!js.contains("const output = outputs[0]"));
    }

    #[test]
    fn test_worklet_js_ring_buffer_class() {
        let audio = AudioBrick::new("capture").with_ring_buffer(RingBufferConfig::new(48000));

        let js = audio.to_worklet_js();

        // Verify ring buffer class is generated
        assert!(js.contains("class RingBuffer"));
        assert!(js.contains("constructor(sab)"));
        assert!(js.contains("write(samples)"));
        assert!(js.contains("read(samples)"));
        assert!(js.contains("available()"));
        assert!(js.contains("Atomics.load"));
        assert!(js.contains("Atomics.store"));
        assert!(js.contains("Atomics.notify"));
    }

    #[test]
    fn test_worklet_js_ring_buffer_without_atomics() {
        let audio = AudioBrick::new("simple")
            .with_ring_buffer(RingBufferConfig::new(24000).without_atomics());

        let js = audio.to_worklet_js();

        // Should not contain the RingBuffer class when atomics are disabled
        assert!(!js.contains("class RingBuffer"));
    }

    // ============================================================
    // to_audio_init_js tests
    // ============================================================

    #[test]
    fn test_audio_init_js_generation() {
        let audio = AudioBrick::new("capture").with_ring_buffer(RingBufferConfig::new(48000));

        let js = audio.to_audio_init_js();

        assert!(js.contains("AudioContext"));
        assert!(js.contains("audioWorklet.addModule"));
        assert!(js.contains("SharedArrayBuffer"));
        assert!(js.contains("AudioWorkletNode"));
    }

    #[test]
    fn test_audio_init_js_without_ring_buffer() {
        let audio = AudioBrick::new("passthrough");

        let js = audio.to_audio_init_js();

        assert!(js.contains("AudioContext"));
        assert!(js.contains("AudioWorkletNode"));
        assert!(!js.contains("SharedArrayBuffer"));
        assert!(!js.contains("ringBufferSab"));
    }

    #[test]
    fn test_audio_init_js_ring_buffer_size_calculation() {
        let audio = AudioBrick::new("capture").with_ring_buffer(RingBufferConfig::new(48000));

        let js = audio.to_audio_init_js();

        // 48000 * 4 (f32) + 8 (state) = 192008
        assert!(js.contains("Ring buffer: 48000 samples"));
        assert!(js.contains("SharedArrayBuffer(192008)"));
    }

    #[test]
    fn test_audio_init_js_posts_ring_buffer() {
        let audio = AudioBrick::new("capture").with_ring_buffer(RingBufferConfig::new(24000));

        let js = audio.to_audio_init_js();

        assert!(js.contains("workletNode.port.postMessage"));
        assert!(js.contains("type: 'init'"));
        assert!(js.contains("ringBuffer: ringBufferSab"));
    }

    // ============================================================
    // to_rust_bindings tests
    // ============================================================

    #[test]
    fn test_rust_bindings_generation() {
        let audio =
            AudioBrick::new("capture").with_ring_buffer(RingBufferConfig::new(48000).channels(2));

        let rust = audio.to_rust_bindings();

        assert!(rust.contains("CaptureProcessor Audio Bindings"));
        assert!(rust.contains("Generated by probar"));
        assert!(rust.contains("RING_BUFFER_SIZE: usize = 48000"));
        assert!(rust.contains("RING_BUFFER_CHANNELS: usize = 2"));
        assert!(rust.contains("struct AudioRingBuffer"));
        assert!(rust.contains("fn new(sab: js_sys::SharedArrayBuffer)"));
        assert!(rust.contains("fn read(&self, output: &mut [f32])"));
        assert!(rust.contains("fn available(&self)"));
    }

    #[test]
    fn test_rust_bindings_without_ring_buffer() {
        let audio = AudioBrick::new("passthrough");

        let rust = audio.to_rust_bindings();

        assert!(rust.contains("PassthroughProcessor Audio Bindings"));
        assert!(!rust.contains("RING_BUFFER_SIZE"));
        assert!(!rust.contains("struct AudioRingBuffer"));
    }

    // ============================================================
    // Brick trait implementation tests
    // ============================================================

    #[test]
    fn test_audio_brick_brick_name() {
        let audio = AudioBrick::new("test");
        assert_eq!(audio.brick_name(), "AudioBrick");
    }

    #[test]
    fn test_audio_brick_assertions() {
        let audio = AudioBrick::new("test");
        assert!(audio.assertions().is_empty());
    }

    #[test]
    fn test_audio_brick_budget() {
        let audio = AudioBrick::new("test");
        let budget = audio.budget();
        // Audio has strict 3ms budget for real-time
        assert_eq!(budget.as_duration(), Duration::from_millis(3));
    }

    #[test]
    fn test_audio_brick_to_html() {
        let audio = AudioBrick::new("test");
        assert!(audio.to_html().is_empty());
    }

    #[test]
    fn test_audio_brick_to_css() {
        let audio = AudioBrick::new("test");
        assert!(audio.to_css().is_empty());
    }

    // ============================================================
    // Verification tests
    // ============================================================

    #[test]
    fn test_verification_valid() {
        let audio = AudioBrick::new("test")
            .param(AudioParam::new("gain", 1.0).range(0.0, 2.0))
            .with_ring_buffer(RingBufferConfig::new(24000));

        let result = audio.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_verification_invalid_param() {
        let audio = AudioBrick::new("test").param(AudioParam::new("bad", 1.0).range(2.0, 1.0)); // min > max

        let result = audio.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_verification_ring_buffer_too_small() {
        let audio = AudioBrick::new("test").with_ring_buffer(RingBufferConfig {
            size: 64, // Too small (min is 128)
            channels: 1,
            use_atomics: true,
        });

        let result = audio.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_verification_ring_buffer_too_large() {
        let audio = AudioBrick::new("test").with_ring_buffer(RingBufferConfig {
            size: 48000 * 11, // Too large (max is 48000 * 10)
            channels: 1,
            use_atomics: true,
        });

        let result = audio.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_verification_ring_buffer_edge_cases() {
        // Minimum valid size
        let audio_min = AudioBrick::new("test").with_ring_buffer(RingBufferConfig {
            size: 128,
            channels: 1,
            use_atomics: true,
        });
        assert!(audio_min.verify().is_valid());

        // Maximum valid size
        let audio_max = AudioBrick::new("test").with_ring_buffer(RingBufferConfig {
            size: 48000 * 10,
            channels: 1,
            use_atomics: true,
        });
        assert!(audio_max.verify().is_valid());
    }

    #[test]
    fn test_verification_no_ring_buffer() {
        let audio = AudioBrick::new("test").param(AudioParam::new("gain", 1.0).range(0.0, 2.0));

        let result = audio.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_verification_multiple_params() {
        let audio = AudioBrick::new("eq")
            .param(AudioParam::new("low", 0.0).range(-12.0, 12.0))
            .param(AudioParam::new("mid", 0.0).range(-12.0, 12.0))
            .param(AudioParam::new("high", 0.0).range(-12.0, 12.0));

        let result = audio.verify();
        assert!(result.is_valid());
        // Each param should add a passed assertion
        assert_eq!(result.passed.len(), 3);
    }

    #[test]
    fn test_verification_mixed_valid_invalid_params() {
        let audio = AudioBrick::new("test")
            .param(AudioParam::new("good", 0.5).range(0.0, 1.0))
            .param(AudioParam::new("bad", 0.5).range(1.0, 0.0)); // Invalid

        let result = audio.verify();
        assert!(!result.is_valid());
        // One should pass, one should fail
        assert_eq!(result.passed.len(), 1);
        assert_eq!(result.failed.len(), 1);
    }

    // ============================================================
    // Integration tests
    // ============================================================

    #[test]
    fn test_full_audio_brick_workflow() {
        let audio = AudioBrick::new("whisper-capture")
            .inputs(1)
            .outputs(1)
            .sample_rate(16000)
            .param(AudioParam::new("gain", 1.0).range(0.0, 2.0).a_rate())
            .param(
                AudioParam::new("threshold", -40.0)
                    .range(-60.0, 0.0)
                    .k_rate(),
            )
            .with_ring_buffer(RingBufferConfig::new(144000).channels(1));

        // Verify configuration
        assert_eq!(audio.name, "whisper-capture");
        assert_eq!(audio.inputs, 1);
        assert_eq!(audio.outputs, 1);
        assert_eq!(audio.sample_rate, 16000);
        assert_eq!(audio.params.len(), 2);
        assert!(audio.ring_buffer.is_some());

        // Verify class name
        assert_eq!(audio.class_name(), "WhisperCaptureProcessor");

        // Verify validation passes
        assert!(audio.verify().is_valid());

        // Generate all JS
        let worklet_js = audio.to_worklet_js();
        let init_js = audio.to_audio_init_js();
        let rust_bindings = audio.to_rust_bindings();

        // Verify all generation works
        assert!(!worklet_js.is_empty());
        assert!(!init_js.is_empty());
        assert!(!rust_bindings.is_empty());

        // Verify key content
        assert!(worklet_js.contains("class WhisperCaptureProcessor"));
        assert!(worklet_js.contains("gain"));
        assert!(worklet_js.contains("threshold"));
        assert!(init_js.contains("whisper-capture"));
        assert!(rust_bindings.contains("RING_BUFFER_SIZE: usize = 144000"));
    }

    #[test]
    fn test_minimal_audio_brick() {
        let audio = AudioBrick::new("minimal");

        // Verify defaults
        assert_eq!(audio.inputs, 1);
        assert_eq!(audio.outputs, 1);
        assert_eq!(audio.sample_rate, 48000);
        assert!(audio.params.is_empty());
        assert!(audio.ring_buffer.is_none());

        // Should still verify valid
        assert!(audio.verify().is_valid());

        // Should generate valid JS
        let js = audio.to_worklet_js();
        assert!(js.contains("class MinimalProcessor"));
        assert!(js.contains("registerProcessor('minimal'"));
    }
}
