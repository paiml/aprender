//! wasm32 live microphone source: `getUserMedia` -> `AudioContext` -> `AnalyserNode`
//! (aprender#6891). Rust only, no hand-written JS. It decides nothing: the byte-to-float
//! mapping is `browser::audio_level::bytes_to_samples`.

use super::audio_level::bytes_to_samples;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{AnalyserNode, AudioContext, MediaStreamConstraints};

/// An open microphone with an analyser reading its time-domain samples.
pub struct AudioLevelSource {
    ctx: AudioContext,
    analyser: AnalyserNode,
    bytes: Vec<u8>,
}

impl AudioLevelSource {
    /// Ask for the microphone and wire it to an analyser with `fft_size` samples per read.
    ///
    /// # Errors
    /// The first browser call that fails: no `window`, no `mediaDevices`, permission
    /// refused, no device, or the audio graph cannot be built.
    pub async fn open(fft_size: u32) -> Result<Self, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
        let devices = window.navigator().media_devices()?;
        let constraints = MediaStreamConstraints::new();
        constraints.set_audio(&JsValue::TRUE);
        let stream = JsFuture::from(devices.get_user_media_with_constraints(&constraints)?)
            .await?
            .dyn_into::<web_sys::MediaStream>()?;
        let ctx = AudioContext::new()?;
        let analyser = ctx.create_analyser()?;
        analyser.set_fft_size(fft_size);
        ctx.create_media_stream_source(&stream)?
            .connect_with_audio_node(&analyser)?;
        let bytes = vec![0u8; analyser.fft_size() as usize];
        Ok(Self {
            ctx,
            analyser,
            bytes,
        })
    }

    /// Replace `out` with the latest samples in [-1, 1]; silence reads as 0.0.
    pub fn read_into(&mut self, out: &mut Vec<f32>) {
        self.analyser.get_byte_time_domain_data(&mut self.bytes);
        bytes_to_samples(&self.bytes, out);
    }

    /// Release the audio context.
    ///
    /// # Errors
    /// The browser refused to close the context.
    pub async fn close(self) -> Result<(), JsValue> {
        JsFuture::from(self.ctx.close()?).await.map(|_| ())
    }
}
