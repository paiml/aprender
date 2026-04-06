//! AWP-03: Dual-Waker Payload Backpressure

/// Dual-waker state for async backpressure.
///
/// Tracks two wakers: one for the producer, one for the consumer.
/// Enables efficient producer/consumer coordination.
#[derive(Debug, Default)]
pub struct DualWakerState {
    /// Producer is waiting
    producer_waiting: bool,
    /// Consumer is waiting
    consumer_waiting: bool,
    /// Buffer fill level (0-100%)
    fill_percent: u8,
    /// High watermark for backpressure (%)
    high_watermark: u8,
    /// Low watermark for resume (%)
    low_watermark: u8,
}

impl DualWakerState {
    /// Create new state with watermarks.
    pub fn new(low_watermark: u8, high_watermark: u8) -> Self {
        Self {
            producer_waiting: false,
            consumer_waiting: false,
            fill_percent: 0,
            high_watermark: high_watermark.min(100),
            low_watermark: low_watermark.min(high_watermark),
        }
    }

    /// Update fill level and determine who should wake.
    pub fn update_fill(&mut self, fill_percent: u8) -> WakeDecision {
        let old_fill = self.fill_percent;
        self.fill_percent = fill_percent.min(100);

        // Crossed high watermark going up - pause producer
        if old_fill < self.high_watermark && self.fill_percent >= self.high_watermark {
            return WakeDecision::PauseProducer;
        }

        // Crossed low watermark going down - resume producer
        if old_fill > self.low_watermark && self.fill_percent <= self.low_watermark {
            return WakeDecision::WakeProducer;
        }

        // Data available - wake consumer if waiting
        if self.fill_percent > 0 && self.consumer_waiting {
            return WakeDecision::WakeConsumer;
        }

        WakeDecision::None
    }

    /// Producer is now waiting.
    pub fn producer_wait(&mut self) {
        self.producer_waiting = true;
    }

    /// Consumer is now waiting.
    pub fn consumer_wait(&mut self) {
        self.consumer_waiting = true;
    }

    /// Producer was woken.
    pub fn producer_woke(&mut self) {
        self.producer_waiting = false;
    }

    /// Consumer was woken.
    pub fn consumer_woke(&mut self) {
        self.consumer_waiting = false;
    }

    /// Check if producer should be allowed to produce.
    #[must_use]
    pub fn can_produce(&self) -> bool {
        self.fill_percent < self.high_watermark
    }

    /// Check if consumer has data to consume.
    #[must_use]
    pub fn can_consume(&self) -> bool {
        self.fill_percent > 0
    }
}

/// Decision on which waker to invoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeDecision {
    /// No action needed
    None,
    /// Wake the producer (buffer drained below low watermark)
    WakeProducer,
    /// Wake the consumer (data available)
    WakeConsumer,
    /// Pause the producer (buffer above high watermark)
    PauseProducer,
}
