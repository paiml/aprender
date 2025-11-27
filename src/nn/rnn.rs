//! Recurrent Neural Network layers.
//!
//! # Layers
//! - GRU: Gated Recurrent Unit (Cho et al., 2014)
//! - LSTM: Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)

use super::linear::Linear;
use super::module::Module;
use crate::autograd::Tensor;

/// Gated Recurrent Unit (GRU) layer.
///
/// ```text
/// r_t = σ(W_ir @ x_t + W_hr @ h_{t-1} + b_r)  // reset gate
/// z_t = σ(W_iz @ x_t + W_hz @ h_{t-1} + b_z)  // update gate
/// n_t = tanh(W_in @ x_t + r_t * (W_hn @ h_{t-1}) + b_n)  // candidate
/// h_t = (1 - z_t) * n_t + z_t * h_{t-1}  // hidden state
/// ```
pub struct GRU {
    input_size: usize,
    hidden_size: usize,
    // Gates: reset, update, new
    w_ir: Linear,
    w_hr: Linear,
    w_iz: Linear,
    w_hz: Linear,
    w_in: Linear,
    w_hn: Linear,
    training: bool,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            w_ir: Linear::new(input_size, hidden_size),
            w_hr: Linear::new(hidden_size, hidden_size),
            w_iz: Linear::new(input_size, hidden_size),
            w_hz: Linear::new(hidden_size, hidden_size),
            w_in: Linear::new(input_size, hidden_size),
            w_hn: Linear::new(hidden_size, hidden_size),
            training: true,
        }
    }

    /// Forward pass for single timestep.
    pub fn forward_step(&self, x: &Tensor, h: &Tensor) -> Tensor {
        // Reset gate
        let r = sigmoid_tensor(&add_tensors(&self.w_ir.forward(x), &self.w_hr.forward(h)));

        // Update gate
        let z = sigmoid_tensor(&add_tensors(&self.w_iz.forward(x), &self.w_hz.forward(h)));

        // Candidate hidden state
        let n = tanh_tensor(&add_tensors(
            &self.w_in.forward(x),
            &mul_tensors(&r, &self.w_hn.forward(h)),
        ));

        // New hidden state: (1-z)*n + z*h
        let one_minus_z = sub_from_one(&z);
        add_tensors(&mul_tensors(&one_minus_z, &n), &mul_tensors(&z, h))
    }

    /// Forward pass for sequence [batch, seq_len, input_size].
    pub fn forward_sequence(&self, x: &Tensor, h0: Option<&Tensor>) -> (Tensor, Tensor) {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut h = match h0 {
            Some(h) => h.clone(),
            None => Tensor::zeros(&[batch, self.hidden_size]),
        };

        let mut outputs = Vec::with_capacity(seq_len * batch * self.hidden_size);

        for t in 0..seq_len {
            let xt = slice_timestep(x, t);
            h = self.forward_step(&xt, &h);
            outputs.extend_from_slice(h.data());
        }

        let output = Tensor::new(&outputs, &[batch, seq_len, self.hidden_size]);
        (output, h)
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (output, _) = self.forward_sequence(input, None);
        output
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut p = self.w_ir.parameters();
        p.extend(self.w_hr.parameters());
        p.extend(self.w_iz.parameters());
        p.extend(self.w_hz.parameters());
        p.extend(self.w_in.parameters());
        p.extend(self.w_hn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut p = self.w_ir.parameters_mut();
        p.extend(self.w_hr.parameters_mut());
        p.extend(self.w_iz.parameters_mut());
        p.extend(self.w_hz.parameters_mut());
        p.extend(self.w_in.parameters_mut());
        p.extend(self.w_hn.parameters_mut());
        p
    }

    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for GRU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GRU")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .finish_non_exhaustive()
    }
}

// Helper functions
fn sigmoid_tensor(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    Tensor::new(&data, x.shape())
}

fn tanh_tensor(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|&v| v.tanh()).collect();
    Tensor::new(&data, x.shape())
}

fn add_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data())
        .map(|(&x, &y)| x + y)
        .collect();
    Tensor::new(&data, a.shape())
}

fn mul_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data())
        .map(|(&x, &y)| x * y)
        .collect();
    Tensor::new(&data, a.shape())
}

fn sub_from_one(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|&v| 1.0 - v).collect();
    Tensor::new(&data, x.shape())
}

fn slice_timestep(x: &Tensor, t: usize) -> Tensor {
    let batch = x.shape()[0];
    let input_size = x.shape()[2];
    let offset = t * input_size;

    let mut data = Vec::with_capacity(batch * input_size);
    for b in 0..batch {
        let start = b * x.shape()[1] * input_size + offset;
        data.extend_from_slice(&x.data()[start..start + input_size]);
    }
    Tensor::new(&data, &[batch, input_size])
}

/// Bidirectional RNN wrapper.
///
/// Processes sequence in both forward and backward directions,
/// concatenating outputs.
pub struct Bidirectional {
    forward_rnn: GRU,
    backward_rnn: GRU,
    input_size: usize,
    hidden_size: usize,
    training: bool,
}

impl Bidirectional {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            forward_rnn: GRU::new(input_size, hidden_size),
            backward_rnn: GRU::new(input_size, hidden_size),
            input_size,
            hidden_size,
            training: true,
        }
    }

    /// Forward pass returns concatenated [forward; backward] outputs.
    pub fn forward_sequence(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        // Forward pass
        let (fwd_out, fwd_h) = self.forward_rnn.forward_sequence(x, None);

        // Backward pass (reverse sequence)
        let x_rev = reverse_sequence(x);
        let (bwd_out_rev, bwd_h) = self.backward_rnn.forward_sequence(&x_rev, None);
        let bwd_out = reverse_sequence(&bwd_out_rev);

        // Concatenate outputs
        let output = concat_last_dim(&fwd_out, &bwd_out, batch, seq_len, self.hidden_size);

        (output, fwd_h, bwd_h)
    }

    pub fn output_size(&self) -> usize {
        self.hidden_size * 2
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for Bidirectional {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (output, _, _) = self.forward_sequence(input);
        output
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut p = self.forward_rnn.parameters();
        p.extend(self.backward_rnn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut p = self.forward_rnn.parameters_mut();
        p.extend(self.backward_rnn.parameters_mut());
        p
    }

    fn train(&mut self) {
        self.training = true;
        self.forward_rnn.train();
        self.backward_rnn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.forward_rnn.eval();
        self.backward_rnn.eval();
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for Bidirectional {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bidirectional")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .finish_non_exhaustive()
    }
}

fn reverse_sequence(x: &Tensor) -> Tensor {
    let (batch, seq_len, features) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let mut data = vec![0.0; batch * seq_len * features];

    for b in 0..batch {
        for t in 0..seq_len {
            let src = b * seq_len * features + t * features;
            let dst = b * seq_len * features + (seq_len - 1 - t) * features;
            data[dst..dst + features].copy_from_slice(&x.data()[src..src + features]);
        }
    }
    Tensor::new(&data, &[batch, seq_len, features])
}

fn concat_last_dim(a: &Tensor, b: &Tensor, batch: usize, seq_len: usize, hidden: usize) -> Tensor {
    let out_size = hidden * 2;
    let mut data = vec![0.0; batch * seq_len * out_size];

    for ba in 0..batch {
        for t in 0..seq_len {
            let dst = ba * seq_len * out_size + t * out_size;
            let src_a = ba * seq_len * hidden + t * hidden;
            let src_b = ba * seq_len * hidden + t * hidden;

            data[dst..dst + hidden].copy_from_slice(&a.data()[src_a..src_a + hidden]);
            data[dst + hidden..dst + out_size].copy_from_slice(&b.data()[src_b..src_b + hidden]);
        }
    }
    Tensor::new(&data, &[batch, seq_len, out_size])
}

/// Long Short-Term Memory (LSTM) layer.
///
/// Standard LSTM with forget, input, output gates and cell state.
///
/// ```text
/// f_t = σ(W_if @ x_t + W_hf @ h_{t-1} + b_f)  // forget gate
/// i_t = σ(W_ii @ x_t + W_hi @ h_{t-1} + b_i)  // input gate
/// g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)  // candidate cell
/// o_t = σ(W_io @ x_t + W_ho @ h_{t-1} + b_o)  // output gate
/// c_t = f_t * c_{t-1} + i_t * g_t  // cell state
/// h_t = o_t * tanh(c_t)  // hidden state
/// ```
///
/// # Reference
///
/// Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
pub struct LSTM {
    input_size: usize,
    hidden_size: usize,
    // Gates: forget, input, cell, output
    w_if: Linear,
    w_hf: Linear,
    w_ii: Linear,
    w_hi: Linear,
    w_ig: Linear,
    w_hg: Linear,
    w_io: Linear,
    w_ho: Linear,
    training: bool,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            w_if: Linear::new(input_size, hidden_size),
            w_hf: Linear::new(hidden_size, hidden_size),
            w_ii: Linear::new(input_size, hidden_size),
            w_hi: Linear::new(hidden_size, hidden_size),
            w_ig: Linear::new(input_size, hidden_size),
            w_hg: Linear::new(hidden_size, hidden_size),
            w_io: Linear::new(input_size, hidden_size),
            w_ho: Linear::new(hidden_size, hidden_size),
            training: true,
        }
    }

    /// Forward pass for single timestep.
    pub fn forward_step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> (Tensor, Tensor) {
        // Forget gate
        let f = sigmoid_tensor(&add_tensors(&self.w_if.forward(x), &self.w_hf.forward(h)));

        // Input gate
        let i = sigmoid_tensor(&add_tensors(&self.w_ii.forward(x), &self.w_hi.forward(h)));

        // Candidate cell
        let g = tanh_tensor(&add_tensors(&self.w_ig.forward(x), &self.w_hg.forward(h)));

        // Output gate
        let o = sigmoid_tensor(&add_tensors(&self.w_io.forward(x), &self.w_ho.forward(h)));

        // New cell state: c_t = f * c_{t-1} + i * g
        let c_new = add_tensors(&mul_tensors(&f, c), &mul_tensors(&i, &g));

        // New hidden state: h_t = o * tanh(c_t)
        let h_new = mul_tensors(&o, &tanh_tensor(&c_new));

        (h_new, c_new)
    }

    /// Forward pass for sequence [batch, seq_len, input_size].
    pub fn forward_sequence(
        &self,
        x: &Tensor,
        h0: Option<&Tensor>,
        c0: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor) {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut h = match h0 {
            Some(h) => h.clone(),
            None => Tensor::zeros(&[batch, self.hidden_size]),
        };

        let mut c = match c0 {
            Some(c) => c.clone(),
            None => Tensor::zeros(&[batch, self.hidden_size]),
        };

        let mut outputs = Vec::with_capacity(seq_len * batch * self.hidden_size);

        for t in 0..seq_len {
            let xt = slice_timestep(x, t);
            let (h_new, c_new) = self.forward_step(&xt, &h, &c);
            h = h_new;
            c = c_new;
            outputs.extend_from_slice(h.data());
        }

        let output = Tensor::new(&outputs, &[batch, seq_len, self.hidden_size]);
        (output, h, c)
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (output, _, _) = self.forward_sequence(input, None, None);
        output
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut p = self.w_if.parameters();
        p.extend(self.w_hf.parameters());
        p.extend(self.w_ii.parameters());
        p.extend(self.w_hi.parameters());
        p.extend(self.w_ig.parameters());
        p.extend(self.w_hg.parameters());
        p.extend(self.w_io.parameters());
        p.extend(self.w_ho.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut p = self.w_if.parameters_mut();
        p.extend(self.w_hf.parameters_mut());
        p.extend(self.w_ii.parameters_mut());
        p.extend(self.w_hi.parameters_mut());
        p.extend(self.w_ig.parameters_mut());
        p.extend(self.w_hg.parameters_mut());
        p.extend(self.w_io.parameters_mut());
        p.extend(self.w_ho.parameters_mut());
        p
    }

    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for LSTM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LSTM")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gru_creation() {
        let gru = GRU::new(10, 20);
        assert_eq!(gru.input_size(), 10);
        assert_eq!(gru.hidden_size(), 20);
    }

    #[test]
    fn test_gru_forward_step() {
        let gru = GRU::new(4, 8);
        let x = Tensor::ones(&[2, 4]);
        let h = Tensor::zeros(&[2, 8]);

        let h_new = gru.forward_step(&x, &h);
        assert_eq!(h_new.shape(), &[2, 8]);
    }

    #[test]
    fn test_gru_forward_sequence() {
        let gru = GRU::new(4, 8);
        let x = Tensor::ones(&[2, 5, 4]); // batch=2, seq=5, input=4

        let (output, h_final) = gru.forward_sequence(&x, None);
        assert_eq!(output.shape(), &[2, 5, 8]);
        assert_eq!(h_final.shape(), &[2, 8]);
    }

    #[test]
    fn test_gru_module_forward() {
        let gru = GRU::new(4, 8);
        let x = Tensor::ones(&[2, 5, 4]);

        let output = gru.forward(&x);
        assert_eq!(output.shape(), &[2, 5, 8]);
    }

    #[test]
    fn test_gru_parameters() {
        let gru = GRU::new(4, 8);
        let params = gru.parameters();
        // 6 linear layers * 2 (weight + bias) = 12
        assert_eq!(params.len(), 12);
    }

    #[test]
    fn test_gru_train_eval() {
        let mut gru = GRU::new(4, 8);
        assert!(gru.training());
        gru.eval();
        assert!(!gru.training());
    }

    #[test]
    fn test_gru_with_initial_hidden() {
        let gru = GRU::new(4, 8);
        let x = Tensor::ones(&[2, 3, 4]);
        let h0 = Tensor::ones(&[2, 8]);

        let (output, _) = gru.forward_sequence(&x, Some(&h0));
        assert_eq!(output.shape(), &[2, 3, 8]);
    }

    #[test]
    fn test_sigmoid_tensor() {
        let x = Tensor::new(&[0.0, 10.0, -10.0], &[3]);
        let y = sigmoid_tensor(&x);

        assert!((y.data()[0] - 0.5).abs() < 1e-5);
        assert!(y.data()[1] > 0.99);
        assert!(y.data()[2] < 0.01);
    }

    #[test]
    fn test_tanh_tensor() {
        let x = Tensor::new(&[0.0, 10.0, -10.0], &[3]);
        let y = tanh_tensor(&x);

        assert!((y.data()[0] - 0.0).abs() < 1e-5);
        assert!((y.data()[1] - 1.0).abs() < 1e-5);
        assert!((y.data()[2] + 1.0).abs() < 1e-5);
    }

    // Bidirectional tests

    #[test]
    fn test_bidirectional_creation() {
        let bi = Bidirectional::new(4, 8);
        assert_eq!(bi.hidden_size(), 8);
        assert_eq!(bi.output_size(), 16);
    }

    #[test]
    fn test_bidirectional_forward() {
        let bi = Bidirectional::new(4, 8);
        let x = Tensor::ones(&[2, 5, 4]); // batch=2, seq=5, input=4

        let output = bi.forward(&x);
        assert_eq!(output.shape(), &[2, 5, 16]); // hidden*2
    }

    #[test]
    fn test_bidirectional_forward_sequence() {
        let bi = Bidirectional::new(4, 8);
        let x = Tensor::ones(&[2, 5, 4]);

        let (output, fwd_h, bwd_h) = bi.forward_sequence(&x);
        assert_eq!(output.shape(), &[2, 5, 16]);
        assert_eq!(fwd_h.shape(), &[2, 8]);
        assert_eq!(bwd_h.shape(), &[2, 8]);
    }

    #[test]
    fn test_bidirectional_parameters() {
        let bi = Bidirectional::new(4, 8);
        let params = bi.parameters();
        // 2 GRUs * 12 params each = 24
        assert_eq!(params.len(), 24);
    }

    #[test]
    fn test_bidirectional_train_eval() {
        let mut bi = Bidirectional::new(4, 8);
        assert!(bi.training());
        bi.eval();
        assert!(!bi.training());
    }

    #[test]
    fn test_reverse_sequence() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]);
        let rev = reverse_sequence(&x);

        // [1,2], [3,4], [5,6] -> [5,6], [3,4], [1,2]
        assert_eq!(rev.data(), &[5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }

    // LSTM Tests

    #[test]
    fn test_lstm_creation() {
        let lstm = LSTM::new(10, 20);
        assert_eq!(lstm.input_size(), 10);
        assert_eq!(lstm.hidden_size(), 20);
    }

    #[test]
    fn test_lstm_forward_step() {
        let lstm = LSTM::new(4, 8);
        let x = Tensor::ones(&[2, 4]);
        let h = Tensor::zeros(&[2, 8]);
        let c = Tensor::zeros(&[2, 8]);

        let (h_new, c_new) = lstm.forward_step(&x, &h, &c);
        assert_eq!(h_new.shape(), &[2, 8]);
        assert_eq!(c_new.shape(), &[2, 8]);
    }

    #[test]
    fn test_lstm_forward_sequence() {
        let lstm = LSTM::new(4, 8);
        let x = Tensor::ones(&[2, 5, 4]); // batch=2, seq=5, input=4

        let (output, h_final, c_final) = lstm.forward_sequence(&x, None, None);
        assert_eq!(output.shape(), &[2, 5, 8]);
        assert_eq!(h_final.shape(), &[2, 8]);
        assert_eq!(c_final.shape(), &[2, 8]);
    }

    #[test]
    fn test_lstm_module_forward() {
        let lstm = LSTM::new(4, 8);
        let x = Tensor::ones(&[2, 5, 4]);

        let output = lstm.forward(&x);
        assert_eq!(output.shape(), &[2, 5, 8]);
    }

    #[test]
    fn test_lstm_parameters() {
        let lstm = LSTM::new(4, 8);
        let params = lstm.parameters();
        // 8 linear layers * 2 (weight + bias) = 16
        assert_eq!(params.len(), 16);
    }

    #[test]
    fn test_lstm_train_eval() {
        let mut lstm = LSTM::new(4, 8);
        assert!(lstm.training());
        lstm.eval();
        assert!(!lstm.training());
    }

    #[test]
    fn test_lstm_with_initial_state() {
        let lstm = LSTM::new(4, 8);
        let x = Tensor::ones(&[2, 3, 4]);
        let h0 = Tensor::ones(&[2, 8]);
        let c0 = Tensor::ones(&[2, 8]);

        let (output, _, _) = lstm.forward_sequence(&x, Some(&h0), Some(&c0));
        assert_eq!(output.shape(), &[2, 3, 8]);
    }

    #[test]
    fn test_lstm_cell_state_changes() {
        let lstm = LSTM::new(4, 8);
        let x = Tensor::ones(&[1, 4]);
        let h = Tensor::zeros(&[1, 8]);
        let c = Tensor::zeros(&[1, 8]);

        let (h_new, c_new) = lstm.forward_step(&x, &h, &c);

        // Cell state should have changed from initial zeros
        let c_sum: f32 = c_new.data().iter().sum();
        assert!(c_sum.abs() > 1e-6, "Cell state should change");

        // Hidden state bounded by tanh
        for &val in h_new.data() {
            assert!((-1.0..=1.0).contains(&val), "Hidden state bounded");
        }
    }
}
