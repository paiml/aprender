//! PMAT-5279: the two stateful Gated `DeltaNet` decode kernels over m independent
//! slots in one launch.
//!
//! A batched decode step holds m requests, each with its own conv window and its own
//! recurrent state, in separate device allocations. The single-token kernels
//! ([`super::CausalConv1dSiluKernel`], [`super::DeltaRuleRecurrenceKernel`]) take one
//! state pointer, so the step launched each of them m times per layer. These twins add
//! a grid y dimension for the slot: slot `s` reads its rows `s * stride` into the
//! m-row activation buffers and its state pointer from entry `s` of a device table of
//! u64 pointers. Every arithmetic instruction, and its order, is the single-token
//! kernel's, so slot `s` is bitwise the single-token kernel run on slot `s` alone; only
//! the address arithmetic differs.

use crate::kernels::gdn::{emit_exp_f32, emit_silu_f32, ELEMENTWISE_BLOCK};
use crate::kernels::Kernel;
use crate::ptx::builder::{KernelBuilder, PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType, VirtualReg};

/// `base + slot * bytes`.
fn slot_base(
    ctx: &mut KernelBuilder<'_>,
    base: VirtualReg,
    slot: VirtualReg,
    bytes: u32,
) -> VirtualReg {
    let bytes_r = ctx.mov_u32_imm(bytes);
    let off = ctx.mul_wide_u32_reg(slot, bytes_r);
    ctx.add_u64(base, off)
}

/// Slot `slot`'s state pointer: entry `slot` of the u64 table at `table`.
fn slot_state(ctx: &mut KernelBuilder<'_>, table: VirtualReg, slot: VirtualReg) -> VirtualReg {
    let addr = slot_base(ctx, table, slot, 8);
    ctx.ld_global_u64(addr)
}

/// [`super::CausalConv1dSiluKernel`] over m slots: grid `(ceil(channels / 256), m)`.
///
/// Slot `s` reads `input[s * row_stride ..]`, writes `output[s * row_stride ..]` and
/// advances the window at `states[s]`.
#[derive(Debug, Clone, Copy)]
pub struct CausalConv1dSiluSlotsKernel {
    /// Number of convolution channels (`conv_dim`).
    pub channels: u32,
    /// Convolution kernel width (`conv_kernel`, 4 for Qwen3.5).
    pub kernel_size: u32,
    /// Elements between two slots' rows in `input` and `output`.
    pub row_stride: u32,
}

impl CausalConv1dSiluSlotsKernel {
    /// Create the kernel.
    ///
    /// # Panics
    /// If `kernel_size` is zero, or `row_stride` is narrower than `channels`.
    #[must_use]
    pub fn new(channels: u32, kernel_size: u32, row_stride: u32) -> Self {
        assert!(kernel_size >= 1, "kernel_size must be >= 1");
        assert!(
            row_stride >= channels,
            "a slot's row must hold every channel"
        );
        Self {
            channels,
            kernel_size,
            row_stride,
        }
    }

    /// Launch grid for `slots` slots.
    #[must_use]
    pub const fn grid(&self, slots: u32) -> (u32, u32, u32) {
        (self.channels.div_ceil(ELEMENTWISE_BLOCK), slots, 1)
    }

    /// Launch block (one thread per channel).
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (ELEMENTWISE_BLOCK, 1, 1)
    }
}

impl Kernel for CausalConv1dSiluSlotsKernel {
    fn name(&self) -> &str {
        "gdn_causal_conv1d_silu_slots"
    }

    fn build_ptx(&self) -> PtxKernel {
        let channels = self.channels;
        let k = self.kernel_size;
        let state_len = k - 1;
        let row_bytes = self.row_stride * 4;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "input_ptr") // [m][row_stride]
            .param(PtxType::U64, "states_ptr") // [m] u64 -> [channels][kernel_size - 1]
            .param(PtxType::U64, "weight_ptr") // [channels][kernel_size]
            .param(PtxType::U64, "output_ptr") // [m][row_stride]
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let cta = ctx.special_reg(PtxReg::CtaIdX);
                let slot = ctx.special_reg(PtxReg::CtaIdY);
                let block = ctx.mov_u32_imm(ELEMENTWISE_BLOCK);
                let chan = ctx.mad_lo_u32(cta, block, tid);

                let channels_r = ctx.mov_u32_imm(channels);
                let in_bounds = ctx.setp_lt_u32(chan, channels_r);
                ctx.branch_if_not(in_bounds, "gdn_conv_slots_exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let states_ptr = ctx.load_param_u64("states_ptr");
                let weight_ptr = ctx.load_param_u64("weight_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let input_row = slot_base(ctx, input_ptr, slot, row_bytes);
                let output_row = slot_base(ctx, output_ptr, slot, row_bytes);
                let state_ptr = slot_state(ctx, states_ptr, slot);

                let four = ctx.mov_u32_imm(4);
                let chan_bytes = ctx.mul_wide_u32_reg(chan, four);

                let in_addr = ctx.add_u64(input_row, chan_bytes);
                let x = ctx.ld_global_f32(in_addr);

                let s_stride = ctx.mov_u32_imm(state_len * 4);
                let s_off = ctx.mul_wide_u32_reg(chan, s_stride);
                let s_base = ctx.add_u64(state_ptr, s_off);
                let w_stride = ctx.mov_u32_imm(k * 4);
                let w_off = ctx.mul_wide_u32_reg(chan, w_stride);
                let w_base = ctx.add_u64(weight_ptr, w_off);

                // From here on, the single-token kernel's body: read the whole window,
                // sum in the CPU's order with separate multiply and add, shift, SiLU.
                let state_addrs: Vec<_> = (0..state_len)
                    .map(|i| {
                        let off = ctx.mov_u64_imm(u64::from(i * 4));
                        ctx.add_u64(s_base, off)
                    })
                    .collect();
                let state_vals: Vec<_> = state_addrs
                    .iter()
                    .map(|&addr| ctx.ld_global_f32(addr))
                    .collect();

                let sum = ctx.mov_f32_imm(0.0);
                for i in 0..state_len {
                    let off = ctx.mov_u64_imm(u64::from(i * 4));
                    let w_addr = ctx.add_u64(w_base, off);
                    let w_val = ctx.ld_global_f32(w_addr);
                    let prod = ctx.mul_f32(state_vals[i as usize], w_val);
                    ctx.add_f32_inplace(sum, prod);
                }
                let last_off = ctx.mov_u64_imm(u64::from((k - 1) * 4));
                let w_last_addr = ctx.add_u64(w_base, last_off);
                let w_last = ctx.ld_global_f32(w_last_addr);
                let prod_last = ctx.mul_f32(x, w_last);
                ctx.add_f32_inplace(sum, prod_last);

                for i in 0..state_len.saturating_sub(1) {
                    ctx.st_global_f32(state_addrs[i as usize], state_vals[i as usize + 1]);
                }
                if state_len >= 1 {
                    ctx.st_global_f32(state_addrs[state_len as usize - 1], x);
                }

                let activated = emit_silu_f32(ctx, sum);
                let out_addr = ctx.add_u64(output_row, chan_bytes);
                ctx.st_global_f32(out_addr, activated);

                ctx.label("gdn_conv_slots_exit");
                ctx.ret();
            })
    }
}

/// [`super::DeltaRuleRecurrenceKernel`] over m slots: grid `(num_v_heads, m)`.
///
/// Slot `s` reads `q`, `k` and `v` at `s * qkv_stride` past their slot-0 pointers,
/// `beta` and `gate` at `s * num_v_heads`, writes `output` at
/// `s * num_v_heads * head_v_dim`, and updates the state at `states[s]`.
#[derive(Debug, Clone, Copy)]
pub struct DeltaRuleSlotsKernel {
    /// Number of key/query heads.
    pub num_k_heads: u32,
    /// Key/query head width `Dk`.
    pub head_k_dim: u32,
    /// Number of value heads — one block each per slot.
    pub num_v_heads: u32,
    /// Value head width `Dv`.
    pub head_v_dim: u32,
    /// Elements between two slots' `q` (and `k`, and `v`) rows.
    pub qkv_stride: u32,
}

impl DeltaRuleSlotsKernel {
    /// Create the kernel; a `num_k_heads` of zero is clamped to one, as in the
    /// single-token kernel.
    #[must_use]
    pub const fn new(
        num_k_heads: u32,
        head_k_dim: u32,
        num_v_heads: u32,
        head_v_dim: u32,
        qkv_stride: u32,
    ) -> Self {
        Self {
            num_k_heads: if num_k_heads == 0 { 1 } else { num_k_heads },
            head_k_dim,
            num_v_heads,
            head_v_dim,
            qkv_stride,
        }
    }

    /// Launch grid for `slots` slots.
    #[must_use]
    pub const fn grid(&self, slots: u32) -> (u32, u32, u32) {
        (self.num_v_heads, slots, 1)
    }

    /// Launch block — one thread per state row.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (self.head_v_dim, 1, 1)
    }

    /// Whether a thread's state row is held in registers for the step: unrolled,
    /// read once and written once. The loops read the row twice and write it twice,
    /// each load waiting on the one before. Widths past [`MAX_REGISTER_ROW`] or not a
    /// multiple of 4 keep the loops.
    #[must_use]
    pub const fn row_in_registers(&self) -> bool {
        self.head_k_dim % 4 == 0 && self.head_k_dim <= MAX_REGISTER_ROW
    }
}

/// Widest state row [`DeltaRuleSlotsKernel`] holds in registers.
pub const MAX_REGISTER_ROW: u32 = 128;

/// `[row, k, q, v_j]` addresses for one thread's recurrence.
type RecurrenceRows = [VirtualReg; 4];

/// The single-token kernel's two ascending passes over row `j`, as its loops.
/// Returns `sum_i s_new[i] * q[i]`.
fn emit_recurrence_loops(
    ctx: &mut KernelBuilder<'_>,
    [s_row, k_base, q_base, v_addr]: RecurrenceRows,
    exp_gate: VirtualReg,
    beta: VirtualReg,
    dk_r: VirtualReg,
    four: VirtualReg,
) -> VirtualReg {
    // Steps 1 + 2: decay row j in place and dot it with k, i ascending.
    let sum = ctx.mov_f32_imm(0.0);
    let i = ctx.mov_u32_imm(0);
    ctx.label("gdn_dr_slots_decay_loop");
    let go = ctx.setp_lt_u32(i, dk_r);
    ctx.branch_if_not(go, "gdn_dr_slots_decay_end");
    let off = ctx.mul_wide_u32_reg(i, four);
    let s_addr = ctx.add_u64(s_row, off);
    let k_addr = ctx.add_u64(k_base, off);
    let s_val = ctx.ld_global_f32(s_addr);
    let s_scaled = ctx.mul_f32(s_val, exp_gate);
    ctx.st_global_f32(s_addr, s_scaled);
    let k_val = ctx.ld_global_f32(k_addr);
    let prod = ctx.mul_f32(s_scaled, k_val);
    ctx.add_f32_inplace(sum, prod);
    ctx.add_u32_inplace(i, 1);
    ctx.branch("gdn_dr_slots_decay_loop");
    ctx.label("gdn_dr_slots_decay_end");

    // delta[j] = (v[j] - sum) * beta
    let v_j = ctx.ld_global_f32(v_addr);
    let diff = ctx.sub_f32(v_j, sum);
    let delta = ctx.mul_f32(diff, beta);

    // Steps 3 + 4: update row j and dot it with q, i ascending.
    let out_sum = ctx.mov_f32_imm(0.0);
    let i2 = ctx.mov_u32_imm(0);
    ctx.label("gdn_dr_slots_update_loop");
    let go2 = ctx.setp_lt_u32(i2, dk_r);
    ctx.branch_if_not(go2, "gdn_dr_slots_update_end");
    let off2 = ctx.mul_wide_u32_reg(i2, four);
    let s_addr2 = ctx.add_u64(s_row, off2);
    let k_addr2 = ctx.add_u64(k_base, off2);
    let q_addr2 = ctx.add_u64(q_base, off2);
    let s_old = ctx.ld_global_f32(s_addr2);
    let k_val2 = ctx.ld_global_f32(k_addr2);
    let upd = ctx.mul_f32(k_val2, delta);
    let s_new = ctx.add_f32(s_old, upd);
    ctx.st_global_f32(s_addr2, s_new);
    let q_val = ctx.ld_global_f32(q_addr2);
    let prod2 = ctx.mul_f32(s_new, q_val);
    ctx.add_f32_inplace(out_sum, prod2);
    ctx.add_u32_inplace(i2, 1);
    ctx.branch("gdn_dr_slots_update_loop");
    ctx.label("gdn_dr_slots_update_end");
    out_sum
}

/// `dk` consecutive f32 from `base`, as `dk / 4` vector loads issued together.
fn load_row_v4(ctx: &mut KernelBuilder<'_>, base: VirtualReg, dk: u32) -> Vec<VirtualReg> {
    let mut row = Vec::with_capacity(dk as usize);
    for quad in 0..dk / 4 {
        let off = ctx.mov_u64_imm(u64::from(quad) * 16);
        let addr = ctx.add_u64(base, off);
        row.extend(ctx.ld_global_f32_v4(addr));
    }
    row
}

/// The same two passes unrolled, with row `j` read once into registers and written
/// once. Every float instruction and its order is the loops': the decayed row is
/// kept in registers instead of being stored and reloaded, which changes no value.
fn emit_recurrence_in_registers(
    ctx: &mut KernelBuilder<'_>,
    [s_row, k_base, q_base, v_addr]: RecurrenceRows,
    exp_gate: VirtualReg,
    beta: VirtualReg,
    dk: u32,
) -> VirtualReg {
    let s_old = load_row_v4(ctx, s_row, dk);
    let k = load_row_v4(ctx, k_base, dk);

    // Steps 1 + 2.
    let sum = ctx.mov_f32_imm(0.0);
    let mut s_scaled = Vec::with_capacity(dk as usize);
    for (&s_val, &k_val) in s_old.iter().zip(&k) {
        let scaled = ctx.mul_f32(s_val, exp_gate);
        let prod = ctx.mul_f32(scaled, k_val);
        ctx.add_f32_inplace(sum, prod);
        s_scaled.push(scaled);
    }

    let v_j = ctx.ld_global_f32(v_addr);
    let diff = ctx.sub_f32(v_j, sum);
    let delta = ctx.mul_f32(diff, beta);

    // Steps 3 + 4.
    let q = load_row_v4(ctx, q_base, dk);
    let out_sum = ctx.mov_f32_imm(0.0);
    let mut s_new = Vec::with_capacity(dk as usize);
    for ((&scaled, &k_val), &q_val) in s_scaled.iter().zip(&k).zip(&q) {
        let upd = ctx.mul_f32(k_val, delta);
        let new = ctx.add_f32(scaled, upd);
        let prod = ctx.mul_f32(new, q_val);
        ctx.add_f32_inplace(out_sum, prod);
        s_new.push(new);
    }
    for (pair, chunk) in s_new.chunks_exact(2).enumerate() {
        let off = ctx.mov_u64_imm(pair as u64 * 8);
        let addr = ctx.add_u64(s_row, off);
        ctx.st_global_f32_v2(addr, chunk[0], chunk[1]);
    }
    out_sum
}

impl Kernel for DeltaRuleSlotsKernel {
    fn name(&self) -> &str {
        "gdn_delta_rule_slots"
    }

    fn build_ptx(&self) -> PtxKernel {
        let dk = self.head_k_dim;
        let dv = self.head_v_dim;
        let nk = self.num_k_heads;
        let nv = self.num_v_heads;
        let scale = 1.0 / (dk as f32).sqrt();
        let qkv_bytes = self.qkv_stride * 4;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // slot 0's [num_k_heads * Dk]
            .param(PtxType::U64, "k_ptr") // slot 0's [num_k_heads * Dk]
            .param(PtxType::U64, "v_ptr") // slot 0's [num_v_heads * Dv]
            .param(PtxType::U64, "beta_ptr") // [m][num_v_heads]
            .param(PtxType::U64, "gate_ptr") // [m][num_v_heads] (dt)
            .param(PtxType::U64, "states_ptr") // [m] u64 -> [num_v_heads * Dv * Dk]
            .param(PtxType::U64, "output_ptr") // [m][num_v_heads * Dv]
            .shared_memory(0)
            .build(|ctx| {
                let j = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);
                let slot = ctx.special_reg(PtxReg::CtaIdY);

                let dv_r = ctx.mov_u32_imm(dv);
                let row_in_bounds = ctx.setp_lt_u32(j, dv_r);
                ctx.branch_if_not(row_in_bounds, "gdn_dr_slots_exit");

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let beta_ptr = ctx.load_param_u64("beta_ptr");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let states_ptr = ctx.load_param_u64("states_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let q_ptr = slot_base(ctx, q_ptr, slot, qkv_bytes);
                let k_ptr = slot_base(ctx, k_ptr, slot, qkv_bytes);
                let v_ptr = slot_base(ctx, v_ptr, slot, qkv_bytes);
                let beta_ptr = slot_base(ctx, beta_ptr, slot, nv * 4);
                let gate_ptr = slot_base(ctx, gate_ptr, slot, nv * 4);
                let output_ptr = slot_base(ctx, output_ptr, slot, nv * dv * 4);
                let state_ptr = slot_state(ctx, states_ptr, slot);

                let four = ctx.mov_u32_imm(4);
                let dk_r = ctx.mov_u32_imm(dk);
                let dk_bytes = ctx.mov_u32_imm(dk * 4);
                let dv_bytes = ctx.mov_u32_imm(dv * 4);

                let kh = if nk == nv { h } else { ctx.rem_u32(h, nk) };
                let qk_head_off = ctx.mul_wide_u32_reg(kh, dk_bytes);
                let q_base = ctx.add_u64(q_ptr, qk_head_off);
                let k_base = ctx.add_u64(k_ptr, qk_head_off);

                let v_head_off = ctx.mul_wide_u32_reg(h, dv_bytes);
                let v_base = ctx.add_u64(v_ptr, v_head_off);
                let out_base = ctx.add_u64(output_ptr, v_head_off);

                let state_head_bytes = ctx.mov_u32_imm(dv * dk * 4);
                let state_head_off = ctx.mul_wide_u32_reg(h, state_head_bytes);
                let state_head = ctx.add_u64(state_ptr, state_head_off);
                let row_off = ctx.mul_wide_u32_reg(j, dk_bytes);
                let s_row = ctx.add_u64(state_head, row_off);

                let scalar_off = ctx.mul_wide_u32_reg(h, four);
                let beta_addr = ctx.add_u64(beta_ptr, scalar_off);
                let gate_addr = ctx.add_u64(gate_ptr, scalar_off);
                let beta = ctx.ld_global_f32(beta_addr);
                let gate = ctx.ld_global_f32(gate_addr);
                let exp_gate = emit_exp_f32(ctx, gate);

                // From here on, the single-token kernel's two ascending passes.
                let v_off = ctx.mul_wide_u32_reg(j, four);
                let v_addr = ctx.add_u64(v_base, v_off);
                let rows = [s_row, k_base, q_base, v_addr];
                let out_sum = if self.row_in_registers() {
                    emit_recurrence_in_registers(ctx, rows, exp_gate, beta, dk)
                } else {
                    emit_recurrence_loops(ctx, rows, exp_gate, beta, dk_r, four)
                };

                let scale_r = ctx.mov_f32_imm(scale);
                let result = ctx.mul_f32(out_sum, scale_r);
                let out_addr = ctx.add_u64(out_base, v_off);
                ctx.st_global_f32(out_addr, result);

                ctx.label("gdn_dr_slots_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;
    use crate::kernels::gdn::{CausalConv1dSiluKernel, DeltaRuleRecurrenceKernel};

    /// The arithmetic of a PTX listing: every float instruction in order.
    fn float_ops(ptx: &str) -> Vec<String> {
        ptx.lines()
            .map(str::trim)
            .filter(|l| l.contains(".f32") && !l.starts_with("ld.") && !l.starts_with("st."))
            .map(|l| l.split_whitespace().next().unwrap_or("").to_string())
            .collect()
    }

    #[test]
    fn conv_slots_has_the_single_kernels_arithmetic() {
        let one = CausalConv1dSiluKernel::new(8192, 4).emit_ptx();
        let kern = CausalConv1dSiluSlotsKernel::new(8192, 4, 8192);
        let slots = kern.emit_ptx();
        assert!(slots.contains("%ctaid.y"), "{slots}");
        assert!(slots.contains("ld.global.u64"), "{slots}");
        assert_eq!(float_ops(&one), float_ops(&slots));
        assert_eq!(kern.grid(5), (32, 5, 1));
    }

    /// `ptx` with the lines between each `(loop, end)` label pair repeated `n` times:
    /// the listing of the loops unrolled.
    fn unrolled(ptx: &str, loops: &[(&str, &str)], n: usize) -> String {
        let mut out = Vec::new();
        let mut body: Option<(Vec<&str>, &str)> = None;
        for line in ptx.lines() {
            let t = line.trim();
            if let Some((lines, end)) = body.as_mut() {
                if t.starts_with(*end) {
                    for _ in 0..n {
                        out.extend(lines.iter().copied());
                    }
                    body = None;
                } else {
                    lines.push(line);
                }
            } else if let Some(&(_, end)) =
                loops.iter().find(|(l, _)| t.starts_with(&format!("{l}:")))
            {
                body = Some((Vec::new(), end));
            } else {
                out.push(line);
            }
        }
        out.join("\n")
    }

    #[test]
    fn delta_slots_has_the_single_kernels_arithmetic() {
        let loops = [
            ("gdn_dr_decay_loop", "gdn_dr_decay_end"),
            ("gdn_dr_update_loop", "gdn_dr_update_end"),
        ];
        for (nk, nv) in [(16u32, 16u32), (16, 32)] {
            let one = DeltaRuleRecurrenceKernel::new(nk, 128, nv, 128).emit_ptx();
            let kern = DeltaRuleSlotsKernel::new(nk, 128, nv, 128, 8192);
            assert!(kern.row_in_registers());
            let slots = kern.emit_ptx();
            assert!(slots.contains("%ctaid.y"), "{slots}");
            assert!(!slots.contains("gdn_dr_slots_decay_loop"), "{slots}");
            assert_eq!(float_ops(&unrolled(&one, &loops, 128)), float_ops(&slots));
            assert_eq!(kern.grid(3), (nv, 3, 1));
            assert_eq!(kern.block(), (128, 1, 1));
        }
    }

    #[test]
    fn delta_slots_past_the_register_row_keeps_the_loops() {
        for dk in [MAX_REGISTER_ROW + 4, 126] {
            let one = DeltaRuleRecurrenceKernel::new(16, dk, 16, 128).emit_ptx();
            let kern = DeltaRuleSlotsKernel::new(16, dk, 16, 128, 8192);
            assert!(!kern.row_in_registers());
            let slots = kern.emit_ptx();
            assert!(slots.contains("gdn_dr_slots_decay_loop"), "{slots}");
            assert_eq!(float_ops(&one), float_ops(&slots));
        }
    }

    #[test]
    #[should_panic(expected = "every channel")]
    fn conv_slots_refuses_a_stride_narrower_than_the_channels() {
        let _ = CausalConv1dSiluSlotsKernel::new(8, 4, 7);
    }
}
