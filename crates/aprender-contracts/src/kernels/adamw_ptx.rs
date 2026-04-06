/// PTX assembly for the AdamW optimizer kernel.
///
/// Elementwise: 1 thread per parameter. Each thread updates one (param, m, v) triple.
pub fn adamw_step_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// AdamW kernel: 1 thread per parameter.
// Params: params_ptr, grads_ptr, m_ptr, v_ptr, lr, beta1, beta2, eps, wd, bc1, bc2, n
.visible .entry adamw_kernel(
    .param .u64 params_ptr,
    .param .u64 grads_ptr,
    .param .u64 m_ptr,
    .param .u64 v_ptr,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 wd,
    .param .f32 bc1,
    .param .f32 bc2,
    .param .u32 n
)
{
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %p_base, %g_base, %m_base, %v_base, %off;
    .reg .f32 %p, %g, %mi, %vi, %lr, %b1, %b2, %eps, %wd, %bc1, %bc2;
    .reg .f32 %one_minus_b1, %one_minus_b2, %g_sq;
    .reg .f32 %new_m, %new_v, %m_hat, %v_hat, %sqrt_v, %denom;
    .reg .f32 %adaptive, %decay, %update, %delta, %new_p;
    .reg .f32 %k_one;
    .reg .pred %pred;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %pred, %idx, %n;
    @%pred bra DONE;

    ld.param.u64 %p_base, [params_ptr];
    ld.param.u64 %g_base, [grads_ptr];
    ld.param.u64 %m_base, [m_ptr];
    ld.param.u64 %v_base, [v_ptr];
    ld.param.f32 %lr, [lr];
    ld.param.f32 %b1, [beta1];
    ld.param.f32 %b2, [beta2];
    ld.param.f32 %eps, [eps];
    ld.param.f32 %wd, [wd];
    ld.param.f32 %bc1, [bc1];
    ld.param.f32 %bc2, [bc2];

    mov.f32 %k_one, 0f3F800000;
    sub.f32 %one_minus_b1, %k_one, %b1;
    sub.f32 %one_minus_b2, %k_one, %b2;

    mul.wide.u32 %off, %idx, 4;

    // Load param, grad, m, v
    add.u64 %p_base, %p_base, %off;
    add.u64 %g_base, %g_base, %off;
    add.u64 %m_base, %m_base, %off;
    add.u64 %v_base, %v_base, %off;

    ld.global.f32 %p, [%p_base];
    ld.global.f32 %g, [%g_base];
    ld.global.f32 %mi, [%m_base];
    ld.global.f32 %vi, [%v_base];

    // m = beta1*m + (1-beta1)*g
    mul.f32 %new_m, %b1, %mi;
    fma.rn.f32 %new_m, %one_minus_b1, %g, %new_m;

    // v = beta2*v + (1-beta2)*g^2
    mul.f32 %g_sq, %g, %g;
    mul.f32 %new_v, %b2, %vi;
    fma.rn.f32 %new_v, %one_minus_b2, %g_sq, %new_v;

    // Store updated m, v
    st.global.f32 [%m_base], %new_m;
    st.global.f32 [%v_base], %new_v;

    // Bias correction
    mul.f32 %m_hat, %new_m, %bc1;
    mul.f32 %v_hat, %new_v, %bc2;

    // param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
    sqrt.approx.f32 %sqrt_v, %v_hat;
    add.f32 %denom, %sqrt_v, %eps;
    div.approx.f32 %adaptive, %m_hat, %denom;
    fma.rn.f32 %update, %wd, %p, %adaptive;
    mul.f32 %delta, %lr, %update;
    sub.f32 %new_p, %p, %delta;
    st.global.f32 [%p_base], %new_p;

DONE:
    ret;
}
"#
}
