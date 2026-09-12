//! The per-op parity table (L0-1b, #2971): pure aggregation over per-position
//! stage pairs, and the one question the instrument answers — which op, in
//! forward order, is the FIRST whose min cosine falls under the threshold.
//! No I/O and no realizar types, so it is tested in the workspace build.

/// One (stage, layer) row: aggregated over every position compared.
#[derive(Debug, Clone, PartialEq)]
pub struct OpRow {
    /// Canonical stage name (`attn_norm`, `ffn_swigl`, `lm_head`, …).
    pub stage: String,
    /// `None` for whole-model stages (`embedding`, `final_norm`, `lm_head`).
    pub layer: Option<u32>,
    /// Minimum cosine over positions.
    pub min_cosine: f32,
    /// Position at which the minimum cosine occurred.
    pub min_position: usize,
    /// Maximum |cpu - gpu| over positions and elements.
    pub max_abs: f32,
    /// Positions compared.
    pub positions: usize,
}

/// Forward order of the per-layer stages the two taps emit in common.
pub const LAYER_ORDER: &[&str] = &[
    "attn_norm",
    "qkv_matmul",
    "q_post_rope",
    "k_post_rope",
    "attention",
    "attn_out",
    "post_attn_residual",
    "ffn_norm",
    "ffn_up",
    "ffn_swigl",
    "ffn_out",
    "post_ffn_residual",
];

/// Sort key in forward order: embedding, then layer by layer in `LAYER_ORDER`,
/// then final_norm, lm_head. Unknown stages sort after the known ones of their layer.
#[must_use]
pub fn forward_key(stage: &str, layer: Option<u32>) -> (u32, u32, usize) {
    match (stage, layer) {
        ("embedding", _) => (0, 0, 0),
        ("final_norm", _) => (2, 0, 0),
        ("lm_head", _) => (2, 0, 1),
        (s, Some(l)) => (
            1,
            l,
            LAYER_ORDER
                .iter()
                .position(|k| *k == s)
                .unwrap_or(LAYER_ORDER.len()),
        ),
        (_, None) => (3, 0, 0),
    }
}

/// Cosine similarity; `0.0` when either side has zero norm or the lengths differ.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Maximum absolute elementwise difference (`inf` when the lengths differ).
#[must_use]
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Aggregate `(stage, layer, position, cpu, gpu)` samples into forward-ordered rows.
#[must_use]
pub fn aggregate(samples: &[(String, Option<u32>, usize, Vec<f32>, Vec<f32>)]) -> Vec<OpRow> {
    let mut rows: Vec<OpRow> = Vec::new();
    for (stage, layer, pos, cpu, gpu) in samples {
        let c = cosine(cpu, gpu);
        let m = max_abs_diff(cpu, gpu);
        match rows
            .iter_mut()
            .find(|r| r.stage == *stage && r.layer == *layer)
        {
            Some(r) => {
                if c < r.min_cosine {
                    r.min_cosine = c;
                    r.min_position = *pos;
                }
                r.max_abs = r.max_abs.max(m);
                r.positions += 1;
            }
            None => rows.push(OpRow {
                stage: stage.clone(),
                layer: *layer,
                min_cosine: c,
                min_position: *pos,
                max_abs: m,
                positions: 1,
            }),
        }
    }
    rows.sort_by_key(|r| forward_key(&r.stage, r.layer));
    rows
}

/// The FIRST row in forward order whose min cosine is under `threshold`.
#[must_use]
pub fn first_divergence(rows: &[OpRow], threshold: f32) -> Option<&OpRow> {
    rows.iter().find(|r| r.min_cosine < threshold)
}

/// One line per row, forward order, the diverging rows marked.
#[must_use]
pub fn render(rows: &[OpRow], threshold: f32) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<6} {:<20} {:>10} {:>6} {:>12} {:>5}  mark\n",
        "layer", "op", "min_cos", "@pos", "max_abs", "n"
    ));
    for r in rows {
        let layer = r.layer.map_or_else(|| "-".to_string(), |l| l.to_string());
        let mark = if r.min_cosine < threshold {
            "<-- RED"
        } else {
            ""
        };
        out.push_str(&format!(
            "{:<6} {:<20} {:>10.6} {:>6} {:>12.5} {:>5}  {}\n",
            layer, r.stage, r.min_cosine, r.min_position, r.max_abs, r.positions, mark
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(
        stage: &str,
        layer: Option<u32>,
        pos: usize,
        cpu: &[f32],
        gpu: &[f32],
    ) -> (String, Option<u32>, usize, Vec<f32>, Vec<f32>) {
        (stage.to_string(), layer, pos, cpu.to_vec(), gpu.to_vec())
    }

    #[test]
    fn cosine_and_max_abs_are_what_they_say() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(
            cosine(&[1.0], &[1.0, 2.0]),
            0.0,
            "length mismatch is 0, never a pass"
        );
        assert_eq!(max_abs_diff(&[1.0, 2.0], &[1.5, 2.0]), 0.5);
        assert!(max_abs_diff(&[1.0], &[1.0, 1.0]).is_infinite());
    }

    #[test]
    fn rows_come_out_in_forward_order_and_the_first_red_op_is_named() {
        // layer 3 ffn_swigl diverges at position 5; layer 0 attn_out is clean; lm_head diverges too
        // (downstream). The instrument must name ffn_swigl@3, not lm_head.
        let samples = vec![
            s("lm_head", None, 0, &[1.0, 2.0], &[2.0, -1.0]),
            s("ffn_swigl", Some(3), 5, &[1.0, 0.0], &[0.0, 1.0]),
            s("ffn_swigl", Some(3), 4, &[1.0, 0.0], &[1.0, 0.0]),
            s("attn_out", Some(0), 5, &[1.0, 1.0], &[1.0, 1.0]),
            s("embedding", None, 5, &[1.0, 1.0], &[1.0, 1.0]),
            s("post_attn_residual", Some(3), 5, &[1.0, 1.0], &[1.0, 1.0]),
        ];
        let rows = aggregate(&samples);
        let order: Vec<(String, Option<u32>)> =
            rows.iter().map(|r| (r.stage.clone(), r.layer)).collect();
        assert_eq!(
            order,
            vec![
                ("embedding".into(), None),
                ("attn_out".into(), Some(0)),
                ("post_attn_residual".into(), Some(3)),
                ("ffn_swigl".into(), Some(3)),
                ("lm_head".into(), None),
            ]
        );
        let first = first_divergence(&rows, 0.98).expect("something is red");
        assert_eq!(
            (
                first.stage.as_str(),
                first.layer,
                first.min_position,
                first.positions
            ),
            ("ffn_swigl", Some(3), 5, 2)
        );
        assert!(render(&rows, 0.98).contains("ffn_swigl"));
        assert_eq!(render(&rows, 0.98).matches("<-- RED").count(), 2);
    }

    #[test]
    fn a_clean_table_names_nothing() {
        let rows = aggregate(&[s("attn_norm", Some(0), 0, &[1.0, 2.0], &[1.0, 2.0])]);
        assert!(first_divergence(&rows, 0.98).is_none());
    }
}
