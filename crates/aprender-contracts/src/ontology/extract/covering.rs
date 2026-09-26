//! aprender#3745 S2 (#3777) — a deterministic strength-2 covering array (IPOG, in-parameter-order).
//!
//! Every pair of levels of every two factors appears together in at least one row — except the FORBIDDEN pairs
//! (two args the surface declares `conflicts_with`), which never appear together. Level 0 of every factor is the
//! "not passed" level and conflicts with nothing, so a row can always be completed. Deterministic by
//! construction (no randomness; ties broken by the lowest level, then the first row), so the same surface yields
//! the same cells on every host and every run (R-15) — a receipt keyed by cell id means the same cell tomorrow.

use std::collections::BTreeSet;

/// A forbidden pair `(factor i, level a, factor j, level b)` with `i < j`.
pub type Pair = (usize, usize, usize, usize);

/// Rows of level indices, one entry per factor, covering every allowed pair of `levels`.
#[must_use]
pub fn pairwise(levels: &[usize], forbidden: &BTreeSet<Pair>) -> Vec<Vec<usize>> {
    let n = levels.len();
    if n == 0 || levels.contains(&0) {
        return vec![Vec::new(); usize::from(n == 0)];
    }
    if n == 1 {
        return (0..levels[0]).map(|a| vec![a]).collect();
    }
    let mut rows: Vec<Vec<Option<usize>>> = Vec::new();
    for a in 0..levels[0] {
        for b in 0..levels[1] {
            if !forbidden.contains(&(0, a, 1, b)) {
                rows.push(vec![Some(a), Some(b)]);
            }
        }
    }
    for k in 2..n {
        extend(&mut rows, levels, forbidden, k);
    }
    rows.into_iter()
        .map(|r| r.into_iter().map(|x| x.unwrap_or(0)).collect())
        .collect()
}

/// Add factor `k`: horizontally (one level per existing row), then vertically (rows for what is left).
fn extend(
    rows: &mut Vec<Vec<Option<usize>>>,
    levels: &[usize],
    forbidden: &BTreeSet<Pair>,
    k: usize,
) {
    let mut uncovered: BTreeSet<(usize, usize, usize)> = BTreeSet::new(); // (j, b, a): factor j level b with k=a
    for j in 0..k {
        for b in 0..levels[j] {
            for a in 0..levels[k] {
                if !forbidden.contains(&(j, b, k, a)) {
                    uncovered.insert((j, b, a));
                }
            }
        }
    }
    for row in rows.iter_mut() {
        let best = (0..levels[k])
            .filter(|&a| allowed(row, forbidden, k, a))
            .max_by_key(|&a| (gain(row, &uncovered, a), std::cmp::Reverse(a)))
            .unwrap_or(0);
        row.push(Some(best));
        cover(row, &mut uncovered, k);
    }
    let pending: Vec<(usize, usize, usize)> = uncovered.iter().copied().collect();
    for (j, b, a) in pending {
        if !uncovered.contains(&(j, b, a)) {
            continue;
        }
        let slot = rows
            .iter()
            .position(|r| r[k] == Some(a) && r[j].is_none() && allowed_with(r, forbidden, j, b));
        match slot {
            Some(i) => rows[i][j] = Some(b),
            None => {
                let mut r = vec![None; k + 1];
                r[j] = Some(b);
                r[k] = Some(a);
                rows.push(r);
            }
        }
        let i = slot.unwrap_or(rows.len() - 1);
        cover(&rows[i].clone(), &mut uncovered, k);
    }
}

/// How many uncovered (·, k=a) pairs setting factor k to `a` on `row` would cover.
fn gain(row: &[Option<usize>], uncovered: &BTreeSet<(usize, usize, usize)>, a: usize) -> usize {
    row.iter()
        .enumerate()
        .filter(|(j, v)| v.is_some_and(|b| uncovered.contains(&(*j, b, a))))
        .count()
}

/// Remove the pairs row `row` covers with factor `k`.
fn cover(row: &[Option<usize>], uncovered: &mut BTreeSet<(usize, usize, usize)>, k: usize) {
    let Some(a) = row.get(k).copied().flatten() else {
        return;
    };
    for (j, v) in row.iter().enumerate().take(k) {
        if let Some(b) = v {
            uncovered.remove(&(j, *b, a));
        }
    }
}

/// May factor `k` take level `a` in `row` without forming a forbidden pair?
fn allowed(row: &[Option<usize>], forbidden: &BTreeSet<Pair>, k: usize, a: usize) -> bool {
    row.iter()
        .enumerate()
        .all(|(j, v)| v.is_none_or(|b| !forbidden.contains(&(j, b, k, a))))
}

/// May factor `j` (currently unset in `row`) take level `b`?
fn allowed_with(row: &[Option<usize>], forbidden: &BTreeSet<Pair>, j: usize, b: usize) -> bool {
    row.iter().enumerate().all(|(i, v)| {
        v.is_none_or(|c| {
            let p = if i < j { (i, c, j, b) } else { (j, b, i, c) };
            i == j || !forbidden.contains(&p)
        })
    })
}

/// The allowed pairs `rows` does NOT cover — empty for a correct array. The property the tests hold it to.
#[must_use]
pub fn uncovered(levels: &[usize], forbidden: &BTreeSet<Pair>, rows: &[Vec<usize>]) -> Vec<Pair> {
    let mut out = Vec::new();
    for i in 0..levels.len() {
        for j in i + 1..levels.len() {
            for a in 0..levels[i] {
                for b in 0..levels[j] {
                    let p = (i, a, j, b);
                    if !forbidden.contains(&p) && !rows.iter().any(|r| r[i] == a && r[j] == b) {
                        out.push(p);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn violates(rows: &[Vec<usize>], forbidden: &BTreeSet<Pair>) -> bool {
        rows.iter()
            .any(|r| forbidden.iter().any(|&(i, a, j, b)| r[i] == a && r[j] == b))
    }

    #[test]
    fn every_allowed_pair_is_covered_and_no_forbidden_pair_appears() {
        // run-like: 20 factors, one with 6 levels, one with 3, the rest flags (measured shape of `apr run`)
        let mut levels = vec![2usize; 18];
        levels.insert(3, 6);
        levels.insert(9, 3);
        let forbidden: BTreeSet<Pair> = [(0, 1, 1, 1), (4, 1, 12, 1)].into_iter().collect();
        let rows = pairwise(&levels, &forbidden);
        assert!(uncovered(&levels, &forbidden, &rows).is_empty());
        assert!(
            !violates(&rows, &forbidden),
            "a conflicts_with pair never shares a row"
        );
        assert!(
            rows.len() <= 40,
            "{} rows: IPOG should stay near 6×3",
            rows.len()
        );
        assert_eq!(rows, pairwise(&levels, &forbidden), "deterministic (R-15)");
    }

    #[test]
    fn small_cases_are_exact() {
        assert_eq!(pairwise(&[], &BTreeSet::new()), vec![Vec::<usize>::new()]);
        assert_eq!(
            pairwise(&[3], &BTreeSet::new()),
            vec![vec![0], vec![1], vec![2]]
        );
        let three_flags = pairwise(&[2, 2, 2], &BTreeSet::new());
        assert_eq!(three_flags.len(), 4, "the optimum for 3 binary factors");
        assert!(uncovered(&[2, 2, 2], &BTreeSet::new(), &three_flags).is_empty());
    }

    #[test]
    fn a_sweep_of_shapes_is_always_covering() {
        for n in 2..9 {
            for top in 2..5 {
                let levels: Vec<usize> = (0..n).map(|i| 2 + (i * 7 + top) % (top + 1)).collect();
                let forbidden: BTreeSet<Pair> =
                    [(0, 1, n - 1, 1)].into_iter().filter(|_| n > 2).collect();
                let rows = pairwise(&levels, &forbidden);
                assert!(
                    uncovered(&levels, &forbidden, &rows).is_empty(),
                    "{levels:?}"
                );
                assert!(!violates(&rows, &forbidden), "{levels:?}");
            }
        }
    }
}
