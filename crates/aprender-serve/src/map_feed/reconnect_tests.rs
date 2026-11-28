//! Headless reconnect tests: a `ClientMirror` drops, misses batches, resumes,
//! and must end with the same column hash as the server.

use super::*;

/// Headless client-side mirror of one column.
#[derive(Debug)]
struct ClientMirror {
    column: String,
    seq: u64,
    values: Vec<f32>,
}

impl ClientMirror {
    fn new(column: &str, cells: usize) -> Self {
        Self {
            column: column.to_owned(),
            seq: 0,
            values: vec![0.0; cells],
        }
    }

    fn apply_batch(&mut self, batch: &Batch) -> Result<(), String> {
        if batch.seq != self.seq + 1 {
            return Err(format!(
                "gap: mirror at {} got batch {}",
                self.seq, batch.seq
            ));
        }
        for &(cell, value) in &batch.changes {
            let slot = usize::try_from(cell)
                .ok()
                .and_then(|i| self.values.get_mut(i))
                .ok_or_else(|| format!("cell {cell} out of range"))?;
            *slot = value;
        }
        self.seq = batch.seq;
        Ok(())
    }

    fn apply_resume(&mut self, resume: Resume) -> Result<(), String> {
        match resume {
            Resume::Replay(batches) => {
                for batch in &batches {
                    self.apply_batch(batch)?;
                }
                Ok(())
            },
            Resume::Snapshot { seq, values } => {
                self.seq = seq;
                self.values = values;
                Ok(())
            },
        }
    }

    fn hash(&self) -> u64 {
        column_hash(&self.values)
    }
}

/// Deterministic 64-bit xorshift: the seeded loop needs no proptest.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

fn server_with(cap: usize, cells: usize) -> ColumnFeed {
    let mut feed = ColumnFeed::new(cap);
    feed.add_column("col", cells).expect("add col");
    feed
}

/// Apply `n` deterministic batches to `col` on `feed`.
fn push(feed: &mut ColumnFeed, col: &str, n: usize, cells: u32, salt: u32) {
    for i in 0..n {
        let i = u32::try_from(i).expect("small n");
        feed.apply(col, &[((i + salt) % cells, (i * 3 + salt) as f32 - 1.5)])
            .expect("apply");
    }
}

/// Resume `mirror` against `feed` and assert it converges on the server column.
fn reconnect(feed: &ColumnFeed, mirror: &mut ClientMirror) -> Resume {
    let resume = feed.resume(&mirror.column, mirror.seq).expect("resume");
    let kind = resume.clone();
    mirror.apply_resume(resume).expect("mirror applies resume");
    assert_eq!(mirror.seq, feed.seq(&mirror.column).expect("seq"));
    assert_eq!(mirror.values, feed.values(&mirror.column).expect("values"));
    assert_eq!(
        mirror.hash(),
        column_hash(feed.values(&mirror.column).expect("values"))
    );
    kind
}

#[test]
fn reconnect_replay_path_ends_with_mirror_hash_equal_to_server() {
    let mut feed = server_with(8, 4);
    let mut mirror = ClientMirror::new("col", 4);
    push(&mut feed, "col", 3, 4, 0);
    reconnect(&feed, &mut mirror); // connected at seq 3

    push(&mut feed, "col", 4, 4, 7); // mirror is dropped and misses these
    let kind = reconnect(&feed, &mut mirror);
    assert!(
        matches!(kind, Resume::Replay(ref b) if b.len() == 4),
        "got {kind:?}"
    );
    assert_eq!(
        mirror.hash(),
        column_hash(feed.values("col").expect("values"))
    );
}

#[test]
fn reconnect_overflow_path_ends_with_mirror_hash_equal_to_server() {
    let mut feed = server_with(4, 5);
    let mut mirror = ClientMirror::new("col", 5);
    push(&mut feed, "col", 2, 5, 0);
    reconnect(&feed, &mut mirror); // connected at seq 2

    push(&mut feed, "col", 10, 5, 3); // 10 missed > ring 4: snapshot
    let kind = reconnect(&feed, &mut mirror);
    assert!(
        matches!(kind, Resume::Snapshot { seq: 12, .. }),
        "got {kind:?}"
    );
    assert_eq!(mirror.values, feed.values("col").expect("values"));
    assert_eq!(
        mirror.hash(),
        column_hash(feed.values("col").expect("values"))
    );
}

#[test]
fn reconnect_mirror_rejects_a_replay_that_repeats_a_seen_batch() {
    // A resume that re-sends an already-applied batch must be caught as a gap.
    let mut feed = server_with(8, 2);
    push(&mut feed, "col", 2, 2, 0);
    let mut mirror = ClientMirror::new("col", 2);
    reconnect(&feed, &mut mirror);
    let stale = Batch {
        seq: 2,
        changes: vec![(0, 9.0)],
    };
    assert!(mirror.apply_batch(&stale).is_err());
}

#[test]
fn reconnect_seeded_loop_of_60_batches_keeps_every_mirror_equal() {
    const CELLS: u32 = 6;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut feed = ColumnFeed::new(5);
    feed.add_column("a", CELLS as usize).expect("add a");
    feed.add_column("b", CELLS as usize).expect("add b");
    let mut mirrors = [
        ClientMirror::new("a", CELLS as usize),
        ClientMirror::new("b", CELLS as usize),
    ];
    let mut replays = 0_u32;
    let mut snapshots = 0_u32;

    for _ in 0..60 {
        let which = usize::try_from(rng.below(2)).expect("0 or 1");
        let col = if which == 0 { "a" } else { "b" };
        let n_changes = 1 + rng.below(3);
        let changes: Vec<(u32, f32)> = (0..n_changes)
            .map(|_| {
                let cell = u32::try_from(rng.below(u64::from(CELLS))).expect("cell fits u32");
                let value = (rng.below(2000) as f32) / 8.0 - 125.0;
                (cell, value)
            })
            .collect();
        feed.apply(col, &changes).expect("apply random batch");

        // With probability 1/3 one mirror reconnects now.
        if rng.below(3) == 0 {
            let idx = usize::try_from(rng.below(2)).expect("0 or 1");
            match reconnect(&feed, &mut mirrors[idx]) {
                Resume::Replay(_) => replays += 1,
                Resume::Snapshot { .. } => snapshots += 1,
            }
        }
    }

    for mirror in &mut mirrors {
        let col = mirror.column.clone();
        reconnect(&feed, mirror);
        assert_eq!(
            mirror.hash(),
            column_hash(feed.values(&col).expect("values"))
        );
    }
    assert!(replays > 0, "seeded loop never exercised the replay path");
    assert!(
        snapshots > 0,
        "seeded loop never exercised the snapshot path"
    );
}
