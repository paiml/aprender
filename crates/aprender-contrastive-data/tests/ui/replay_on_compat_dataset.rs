// OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE / DATA-06 / D-19.
//
// Review finding F6: closing the FORWARD path is not enough if the replay path is open. A
// forged manifest claiming the compatibility profile would still need a dataset to be
// replayed against, and `Selection::replay` takes `&PreparedDataset<Canonical>` — so there
// is no such dataset to hand it. The profile check in `check_provenance` is the belt; this
// is the braces, and it is structural.
//
// Expected diagnostic: the same type mismatch, naming `PreparedDataset<Compatibility>` and
// `PreparedDataset<Canonical>`.

use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::manifest::SelectionManifest;
use aprender_contrastive_data::prepared::{Compatibility, PreparedDataset};
use aprender_contrastive_data::select::Selection;

fn replay_against_compat(
    manifest: &SelectionManifest,
    dataset: &PreparedDataset<Compatibility>,
    ledger: &mut AccessLedger,
) {
    let _ = Selection::replay(manifest, dataset, ledger);
}

fn main() {}
