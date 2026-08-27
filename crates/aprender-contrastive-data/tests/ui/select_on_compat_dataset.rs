// OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE / DATA-06 / D-19.
//
// Few-shot selection consumes `&PreparedDataset<Canonical>`. The compatibility profile is a
// DIFFERENT TYPE, not a flag, so a selection run over the SetFit-compatibility dataset —
// whose "test" split is validation and test MERGED — is not rejected at run time. It does
// not compile.
//
// Expected diagnostic: a type mismatch naming both `PreparedDataset<Compatibility>` and
// `PreparedDataset<Canonical>`.

use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::prepared::{Compatibility, PreparedDataset};
use aprender_contrastive_data::select::{FewShotSelector, SelectionConfig};

fn leak(
    dataset: &PreparedDataset<Compatibility>,
    cfg: &SelectionConfig,
    ledger: &mut AccessLedger,
) {
    let _ = FewShotSelector::select(dataset, cfg, ledger);
}

fn main() {}
