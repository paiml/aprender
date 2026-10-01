// TRN-05 / D-08 — the head cannot be fitted on the PAIR stream.
//
// `fit_head(self)` takes zero non-`self` arguments. There is therefore no expression in which
// a caller hands it the pairs stage one consumed, which is what makes D-08's "unique rows,
// each exactly once" structural rather than a convention: the head's input is derived inside
// the transition from the run's own selection, and a pair stream is not something the
// signature can accept.
//
// Why this matters numerically and not just aesthetically: a row that appears in seven pairs
// would carry seven times the weight in the multinomial objective, so the fitted head would
// be a function of the pair BUDGET. `pair_weight_*` in the crate's own tests measures that
// divergence at a shared lambda; this file is the half that says the caller cannot even
// express it.
//
// `LabeledPair` is the crate's real public pair record, used rather than a tuple so the
// diagnostic is about ARITY and not about some placeholder type.
//
// Expected diagnostic: `this method takes 0 arguments but 1 argument was supplied` (E0061).

use aprender_contrastive_data::pairs::LabeledPair;
use entrenar::train::setfit::{EncoderTuned, SetFitRun};

fn fit_on_the_pair_stream(run: SetFitRun<EncoderTuned>, pairs: Vec<LabeledPair>) {
    let _ = run.fit_head(pairs);
}

fn main() {}
