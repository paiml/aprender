// APR-04 — the consumer-side witness type cannot be MINTED from outside aprender-core.
//
// THE OBLIGATION, in one sentence: evaluation, registration, benchmarking, prediction and
// serving accept only `ArtifactReloadedAndVerified` — out-of-crate code cannot mint the
// consumer-side witness type.
//
// `VerifiedSetFitModel` is the value with the classify capability, and `load_setfit_apr` is
// the only function that returns one. That door runs the whole fail-closed ladder: the raw
// length cap, the container CRCs, the typed tag, the deny_unknown_fields document, the
// architecture-derived tensor set, the non-finite scan, the rebuild, and the replay of all
// six embedded probes. A consumer that could construct the type directly would be a SECOND
// verification policy with its own tolerances, which is the drift Pitfall 9 names.
//
// `aprender-train` is the right tier for this proof: it is OUT OF CRATE relative to
// `aprender-core`, which is exactly the boundary APR-04 is about. A compile-fail case inside
// aprender-core would only prove that a module can see its own private fields.
//
// # Why this file contains ONE attempt and not two — measured, not assumed
//
// The plan asked for both illegal doors in one file: the struct literal, and a
// `VerifiedSetFitModel::new()` that does not exist. Both were written, and the blessed
// snapshot then contained ONLY `E0599: no function or associated item named 'new'` — rustc
// aborts after the resolution error, so the E0451 privacy diagnostic this case is actually
// about was never emitted and would have been silently absent from its own evidence. That is
// the same finding `setfit_direct_state_construction.rs` records for E0616 vs E0451: two
// claims that fail in DIFFERENT compiler passes cannot share a snapshot.
//
// The load-bearing half is "the type is not CONSTRUCTIBLE", so that is the half that is here.
// The absent constructor is covered by a source assertion instead (`pub fn new` does not
// appear for this type anywhere in `setfit/artifact.rs`), which is a claim a grep can settle
// and does not need a compiler pass it has to share.
//
// The `allow` is deliberate and narrow: `unimplemented!()` has type `!`, so every field after
// the first is unreachable and the warning would otherwise ride in the snapshot as noise
// about macro expansion rather than about privacy.
//
// Expected diagnostic: `fields 'model', 'head', 'doc' and 'artifact_sha256' of struct
// 'VerifiedSetFitModel' are private` (E0451).

use aprender::setfit::VerifiedSetFitModel;

#[allow(unreachable_code)]
fn forge_a_verified_model() -> VerifiedSetFitModel {
    VerifiedSetFitModel {
        model: unimplemented!(),
        head: unimplemented!(),
        doc: unimplemented!(),
        artifact_sha256: unimplemented!(),
    }
}

fn main() {}
