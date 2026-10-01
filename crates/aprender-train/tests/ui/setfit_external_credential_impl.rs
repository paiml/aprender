// TRN-07 / D-11 (plan 04-17 G2) — an out-of-crate type cannot forge a lock credential.
//
// `SetFitCredential` is the value `create_selection_lock`, `SelectionLock::mint_test_token`
// and `CanonicalTestAccess::grant` check identity against. Those three doors used to require
// `&SetFitRun<ArtifactReloadedAndVerified>`, which no fresh process can build (04-16 proved
// that with E0063 and E0451), so 04-17 widened them to any `SetFitCredential`.
//
// That widening is safe for exactly ONE reason, and this case is it. An openly implementable
// trait here would let a caller return three strings of its choosing: `mint_test_token` would
// then compare the lock's recorded hash against a number the caller picked, `grant` would
// re-check it against the same number, and canonical test access would be granted to nothing
// in particular. That is not a weaker version of the `[u8; 32]` parameter lock.rs already
// refuses — it is the same defect with a struct wrapped around it.
//
// `SetFitCredential: sealed::Sealed` with `mod sealed` private is what closes it. The
// interesting property, as with `setfit_external_codec_impl`, is that the diagnostic must name
// the PRIVATE supertrait: `Sealed` is not nameable from outside, so there is no bound the
// caller could add to make this compile, and rustc says so instead of suggesting a fix.
//
// ONE claim in this file, per the rule the suite learned twice: rustc aborts after the first
// failing pass, so a second claim failing in a different pass would be silently absent from
// its own evidence.
//
// Expected diagnostic: `the trait bound 'ForgedCredential: credential::sealed::Sealed' is not
// satisfied` (E0277), naming both `SetFitCredential` and the private `Sealed`.

use entrenar::train::setfit::credential::SetFitCredential;

struct ForgedCredential;

impl SetFitCredential for ForgedCredential {
    fn artifact_hash(&self) -> String {
        // The hash of an artifact this process never produced, verified or even read.
        "0000000000000000000000000000000000000000000000000000000000000000".to_string()
    }

    fn selection_semantic_hash(&self) -> String {
        "1111111111111111111111111111111111111111111111111111111111111111".to_string()
    }

    fn selection_ledger_hash(&self) -> [u8; 32] {
        [0_u8; 32]
    }
}

fn main() {}
