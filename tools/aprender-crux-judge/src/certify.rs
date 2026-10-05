//! The part of `crux_prompt_certify.py` the judge runs: `check`, which holds
//! a certification receipt against the prompt set it names, and the judge's
//! `certification_ok` wrapper around it (#3962 J2).

use sha2::{Digest, Sha256};

use crate::pyerr::PyResult;
use crate::pyio::{read_bytes, read_text_strict};
use crate::pyjson::loads;
use crate::pyval::{
    no_attr, py_join, py_ne, py_repr, py_str, py_strip, repr_str, slice_chars, Val,
};

pub const SCHEMA: &str = "crux-prompt-certification/v1";

/// `sha256_path(p)`: the hex digest of the file's bytes.
pub fn sha256_path(path: &str) -> PyResult<String> {
    let digest = Sha256::digest(read_bytes(path)?);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// `check(a)` with its stdout captured -> (rc, what it printed).
pub fn check(prompts: &str, receipt_path: &str) -> PyResult<(i32, String)> {
    let mut out = String::new();
    let receipt = loads(&read_text_strict(receipt_path)?)?;
    let schema = receipt.get("schema")?;
    if py_ne(&schema, &Val::str(SCHEMA)) {
        out.push_str(&format!(
            "refused: receipt schema {}, want {}\n",
            py_repr(&schema),
            repr_str(SCHEMA)
        ));
        return Ok((1, out));
    }
    let have_sha = sha256_path(prompts)?;
    let receipt_sha = py_str(&receipt.get("prompts_sha256")?);
    if py_ne(
        &receipt.get("prompts_sha256")?,
        &Val::str(have_sha.as_str()),
    ) {
        out.push_str(&format!(
            "refused: {prompts} is sha256 {}, the certification covers {} - an edited prompt set is \
             uncertified until it is certified again\n",
            &have_sha[..12],
            slice_chars(&receipt_sha, 0, 12)
        ));
        return Ok((1, out));
    }
    for u in receipt
        .get("uncontrolled_detail")?
        .or(Val::List(vec![]))
        .iter()?
    {
        let model = py_str(&u.item("model")?);
        let thinking = py_str(&u.item("thinking")?);
        let verbs = py_join(", ", &u.item("verbs")?)?;
        out.push_str(&format!(
            "note: {model} thinking={thinking} has no certified control for {verbs}\n"
        ));
    }
    let admitted = receipt.item("admitted")?;
    let Val::Dict(by_sha) = &admitted else {
        return Err(no_attr(&admitted, "values"));
    };
    let mut total = 0usize;
    for v in by_sha.values() {
        total += v.len()?;
    }
    out.push_str(&format!("certified: {total} (model, prompt) admissions\n"));
    Ok((0, out))
}

/// `certification_ok(prompts_path, receipt_path)` -> True, or the reason the
/// certifier refused.
pub fn certification_ok(prompts: &str, receipt: Option<&str>) -> PyResult<Val> {
    let Some(receipt) = receipt.filter(|r| !r.is_empty()) else {
        return Ok(Val::str("no --certification receipt was given"));
    };
    match check(prompts, receipt) {
        Ok((0, _)) => Ok(Val::Bool(true)),
        Ok((rc, printed)) => {
            let said = py_strip(&printed);
            Ok(Val::Str(if said.is_empty() {
                format!("refused (rc {rc})")
            } else {
                said.to_string()
            }))
        }
        Err(e) if e.declines() => Ok(Val::Str(format!("certification unreadable: {e}"))),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_and_digest() {
        assert_eq!(
            py_join(", ", &Val::List(vec![Val::str("run"), Val::str("chat")])).unwrap(),
            "run, chat"
        );
        assert_eq!(py_join(", ", &Val::str("ab")).unwrap(), "a, b");
        assert_eq!(
            py_join(", ", &Val::List(vec![Val::int(1)]))
                .unwrap_err()
                .msg,
            "sequence item 0: expected str instance, int found"
        );
        assert_eq!(
            py_join(", ", &Val::int(1)).unwrap_err().msg,
            "can only join an iterable"
        );
        assert_eq!(
            certification_ok("x", None).unwrap().as_str(),
            Some("no --certification receipt was given")
        );
    }
}
