#!/usr/bin/env bash
# check_tokenizer_nonascii_parity.sh — #4005: apr's non-ASCII token ids EQUAL llama-tokenize on
# every tokenizer path apr ships (GGUF byte-level BPE, GGUF SentencePiece, HF tokenizer.json for
# .safetensors, .apr embedded). Re-verifies #3726 (non-ASCII -> id 0) on the release head.
#
# Runs the host-bound test crates/aprender-serve/tests/tokenizer_nonascii_4005.rs against the
# committed golden evidence/tokenizer-parity/nonascii-4005.json. A model the host does not hold,
# a file whose sha256 is not the golden's, and any id mismatch are FAIL -- never a skip.
#
# --self-test plants three defects in a COPY of the golden and requires each to be RED, naming
# its reason: a wrong reference id, a model that is not held, and a changed file (sha256).
#
# Exit: 0 green · 1 red · 2 could not check (cargo absent).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$ROOT" || exit 2
GOLDEN="$ROOT/evidence/tokenizer-parity/nonascii-4005.json"
command -v cargo > /dev/null 2>&1 || { echo "check_tokenizer_nonascii_parity: ENV - cargo is missing" >&2; exit 2; }

run() { # run <golden> <log> -> the test's exit status
  APR_TOKPARITY_GOLDEN="$1" cargo test --quiet -p aprender-serve --test tokenizer_nonascii_4005 \
    -- --include-ignored --test-threads=2 > "$2" 2>&1
}

if [ "${1:-}" = --self-test ]; then
  TMP=$(mktemp -d) || exit 2
  _rm_tmp() {
    case "${TMP:-}" in
      /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
      *) : ;;
    esac
  }
  trap _rm_tmp EXIT
  tmp=$TMP
  bad=0
  plant() { # plant <label> <must-match> <python expr over d>
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); exec(sys.argv[3]); json.dump(d, open(sys.argv[2],"w"), ensure_ascii=False)' \
      "$GOLDEN" "$tmp/$1.json" "$3"
    if run "$tmp/$1.json" "$tmp/$1.log"; then
      echo "FAIL  planted $1 was GREEN -- the gate cannot see it"; bad=1
    elif grep -q -- "$2" "$tmp/$1.log"; then
      echo "ok    planted $1 is RED ($2)"
    else
      echo "FAIL  planted $1 was RED for another reason: $(grep -m1 -E 'FAILED|panicked' "$tmp/$1.log")"; bad=1
    fi
  }
  plant wrong-reference-id "ids differ from llama-tokenize at index 3" 'd["cases"][3]["ids"][3] += 1'
  plant model-not-held     "NOT HELD"                               'd["cases"][0]["subject"] = "no-such-model.gguf"'
  plant changed-file       "is not the golden"                      'd["cases"][2]["subject_sha256"] = d["cases"][2]["subject_sha256"][::-1]'
  [ "$bad" = 0 ] && echo "self-test: PASS -- red on a wrong reference id, an unheld model and a changed file"
  exit "$bad"
fi

log=$(mktemp) || exit 2
if run "$GOLDEN" "$log"; then
  echo "ok    #4005: every tokenizer path's non-ASCII ids equal llama-tokenize ($(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["cases"]))' "$GOLDEN") cases)"
  rm -f "$log"; exit 0
fi
sed -n '/non-ASCII parity FAILED/,/^$/p' "$log"
grep -E 'error(\[|:)' "$log" | head -5
echo "RED   #4005: apr's non-ASCII tokenization differs from llama-tokenize (see above)"
rm -f "$log"; exit 1
