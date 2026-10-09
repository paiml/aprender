#!/usr/bin/env bash
# gen_stream_golden_4918.sh - write T5's golden file for aprender#4918 from a
# checkout of v0.70.2.
#
# T5 (contracts/serve-stream-tool-calls-v1.yaml) says a stream with no `tools`,
# or with `tool_choice: "none"`, is byte-identical to the stream before the
# fix. "Before the fix" has to be measured on the tree that had no fix: a
# golden written by the fixed tree would agree with whatever the fix does. So
# this script refuses any checkout but a clean v0.70.2.
#
# It appends a test module to the checkout's openai_handlers.rs (both SSE
# builders are private there), renders GOLDEN_PIECES through each builder with
# `created` masked to 0, writes the result, and restores the file.
#
# Usage: scripts/gen_stream_golden_4918.sh CHECKOUT OUT
#   CHECKOUT  a clean checkout of v0.70.2
#   OUT       the golden file to write
# Never run it against a later tree. The pieces list must match GOLDEN_PIECES
# in crates/aprender-serve/src/api/tests/stream_tool_calls_4918.rs; T5 checks
# the `# pieces:` line this writes.
set -euo pipefail

readonly TAG="v0.70.2"
readonly TAG_SHA="89e261cda1042ad14dba50034da9a9090abcdecb"
readonly HANDLERS="crates/aprender-serve/src/api/openai_handlers.rs"

if [ "$#" -ne 2 ]; then
    printf 'usage: %s CHECKOUT OUT (CHECKOUT is a clean %s checkout)\n' "$0" "$TAG" >&2
    exit 2
fi
checkout="$1"
out="$(realpath -m "$2")"

head_sha="$(git -C "$checkout" rev-parse HEAD)"
if [ "$head_sha" != "$TAG_SHA" ]; then
    printf 'refused: %s is at %s, not %s (%s)\n' "$checkout" "$head_sha" "$TAG" "$TAG_SHA" >&2
    exit 1
fi
if [ -n "$(git -C "$checkout" status --porcelain)" ]; then
    printf 'refused: %s has local changes\n' "$checkout" >&2
    exit 1
fi
cd "$checkout" || exit 1

body="$(mktemp)"
restore() {
    git checkout -- "$HANDLERS"
    rm -f "${body:?}"
}
trap restore EXIT

cat >>"$HANDLERS" <<'RUST'

#[cfg(test)]
mod golden_4918 {
    use std::sync::Arc;

    use crate::tokenizer::BPETokenizer;

    const PIECES: &[&str] = &[
        "Sure",
        ".",
        " <",
        "tool",
        "_call",
        ">\n",
        "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}",
        "\n</tool_call>",
        " done",
    ];

    fn mask_created(body: &str) -> String {
        const KEY: &str = "\"created\":";
        let mut out = String::with_capacity(body.len());
        let mut rest = body;
        while let Some(at) = rest.find(KEY) {
            out.push_str(&rest[..at + KEY.len()]);
            out.push('0');
            rest = rest[at + KEY.len()..].trim_start_matches(|c: char| c.is_ascii_digit());
        }
        out.push_str(rest);
        out
    }

    async fn text(response: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read SSE body");
        mask_created(&String::from_utf8(bytes.to_vec()).expect("utf-8"))
    }

    #[tokio::test]
    async fn golden_4918_render() {
        let vocab: Vec<String> = PIECES.iter().map(|p| (*p).to_string()).collect();
        let ids: Vec<u32> = (0..u32::try_from(vocab.len()).expect("small")).collect();
        let tokenizer =
            Arc::new(BPETokenizer::new(vocab.clone(), vec![], vocab[0].as_str()).expect("tok"));

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(ids.len());
        for id in &ids {
            tx.send(Ok(*id)).await.expect("send token");
        }
        drop(tx);
        let live = text(super::true_streaming_sse_response(
            rx,
            Arc::clone(&tokenizer),
            "chatcmpl-4918".to_string(),
            "qwen3-coder".to_string(),
            Arc::new(crate::metrics::MetricsCollector::new()),
            std::time::Instant::now(),
            256,
            0,
            None,
            None,
        ))
        .await;
        let replayed = text(super::pregenerated_sse_response(
            ids,
            tokenizer,
            "chatcmpl-4918".to_string(),
            "qwen3-coder".to_string(),
            None,
            256,
            0,
        ))
        .await;
        let pieces = serde_json::to_string(PIECES).expect("pieces");
        let golden = format!("# pieces: {pieces}\n== live\n{live}== replayed\n{replayed}");
        let out = std::env::var("GOLDEN_4918_OUT").expect("GOLDEN_4918_OUT is set");
        std::fs::write(out, golden).expect("write golden body");
    }
}
RUST

GOLDEN_4918_OUT="$body" cargo test -p aprender-serve --lib golden_4918_render
if [ ! -s "$body" ]; then
    printf 'refused: the render test wrote nothing\n' >&2
    exit 1
fi

{
    printf '# aprender#4918 T5 golden: the SSE output before the fix, created masked to 0.\n'
    printf '# Generated from %s %s. Never regenerate it from a later tree.\n' "$TAG" "$TAG_SHA"
    printf '# Command: scripts/gen_stream_golden_4918.sh CHECKOUT OUT, CHECKOUT a clean %s checkout\n' "$TAG"
    cat "$body"
} >"$out"
printf 'wrote %s from %s %s\n' "$out" "$TAG" "$TAG_SHA"
