#!/usr/bin/env bash
# apr_embed_corpus.sh - APR-EMBED-001 EG-0(g): build the pre-registered embedding
# corpus from the project's own MIT-licensed book at one pinned commit, and write
# its sha256 manifest. Deterministic: the same commit gives the same bytes.
#
# Usage: bash scripts/apr_embed_corpus.sh [commit] [outdir]
#   commit  default: the commit recorded in <outdir>/SOURCE, else HEAD
#   outdir  default: docs/audits/APR-EMBED-001/corpus
# Check:  (cd docs/audits/APR-EMBED-001/corpus && sha256sum -c SHA256SUMS)
set -euo pipefail

OUT="${2:-docs/audits/APR-EMBED-001/corpus}"
case "${OUT}" in
    *..*) echo "apr_embed_corpus: refusing an output path containing '..': ${OUT}" >&2; exit 2 ;;
esac
if [ -n "${1:-}" ]; then
    COMMIT="$1"
elif [ -f "${OUT}/SOURCE" ]; then
    COMMIT="$(sed -n 's/^commit=//p' "${OUT}/SOURCE")"
else
    COMMIT="$(git rev-parse HEAD)"
fi
COMMIT="$(git rev-parse "${COMMIT}^{commit}")"
mkdir -p "${OUT}"

# Every book chapter at the pinned commit, in byte order (LC_ALL=C).
FILES="$(git ls-tree -r --name-only "${COMMIT}" -- book/src | grep '\.md$' | grep -v 'SUMMARY\.md$' | LC_ALL=C sort)"
if [ -z "${FILES}" ]; then
    echo "apr_embed_corpus: no book chapters at ${COMMIT}" >&2
    exit 1
fi

# prose <path>: one chapter as "H<TAB>heading" and "P<TAB>prose line" records.
# Code fences, tables, HTML, images and link targets are dropped.
prose() {
    git show "${COMMIT}:$1" | awk '
        /^```/ { fence = !fence; next }
        fence { next }
        /^\|/ || /^</ || /^!\[/ || /^[[:space:]]*$/ { next }
        /^## / { h = substr($0, 4); gsub(/[`*_]/, "", h); print "H\t" h; next }
        /^#/ { next }
        {
            l = $0
            gsub(/\[([^]]*)\]\([^)]*\)/, "&", l)
            while (match(l, /\[[^]]*\]\([^)]*\)/)) {
                t = substr(l, RSTART + 1); t = substr(t, 1, index(t, "]") - 1)
                l = substr(l, 1, RSTART - 1) t substr(l, RSTART + RLENGTH)
            }
            gsub(/[`*_>#]/, "", l); gsub(/\t/, " ", l)
            sub(/^[[:space:]]*([-+]|[0-9]+\.)[[:space:]]+/, "", l)
            gsub(/[[:space:]]+/, " ", l); sub(/^ /, "", l); sub(/ $/, "", l)
            if (l != "") print "P\t" l
        }'
}

# Sections: one row per "## heading" whose body has >= 40 words and whose heading
# has >= 3 words (generic one- and two-word headings do not identify a passage).
# split = cksum(path) mod 10: 0 retrieval, 1-2 heldout, 3-9 train. Split by FILE,
# so no chapter feeds two splits.
SECTIONS="$(mktemp)"
trap 'rm -f "${SECTIONS}"' EXIT
while IFS= read -r f; do
    bucket=$(( $(printf '%s' "${f}" | cksum | cut -d' ' -f1) % 10 ))
    case "${bucket}" in 0) split=retrieval ;; 1|2) split=heldout ;; *) split=train ;; esac
    prose "${f}" | awk -F'\t' -v f="${f}" -v s="${split}" '
        function flush() {
            if (h != "" && split(h, hw, " ") >= 3 && split(b, bw, " ") >= 40) {
                n = 0; body = ""
                for (i = 1; i <= length(bw) && n < 120; i++) { body = body (n ? " " : "") bw[i]; n++ }
                print s "\t" f "\t" h "\t" body
            }
            h = ""; b = ""
        }
        $1 == "H" { flush(); h = $2; next }
        $1 == "P" && h != "" { b = b " " $2 }
        END { flush() }'
done <<< "${FILES}" > "${SECTIONS}"

# retrieval/: queries (heading), docs (document form inputs), qrels (1:1).
# At most 16 queries; every retrieval-split body is a doc, so distractors exist.
mkdir -p "${OUT}/retrieval" "${OUT}/finetune"
awk -F'\t' '$1 == "retrieval"' "${SECTIONS}" | awk -F'\t' -v o="${OUT}/retrieval" '
    { n++; id = sprintf("r%03d", n)
      print id "\t" $3 "\t" $4 > (o "/docs.tsv")
      if (n <= 16) { print "q" substr(id, 2) "\t" $3 > (o "/queries.tsv"); print "q" substr(id, 2) "\t" id > (o "/qrels.tsv") } }'

# finetune/: train and held-out (query, positive passage) pairs, disjoint by file.
awk -F'\t' '$1 == "train"   { n++; printf "t%04d\t%s\t%s\n", n, $3, $4 }' "${SECTIONS}" > "${OUT}/finetune/train.tsv"
awk -F'\t' '$1 == "heldout" { n++; printf "h%04d\t%s\t%s\n", n, $3, $4 }' "${SECTIONS}" > "${OUT}/finetune/heldout.tsv"

# texts.tsv: 12 (title, text) items for the 7 prompts x document-form matrix,
# taken from the retrieval split at 60 words each.
awk -F'\t' '$1 == "retrieval" && n < 12 { n++; split($4, w, " "); t = ""
    for (i = 1; i <= 60 && i <= length(w); i++) t = t (i > 1 ? " " : "") w[i]
    printf "x%02d\t%s\t%s\n", n, $3, t }' "${SECTIONS}" > "${OUT}/texts.tsv"

# batch.tsv: 32 items of mixed length (3, 17, 60, 120 words), so padding differs
# across a batch of 8 or 32.
awk -F'\t' '$1 == "train" && n < 32 { split("3 17 60 120", L, " "); k = L[(n % 4) + 1]; n++
    split($4, w, " "); t = ""
    for (i = 1; i <= k && i <= length(w); i++) t = t (i > 1 ? " " : "") w[i]
    printf "b%02d\t%s\n", n, t }' "${SECTIONS}" > "${OUT}/batch.tsv"

# lengths.tsv: running book prose cut at word counts that target 8, 128, 1024,
# 2000 and 8000 tokens. The exact token count of each item is the oracle's,
# recorded with the goldens (EG-0 c); the target here is a label, not a claim.
while IFS= read -r f; do prose "${f}"; done <<< "${FILES}" | awk -F'\t' '$1 == "P" { print $2 }' \
    | tr '\n' ' ' | awk -v RS=' ' 'NF { w[++n] = $0 } END {
        split("8 128 1024 2000 8000", T, " "); split("6 96 780 1530 6100", W, " ")
        for (j = 1; j <= 5; j++) { t = ""
            for (i = 1; i <= W[j]; i++) t = t (i > 1 ? " " : "") w[i]
            printf "len%05d\t%d\t%s\n", T[j], W[j], t } }' > "${OUT}/lengths.tsv"

# prompts.tsv: the model card's seven task prompts and the document form (M5).
{
    printf 'name\tquery_template\tdocument_template\n'
    printf 'SearchQuery\ttask: search result | query: {content}\ttitle: {title} | text: {content}\n'
    printf 'QuestionAnswering\ttask: question answering | query: {content}\ttitle: {title} | text: {content}\n'
    printf 'FactChecking\ttask: fact checking | query: {content}\ttitle: {title} | text: {content}\n'
    printf 'CodeRetrieval\ttask: code retrieval | query: {content}\ttitle: {title} | text: {content}\n'
    printf 'Classification\ttask: classification | query: {content}\t-\n'
    printf 'Clustering\ttask: clustering | query: {content}\t-\n'
    printf 'SentenceSimilarity\ttask: sentence similarity | query: {content}\t-\n'
    printf 'Document\t-\ttitle: none | text: {content}\n'
} > "${OUT}/prompts.tsv"

printf 'commit=%s\nsource=book/src (MIT, this repository)\ngenerator=scripts/apr_embed_corpus.sh\n' "${COMMIT}" > "${OUT}/SOURCE"

# A corpus with an empty part measured nothing: fail rather than write it.
for p in texts.tsv batch.tsv lengths.tsv retrieval/docs.tsv retrieval/queries.tsv finetune/train.tsv finetune/heldout.tsv; do
    if [ ! -s "${OUT}/${p}" ]; then
        echo "apr_embed_corpus: ${p} is empty" >&2
        exit 1
    fi
done

(cd "${OUT}" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | sed 's|^\./||' | xargs sha256sum > SHA256SUMS)
echo "apr_embed_corpus: ${COMMIT} -> ${OUT}"
