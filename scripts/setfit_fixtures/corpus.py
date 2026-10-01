"""The CORPUS OF RECORD for every SetFit/MiniLM conformance fixture (B6).

Every source text used by ANY committed fixture is defined HERE, exactly once, with a
stable ``case_id``. ``tokenizer_cases.json`` is emitted directly from this module, and
every other model-driven fixture records its ``case_id`` plus a verbatim copy of its
``texts``.

WHY THIS MODULE EXISTS SEPARATELY FROM THE GENERATOR
----------------------------------------------------
``slice_model.py`` needs the corpus before ``generate_fixtures.py`` ever runs: the slice
vocabulary is the CLOSURE of the token ids these texts produce, so the slicer must
tokenize the corpus to decide which embedding rows to keep. Defining the texts in the
generator alone would force the slicer to import the generator (which loads torch models
at import time) or to duplicate the list -- and a duplicated corpus is precisely the
drift B6 exists to prevent. One module, imported by both, keeps "defined once" literally
true.

THE JOIN RULE (B6)
------------------
After the D-08 seal the only public batch producer is ``SetFitMiniLm::tokenize(texts)``,
which takes TEXTS. A fixture carrying token ids only would leave hand-constructing a
``SentenceBatch`` as the only route that compiles on the Rust side -- which silently
bypasses the tokenizer boundary while still passing every listed check. So each
model-driven fixture carries ``case_id`` + ``texts``, and the generator asserts that the
recorded ``input_ids_canonical`` equals the joined case's ``input_ids`` element for
element.

SEQUENCE-LENGTH BUDGET
----------------------
The committed slice keeps only 64 position embeddings, so every case driven through the
SLICE model must tokenize to <= 64 tokens. ``SLICE_DRIVEN_CASES`` lists exactly those;
``generate_fixtures.py`` asserts the bound rather than trusting this comment.
``truncation_long`` deliberately exceeds 256 tokens and is a TOKENIZER-ONLY case -- it
never reaches the slice.
"""

from __future__ import annotations

# Truncation case. Built by repetition on purpose: it must exceed the 256-token
# truncation bound while adding only a handful of NEW vocabulary ids, because every
# distinct id widens the committed slice embedding table (and therefore the APR size).
_TRUNCATION_SENTENCE = "the quick brown fox jumps over the lazy dog again and again "
TRUNCATION_TEXT = (_TRUNCATION_SENTENCE * 40).strip()

# case_id -> list of source texts. Order within a case IS the batch row order.
CASES: dict[str, list[str]] = {
    # --- the six required tokenizer case classes -------------------------------
    "single_short": ["A quick brown fox jumps over the lazy dog."],
    "mixed_length_pair": [
        "Short text.",
        "This sentence is deliberately longer so the batch holds two different "
        "lengths and padding is exercised.",
    ],
    "truncation_long": [TRUNCATION_TEXT],
    "cjk": ["東京は日本の首都です。", "机器学习很有趣。"],
    "accented": ["Café naïve résumé façade.", "El niño comió jalapeños."],
    "mixed_case": ["MiXeD CaSe TeXt HeRe.", "ALL CAPS AND lower case."],
    # --- pooling / normalize ----------------------------------------------------
    "pooling_batch": [
        "One line.",
        "Another line that is a good deal longer than the first one.",
        "Third.",
    ],
    # --- pair loss (labels {1.0, 0.0}: one similar pair, one dissimilar pair) ----
    "loss_pair_a": ["The cat sat on the mat.", "Stock markets fell sharply today."],
    "loss_pair_b": [
        "A feline rested on the rug.",
        "The weather is sunny and warm.",
    ],
    # --- padding invariance -----------------------------------------------------
    "invariance_single": ["Padding must not change this sentence."],
    "invariance_batch3": [
        "Tiny.",
        "Padding must not change this sentence.",
        "A noticeably longer sentence that forces the batch to pad the shorter rows.",
    ],
    # --- full pinned model reference (D-10 feed) --------------------------------
    "full_model_trio": [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Stock markets fell sharply today.",
    ],
}

# Which padded row of `invariance_batch3` is the same sentence as `invariance_single`.
INVARIANCE_TARGET_ROW = 1

# Truncation bound recorded in tokenizer_cases.json. This is a TRUNCATION bound only --
# padding is always to longest-in-batch, never padding="max_length" (binding policy).
MAX_LENGTH = 256

# Cases driven through the committed 64-position slice model. Must stay <= 64 tokens.
SLICE_DRIVEN_CASES: tuple[str, ...] = (
    "single_short",
    "mixed_length_pair",
    "pooling_batch",
    "loss_pair_a",
    "loss_pair_b",
    "invariance_single",
    "invariance_batch3",
)

# Cases driven through the FULL pinned model (384 hidden / 512 positions).
FULL_MODEL_CASES: tuple[str, ...] = ("full_model_trio",)

# Labels for the cosine-similarity / MSE pair loss, aligned row-wise with
# loss_pair_a x loss_pair_b. Row 0 is a paraphrase pair (1.0); row 1 is unrelated (0.0).
LOSS_PAIR_LABELS: list[float] = [1.0, 0.0]


def all_texts() -> list[str]:
    """Every text in the corpus, de-duplicated, in stable case order."""
    seen: dict[str, None] = {}
    for texts in CASES.values():
        for t in texts:
            seen.setdefault(t, None)
    return list(seen)
