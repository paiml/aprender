import Mathlib.Data.List.Basic
import Mathlib.Data.List.Nodup
import Mathlib.Data.Finset.Lattice.Fold

/-!
# Tokenizer — encode/decode roundtrip, vocab-id bounds, merge determinism

Proves the *analytic core* of the tokenizer contract (`tokenizer-v1`).

A tokenizer vocabulary is modelled as a `List α` (`α` = a token, e.g. a byte
string).  The index of a token in this list is its **id**.  Encoding is
"look up the id of a token"; decoding is "look up the token at an id".

## Contract: tokenizer-v1

* `decode (encode t) = t`   — roundtrip on the vocabulary (a map inverse).
* `encode t < vocab_size`   — every emitted id is in range `[0, vocab_size)`.
* `encode` is injective on the vocabulary — distinct in-vocab tokens get
  distinct ids (this is exactly what makes `decode` a well-defined inverse).
* Byte-level BPE merge selection is deterministic: the lowest-rank applicable
  merge is uniquely determined (well-definedness of one BPE step).

The file-loading (`tokenizer.json` / GGUF / SentencePiece parsing), UTF-8
byte-boundary / whitespace-normalization, and config-driven special-token
obligations are runtime/empirical and are marked `l4_not_applicable` in the
contract — they have no algebraic statement to discharge here.
-/

namespace ProvableContracts.Tokenizer

variable {α : Type*} [DecidableEq α]

/-- Encode a token to its id: the index of `t` in the vocabulary list. -/
def encode (vocab : List α) (t : α) : ℕ := vocab.idxOf t

/-- Decode an id back to a token: the vocabulary entry at that index
    (`none` when the id is out of range). -/
def decode (vocab : List α) (i : ℕ) : Option α := vocab[i]?

/-- **Vocab-id bound.** Every id produced by `encode` for an in-vocab token is
    a valid index, i.e. strictly less than the vocabulary size. -/
theorem encode_lt_vocab_size (vocab : List α) (t : α) (h : t ∈ vocab) :
    encode vocab t < vocab.length := by
  unfold encode
  exact List.idxOf_lt_length_of_mem h

/-- **Encode/decode roundtrip.** Decoding the id of an in-vocab token recovers
    the token: `decode (encode t) = some t`.  This is the map-inverse property
    at the heart of the tokenizer contract. -/
theorem decode_encode (vocab : List α) (t : α) (h : t ∈ vocab) :
    decode vocab (encode vocab t) = some t := by
  unfold decode encode
  exact List.getElem?_idxOf h

/-- `decode ∘ encode` is the identity (`some`-lifted) on the vocabulary — the
    same statement as `decode_encode`, phrased as a left inverse. -/
theorem decode_left_inverse_encode (vocab : List α) :
    ∀ t ∈ vocab, decode vocab (encode vocab t) = some t :=
  fun t h => decode_encode vocab t h

/-- **Encode is injective on the vocabulary.** Two in-vocab tokens with the
    same id are equal.  This is what makes the id → token map single-valued,
    so `decode` is a genuine inverse rather than an arbitrary choice. -/
theorem encode_injOn (vocab : List α) (t₁ t₂ : α)
    (h₁ : t₁ ∈ vocab) (h₂ : t₂ ∈ vocab)
    (heq : encode vocab t₁ = encode vocab t₂) : t₁ = t₂ := by
  have e₁ : decode vocab (encode vocab t₁) = some t₁ := decode_encode vocab t₁ h₁
  have e₂ : decode vocab (encode vocab t₂) = some t₂ := decode_encode vocab t₂ h₂
  rw [heq] at e₁
  rw [e₂] at e₁
  exact ((Option.some.injEq _ _).mp e₁).symm

/-- **BPE merge-order determinism (structural).**

    In byte-level BPE, at each step the merge chosen is the *applicable* merge
    of minimum rank.  Model the set of ranks of applicable merges as a `Finset ℕ`.
    If two ranks `r₁, r₂` are both minimal over this set, they are equal — so the
    "pick the lowest-rank merge" rule is well-defined and one BPE step is
    deterministic (no dependence on iteration order or tie-breaking). -/
theorem merge_rank_unique {s : Finset ℕ} {r₁ r₂ : ℕ}
    (h₁ : r₁ ∈ s) (h₂ : r₂ ∈ s)
    (m₁ : ∀ x ∈ s, r₁ ≤ x) (m₂ : ∀ x ∈ s, r₂ ≤ x) : r₁ = r₂ :=
  le_antisymm (m₁ r₂ h₂) (m₂ r₁ h₁)

end ProvableContracts.Tokenizer
