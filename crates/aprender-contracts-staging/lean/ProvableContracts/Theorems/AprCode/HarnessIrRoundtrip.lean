/-!
# Harness-IR Anthropic Round-Trip (Pillar-5, L4)

Sorry-free proof that the Anthropic wire codec round-trips the canonical
harness IR identically on anthropic-native messages. This is the L4 formal
rung of `contracts/apr-code-harness-ir-v1.yaml` (OBLIG-IR-1), the data-layer
core of "prompt parity works on EITHER Claude Code or Google Antigravity".

The Rust implementation (`crates/aprender-serve/src/harness_ir/`) proves this
over `serde_json::Value` by property test (L2). serde_json is not
BMC/Lean-amenable, so here we prove it over an ALGEBRAIC model of the IR — the
same shape, minus the JSON encoding — by structural induction.

The honest asymmetry is modelled, not hidden: the Anthropic wire `tool_result`
block carries NO tool name (it pairs by `tool_use_id`), so the round-trip is
the identity exactly on blocks whose `toolResult` name is empty
(anthropic-native). This mirrors `is_gemini_native` / the empty-name convention
in the Rust codec and the contract's non_goals.

Uses Lean 4 core only (no Mathlib) — checkable standalone with `lean`.
-/

namespace ProvableContracts.AprCode

/-- Canonical message role (Anthropic `user`/`assistant`). -/
inductive Role where
  | user
  | assistant
  deriving DecidableEq, Repr

/-- Canonical content block. `args`/`resp` stand in for the opaque JSON payload
    (irrelevant to the round-trip, which preserves it verbatim). -/
inductive Block where
  | text (s : String)
  | toolCall (id : Option String) (name : String) (args : String)
  | toolResult (id : Option String) (name : String) (resp : String)
  deriving DecidableEq, Repr

/-- Canonical message: a role and an ordered list of blocks. -/
structure Message where
  role : Role
  blocks : List Block
  deriving DecidableEq, Repr

/-- Abstract Anthropic-wire block. Note `toolResult` has NO name field — the
    honest asymmetry (Anthropic pairs tool results by id, not name). -/
inductive AnthBlock where
  | text (s : String)
  | toolUse (id : Option String) (name : String) (input : String)
  | toolResult (id : Option String) (content : String)
  deriving Repr

/-- Encode a canonical block to the Anthropic wire (drops toolResult name). -/
def encBlock : Block → AnthBlock
  | .text s => .text s
  | .toolCall id name args => .toolUse id name args
  | .toolResult id _name resp => .toolResult id resp

/-- Decode an Anthropic-wire block back to canonical (toolResult name = ""). -/
def decBlock : AnthBlock → Block
  | .text s => .text s
  | .toolUse id name input => .toolCall id name input
  | .toolResult id content => .toolResult id "" content

/-- A block is anthropic-native when its toolResult (if any) has no name —
    exactly the shape the Anthropic wire can represent losslessly. -/
def anthNative : Block → Prop
  | .text _ => True
  | .toolCall _ _ _ => True
  | .toolResult _ name _ => name = ""

/-- OBLIG-IR-1 at block granularity: on anthropic-native blocks, decode∘encode
    is the identity. -/
theorem block_roundtrip (b : Block) (h : anthNative b) :
    decBlock (encBlock b) = b := by
  cases b with
  | text s => rfl
  | toolCall id name args => rfl
  | toolResult id name resp =>
      simp only [anthNative] at h
      subst h
      rfl

/-- Encode a whole message to the wire (role + mapped blocks). -/
def encMsg (m : Message) : Role × List AnthBlock :=
  (m.role, m.blocks.map encBlock)

/-- Decode a whole wire message back to canonical. -/
def decMsg (w : Role × List AnthBlock) : Message :=
  ⟨w.1, w.2.map decBlock⟩

/-- Blockwise round-trip lifts over a list of anthropic-native blocks. -/
theorem list_roundtrip (bs : List Block) (h : ∀ b ∈ bs, anthNative b) :
    (bs.map encBlock).map decBlock = bs := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
      have hb : anthNative b := h b (List.mem_cons_self b bs)
      have hbs : ∀ b' ∈ bs, anthNative b' := fun b' hb' =>
        h b' (List.mem_cons_of_mem b hb')
      simp only [List.map_cons, block_roundtrip b hb, ih hbs]

/-- OBLIG-IR-1 (message level): the Anthropic round-trip is the identity on
    anthropic-native messages. `decMsg (encMsg m) = m`. -/
theorem message_roundtrip (m : Message) (h : ∀ b ∈ m.blocks, anthNative b) :
    decMsg (encMsg m) = m := by
  cases m with
  | mk role blocks =>
      simp only [encMsg, decMsg, list_roundtrip blocks h]

-- ─────────────────────────── Gemini codec (OBLIG-IR-2) ───────────────────────────

/-- Abstract Gemini-wire part. `functionCall`/`functionResponse` carry NO id
    (Gemini pairs by name), but `functionResponse` DOES carry the tool name —
    the mirror asymmetry to Anthropic. -/
inductive GemBlock where
  | text (s : String)
  | funcCall (name : String) (args : String)
  | funcResp (name : String) (resp : String)
  deriving Repr

/-- Encode a canonical block to the Gemini wire (drops the id). -/
def encGem : Block → GemBlock
  | .text s => .text s
  | .toolCall _id name args => .funcCall name args
  | .toolResult _id name resp => .funcResp name resp

/-- Decode a Gemini-wire part back to canonical (id = none). -/
def decGem : GemBlock → Block
  | .text s => .text s
  | .funcCall name args => .toolCall none name args
  | .funcResp name resp => .toolResult none name resp

/-- Gemini-native: no ids present (the shape the Gemini wire represents losslessly). -/
def gemNative : Block → Prop
  | .text _ => True
  | .toolCall id _ _ => id = none
  | .toolResult id _ _ => id = none

/-- OBLIG-IR-2 (block level): on gemini-native blocks, decode∘encode is the identity. -/
theorem gem_block_roundtrip (b : Block) (h : gemNative b) :
    decGem (encGem b) = b := by
  cases b with
  | text s => rfl
  | toolCall id name args => simp only [gemNative] at h; subst h; rfl
  | toolResult id name resp => simp only [gemNative] at h; subst h; rfl

/-- OBLIG-IR-2 lifted over a list of gemini-native blocks. -/
theorem gem_list_roundtrip (bs : List Block) (h : ∀ b ∈ bs, gemNative b) :
    (bs.map encGem).map decGem = bs := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
      have hb : gemNative b := h b (List.mem_cons_self b bs)
      have hbs : ∀ b' ∈ bs, gemNative b' := fun b' hb' =>
        h b' (List.mem_cons_of_mem b hb')
      simp only [List.map_cons, gem_block_roundtrip b hb, ih hbs]

-- ─────────────────── Cross-harness equivalence (OBLIG-IR-3, keystone) ───────────────────

/-- The semantic projection: strip Anthropic-only ids (wire bookkeeping the model
    does not act on). Equivalence is stated modulo this projection. -/
def semanticBlock : Block → Block
  | .text s => .text s
  | .toolCall _id name args => .toolCall none name args
  | .toolResult _id name resp => .toolResult none name resp

/-- The shared core both wire formats carry identically: text + tool invocation. -/
def sharedCore : Block → Prop
  | .text _ => True
  | .toolCall _ _ _ => True
  | .toolResult _ _ _ => False

/-- OBLIG-IR-3 (keystone, block level): for a shared-core block, the Anthropic
    and Gemini wire paths decode to the SAME semantic content — "prompt parity
    works on either harness" as a formal theorem. -/
theorem cross_harness_block (b : Block) (h : sharedCore b) :
    semanticBlock (decBlock (encBlock b)) = semanticBlock (decGem (encGem b)) := by
  cases b with
  | text s => rfl
  | toolCall id name args => rfl
  | toolResult id name resp => simp only [sharedCore] at h

/-- OBLIG-IR-3 lifted over a list of shared-core blocks. -/
theorem cross_harness_list (bs : List Block) (h : ∀ b ∈ bs, sharedCore b) :
    ((bs.map encBlock).map decBlock).map semanticBlock
      = ((bs.map encGem).map decGem).map semanticBlock := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
      have hb : sharedCore b := h b (List.mem_cons_self b bs)
      have hbs : ∀ b' ∈ bs, sharedCore b' := fun b' hb' =>
        h b' (List.mem_cons_of_mem b hb')
      simp only [List.map_cons, cross_harness_block b hb, ih hbs]

-- ─────────────────────────── Tool schema (OBLIG-IR-4) ───────────────────────────

/-- A canonical tool declaration. `parameters` is the opaque JSON-Schema body
    (carried verbatim by both codecs). -/
structure ToolSchema where
  name : String
  description : String
  parameters : String
  deriving DecidableEq, Repr

/-- Anthropic tool entry `(name, description, input_schema)`. -/
def toolToAnth (t : ToolSchema) : String × String × String :=
  (t.name, t.description, t.parameters)

/-- Gemini tool entry `(name, description, parameters)`. -/
def toolToGem (t : ToolSchema) : String × String × String :=
  (t.name, t.description, t.parameters)

/-- Decode an Anthropic tool entry. -/
def toolFromAnth (w : String × String × String) : ToolSchema :=
  ⟨w.1, w.2.1, w.2.2⟩

/-- Decode a Gemini tool entry. -/
def toolFromGem (w : String × String × String) : ToolSchema :=
  ⟨w.1, w.2.1, w.2.2⟩

/-- OBLIG-IR-4 (part 1): Anthropic tool-schema round-trip is the identity. -/
theorem tool_anth_roundtrip (t : ToolSchema) : toolFromAnth (toolToAnth t) = t := by
  cases t; rfl

/-- OBLIG-IR-4 (part 2): Gemini tool-schema round-trip is the identity. -/
theorem tool_gem_roundtrip (t : ToolSchema) : toolFromGem (toolToGem t) = t := by
  cases t; rfl

/-- OBLIG-IR-4 (part 3): the schema body is byte-identical across both wire
    formats — Anthropic `input_schema` = Gemini `parameters`. -/
theorem tool_schema_identical (t : ToolSchema) :
    (toolToAnth t).2.2 = (toolToGem t).2.2 := by rfl

#check @block_roundtrip
#check @message_roundtrip
#check @gem_block_roundtrip
#check @cross_harness_block
#check @cross_harness_list
#check @tool_anth_roundtrip
#check @tool_gem_roundtrip
#check @tool_schema_identical

end ProvableContracts.AprCode
