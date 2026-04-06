# Phase 3 Quick Reference Card

## At a Glance

**Current**: 14 core traits + iterators  
**Missing**: 7 valuable traits  
**Priority**: 4 MUST-HAVE + 3 HIGHLY-VALUABLE + 3 OPTIONAL  

---

## The 4 MUST-HAVE Traits (Phase 3.1)

| Trait | What It Does | Why It Matters | Complexity |
|-------|-------------|-----------------|-----------|
| `Deref<[T]>` | Auto-coerce to slice | `vec.len()` instead of `vec.as_slice().len()` | 1 line |
| `DerefMut` | Mutable slice coercion | Enables `vec.push()` style patterns | 1 line |
| `AsRef<[T]>` | Explicit borrow | Works with `fn foo<T: AsRef<[u8]>>` | 1 line |
| `AsMut<[T]>` | Explicit mutable borrow | Works with generic mutable bounds | 1 line |

**Time to implement**: 1 day | **Risk**: Minimal | **Impact**: Huge

---

## The 3 HIGHLY-VALUABLE Traits (Phase 3.2)

| Trait | What It Does | Why It Matters | Complexity |
|-------|-------------|-----------------|-----------|
| `PartialOrd` | Enable `<`, `>`, `<=`, `>=` | Comparisons, sorting | Low |
| `Ord` | Total ordering | BTreeMap keys, stable sorts | Low |
| `Hash` | Hashable values | HashMap/HashSet keys | Low |

**Time to implement**: 2-3 days | **Risk**: Low | **Impact**: High

---

## The Optional Polish (Phase 3.3)

| Trait | What It Does | Recommendation |
|-------|-------------|-----------------|
| `Display` | Pretty printing with `{}` | Nice-to-have |
| `Borrow<[T]>` | Advanced borrowing pattern | Skip if AsRef exists |
| `BorrowMut<[T]>` | Advanced mutable borrowing | Skip if AsMut exists |

---

## Implementation Patterns

All Phase 3 traits follow a simple pattern: **Delegate to existing methods**

```rust
// Pattern for Deref/DerefMut/AsRef/AsMut
impl<T> Deref for TruenoVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()  // ← That's it!
    }
}

// Pattern for PartialOrd/Ord/Hash
impl<T: Hash> Hash for TruenoVec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)  // ← Delegate to slice
    }
}
```

No complex logic needed - just forward to slice equivalents.

---

## Testing Pyramid for Phase 3

```
┌────────────────────┐
│  Integration Tests │  8-12 tests
│  (HashMap, BTreeMap,
│   standard library)
├────────────────────┤
│  Property-Based    │  25 properties
│  (Semantics proofs)
├────────────────────┤
│  Unit Tests        │  87 tests
│  (Basic behavior)  │
└────────────────────┘
```

---

## Dependencies Between Traits

```
Phase 3.1
├── Deref ──┬─→ Available immediately
├── DerefMut┤
├── AsRef ──┤
└── AsMut ──┘

        ↓ (prerequisite for Phase 3.2)

Phase 3.2
├── PartialOrd ──→ Sort, comparisons
├── Ord ─────────→ (depends on PartialOrd + Eq)
└── Hash ────────→ HashMap, HashSet
```

**Important**: Merge Phase 3.1 before starting Phase 3.2.

---

## Coverage & Quality Targets

| Metric | Current | Target Phase 3 |
|--------|---------|-----------------|
| Line Coverage | 97.7% | 95%+ (maintain) |
| Mutation Score | 97.7% | 85%+ (per trait) |
| Cyclomatic Complexity | ≤10 | ≤10 (enforce) |
| Tests | 127 | 250+ |
| SATD Comments | 0 | 0 (zero-tolerance) |

---

## Estimated Effort Breakdown

### Phase 3.1 (Deref, DerefMut, AsRef, AsMut)
- Implementation: 8 lines of code
- Tests: ~600 lines
- Time: 1 day
- Risk: Minimal

### Phase 3.2 (PartialOrd, Ord, Hash)  
- Implementation: ~180 lines
- Tests: ~550 lines
- Time: 2-3 days
- Risk: Low

### Phase 3.3 (Display, Borrow)
- Implementation: ~90 lines
- Tests: ~240 lines
- Time: 1-2 days
- Risk: Minimal

**Total Phase 3**: ~2,000 lines (270 impl + 1,730 tests)

---

## Checklist for Implementation

### Before Starting Phase 3.1
- [ ] Phase 2 merged and stable
- [ ] All CI/CD passing
- [ ] Mutation score 85%+
- [ ] Coverage 95%+

### During Phase 3.1
- [ ] Implement 4 traits
- [ ] Write unit tests
- [ ] Add property-based tests
- [ ] Documentation with examples
- [ ] Run mutation testing
- [ ] Run coverage analysis

### Before Phase 3.2
- [ ] Phase 3.1 PR merged
- [ ] All tests passing
- [ ] Coverage maintained
- [ ] No new SATD comments

### During Phase 3.2
- [ ] Implement PartialOrd, Ord, Hash
- [ ] Write comprehensive tests
- [ ] Integration tests (Vec::sort, HashMap, BTreeMap)
- [ ] Property tests for ordering
- [ ] Mutation testing
- [ ] Documentation

### Phase 3.3 (Optional)
- [ ] Implement Display
- [ ] Decide on Borrow/BorrowMut
- [ ] Final testing pass
- [ ] Performance verification

---

## Common Pitfalls to Avoid

### For Deref/AsRef
- **Don't**: Create new slice each time
- **Do**: Use existing `.as_slice()` method

### For PartialOrd/Ord
- **Don't**: Implement custom comparison
- **Do**: Delegate to slice's comparison

### For Hash
- **Don't**: Hash only first element
- **Do**: Hash entire slice consistently

### For Tests
- **Don't**: Skip edge cases (empty vector)
- **Do**: Test with multiple types (i32, String, custom structs)

---

## Reference Examples

### Before Phase 3.1
```rust
let vec = TruenoVec::from(vec![1, 2, 3]);
vec.as_slice().len()  // Verbose
vec.as_slice()[0]     // Extra method call
```

### After Phase 3.1
```rust
let vec = TruenoVec::from(vec![1, 2, 3]);
vec.len()             // Deref coercion
vec[0]                // Direct access
```

### After Phase 3.2
```rust
let v1 = TruenoVec::from(vec![1, 2, 3]);
let v2 = TruenoVec::from(vec![1, 2, 4]);

v1 < v2              // PartialOrd
let mut map = HashMap::new();
map.insert(v1, "value");  // Hash + Eq
```

---

## Success = Following This Plan

| Phase | What | How Long | Quality Target |
|-------|------|----------|-----------------|
| 3.1 | 4 ergonomic traits | 1 day | 95%+ cov, 85%+ mut |
| 3.2 | 3 comparison traits | 2-3 days | 95%+ cov, 85%+ mut |
| 3.3 | 2-3 polish traits | 1-2 days | 95%+ cov, 85%+ mut |

**Total**: ~1 week, ~2,000 lines, 250+ tests

---

## When Phase 3 is Complete

TruenoVec will have:
- ✅ Complete ergonomic interfaces (Deref/AsRef)
- ✅ Full comparison support (PartialOrd/Ord)
- ✅ Hash-based collection integration (Hash)
- ✅ User-friendly output (Display)
- ✅ Advanced borrowing patterns (Borrow)
- ✅ 250+ comprehensive tests
- ✅ 95%+ code coverage
- ✅ 85%+ mutation score
- ✅ Production-ready quality

**Status**: Feature-complete collection type matching std::Vec

