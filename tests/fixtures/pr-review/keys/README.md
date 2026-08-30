# TEST-ONLY signing key

`pr-review-test-TEST-ONLY.key` is an **unencrypted minisign secret key that signs test
fixtures and nothing else**. It is committed deliberately so
`tests/fixtures/pr-review/build-fixtures.sh` can regenerate and re-sign every fixture,
and so the committed signatures can be verified from a clean checkout with no setup.

It is not a credential. It authenticates nothing, protects nothing, and grants no access.

The **production** key of PR-REVIEW-SKILL-002 v2 §4.3 is a different key entirely: its
secret half lives in CI secrets as `PR_REVIEW_SIGNING_KEY` and its public half belongs at
`.github/pr-review.pub`, which PRREV-005/006 owes. `scripts/check_pr_review_receipt.sh`
defaults to that path and REJECTS every receipt while it is absent — an unverifiable
signature is not a verified one.

Per §4.3: a valid signature proves the receipt came from the signing environment. It does
not prove the review was honest, and the receipt says `"attestation_level": "L1-self"` for
exactly that reason. Conflating provenance with diligence would be the most sophisticated
form of theater this repository has yet produced.
