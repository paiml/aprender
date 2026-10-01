//! The `TaskBackend` contract, run against a REAL DynamoDB.
//!
//! The unit tests in `dynamodb.rs` cover the item mapping, which is the half
//! that can be checked without a network. They cannot touch the half that
//! actually carries correctness: whether a conditional `PutItem` fails the way
//! the store's compare-and-swap retry assumes, whether a consistent read really
//! returns a write from another connection, whether `attribute_not_exists`
//! refuses an overwrite. Those are DynamoDB's semantics, and asserting them
//! against a mock would only assert the mock.
//!
//! So this file drives the same scenarios the in-memory backend is tested
//! against, through `AprenderTaskStore`, against a real table. Same contract,
//! different backend — which is the claim the `TaskBackend` seam makes and the
//! only way to falsify it.
//!
//! Its first ever run earned it: every `PutItem` and `Query` the backend issued
//! was being rejected with `ValidationException`, because the
//! `ExpressionAttributeNames` map defined aliases the expressions did not
//! mention. The item mapping — the part the unit tests cover — was correct. The
//! expression is validated only by the service, so the deployed server would
//! have accepted a `train` call and then failed to record anything, on the
//! first request.
//!
//! # Arming it
//!
//! Skipped unless `APRENDER_SETFIT_E2E_TASKS_TABLE` names a table. Two ways to
//! provide one:
//!
//! ```bash
//! # DynamoDB Local — no AWS account, no cost
//! docker run -d -p 8000:8000 amazon/dynamodb-local
//! AWS_ENDPOINT_URL=http://localhost:8000 \
//! AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local AWS_REGION=us-east-1 \
//! APRENDER_SETFIT_E2E_TASKS_TABLE=setfit-training-tasks-test \
//!   cargo test -p aprender-setfit-train-lambda --test dynamodb_contract
//!
//! # The deployed dev table
//! AWS_PROFILE=ze-kasher-dev AWS_REGION=us-east-1 \
//! APRENDER_SETFIT_E2E_TASKS_TABLE=aprender-setfit-training-tasks-dev \
//!   cargo test -p aprender-setfit-train-lambda --test dynamodb_contract
//! ```
//!
//! Every test scopes itself to a fresh random owner id and writes records with
//! a short TTL, so running this against the dev table leaves nothing behind
//! that another owner can see or that outlives the hour.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap()

use std::sync::Arc;

use aprender_mcp_setfit_train::{AprenderTaskStore, BackendError, StoredTask, TaskBackend};
use aprender_setfit_train_lambda::DynamoDbTaskBackend;
use aws_sdk_dynamodb::types::{
    AttributeDefinition, KeySchemaElement, KeyType, ScalarAttributeType,
};
use pmcp::server::task_store::{TaskStore, TaskStoreError};
use pmcp::types::{CallToolResult, Content, TaskStatus};

const TABLE_ENV: &str = "APRENDER_SETFIT_E2E_TASKS_TABLE";

/// A per-test owner, so concurrent runs and a shared dev table cannot collide,
/// and so nothing this test writes is visible to the real `local` owner.
fn test_owner(label: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("test-{label}-{nanos}")
}

/// Connect. Creates the table ONLY against a local endpoint.
///
/// Returns `None` when the suite is not armed, and every test reports that as a
/// skip rather than a pass — a green run of a suite that connected to nothing
/// is exactly the vacuous gate this repo keeps paying for.
///
/// # Why creation is gated on `AWS_ENDPOINT_URL`
///
/// An earlier revision created the table unconditionally, reasoning that
/// against a deployed one it would simply fail with `ResourceInUseException`.
/// It did not fail — it SUCCEEDED, because the table had not been deployed yet,
/// and left a real `aprender-setfit-training-tasks-dev` that CloudFormation
/// does not own. The next `just deploy-training dev` would then have failed
/// with "already exists", and the fix would have looked like a CDK problem.
///
/// A test may create infrastructure it also destroys, in an endpoint that is
/// entirely its own. It may not create infrastructure in an account where a
/// stack is the owner of that name. So: local endpoint, create; real account,
/// require the deploy to have happened and say so.
async fn backend() -> Option<(Arc<DynamoDbTaskBackend>, String)> {
    let table = std::env::var(TABLE_ENV).ok()?;
    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_dynamodb::Client::new(&config);

    if std::env::var_os("AWS_ENDPOINT_URL").is_some() {
        let key = |name: &str, kind: KeyType| {
            KeySchemaElement::builder()
                .attribute_name(name)
                .key_type(kind)
                .build()
        };
        let attr = |name: &str| {
            AttributeDefinition::builder()
                .attribute_name(name)
                .attribute_type(ScalarAttributeType::S)
                .build()
        };
        let _ = client
            .create_table()
            .table_name(&table)
            .billing_mode(aws_sdk_dynamodb::types::BillingMode::PayPerRequest)
            .set_key_schema(Some(vec![
                key("owner_id", KeyType::Hash).ok()?,
                key("task_id", KeyType::Range).ok()?,
            ]))
            .set_attribute_definitions(Some(vec![attr("owner_id").ok()?, attr("task_id").ok()?]))
            .send()
            .await;
    } else if client
        .describe_table()
        .table_name(&table)
        .send()
        .await
        .is_err()
    {
        // Fail once, here, naming the real cause — rather than eight times with
        // whatever the first write happens to say.
        panic!(
            "table `{table}` is not readable in this account.\n\
             This suite does NOT create tables against a real account: the CDK \
             stack owns that name.\n\
             Deploy it first:  just deploy-training dev\n\
             Or run against DynamoDB Local, which the module doc shows."
        );
    }

    Some((
        Arc::new(DynamoDbTaskBackend::new(client, table.clone())),
        table,
    ))
}

/// Print why a test did nothing. Loud, because a silent skip reads as a pass.
fn skipped(test: &str) {
    eprintln!("SKIP {test}: set {TABLE_ENV} to run the DynamoDB contract (see the module doc)");
}

fn result(text: &str) -> CallToolResult {
    CallToolResult::new(vec![Content::Text {
        text: text.to_string(),
    }])
}

fn record(task_id: &str, version: u64, ttl_secs: u64) -> StoredTask {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    StoredTask {
        task: pmcp::types::tasks::Task::new(task_id, TaskStatus::Working)
            .with_ttl(ttl_secs * 1000)
            .with_timestamps("2026-09-03T00:00:00Z", "2026-09-03T00:00:00Z"),
        result: None,
        envelope: None,
        version,
        expires_at_secs: Some(now + ttl_secs),
    }
}

#[tokio::test]
async fn a_record_survives_the_round_trip_to_a_real_table() {
    let Some((backend, table)) = backend().await else {
        return skipped("a_record_survives_the_round_trip_to_a_real_table");
    };
    let owner = test_owner("roundtrip");
    backend
        .put_new(&owner, record("t-round", 1, 3600))
        .await
        .unwrap_or_else(|e| panic!("put_new against {table}: {e}"));
    let back = backend
        .get(&owner, "t-round")
        .await
        .expect("get")
        .expect("a record just written must be readable");
    assert_eq!(back.task.task_id, "t-round");
    assert_eq!(back.version, 1);
    assert_eq!(back.task.status, TaskStatus::Working);
}

#[tokio::test]
async fn put_new_refuses_to_overwrite_an_existing_record() {
    let Some((backend, _)) = backend().await else {
        return skipped("put_new_refuses_to_overwrite_an_existing_record");
    };
    let owner = test_owner("putnew");
    backend
        .put_new(&owner, record("t-dup", 1, 3600))
        .await
        .expect("first insert");
    let err = backend
        .put_new(&owner, record("t-dup", 9, 3600))
        .await
        .expect_err("a second insert at the same key must not overwrite");
    assert!(
        matches!(err, BackendError::Conflict),
        "attribute_not_exists must surface as Conflict, not as an opaque \
         service error the store cannot retry on: {err:?}"
    );
    // And the original must be intact — an overwrite that reported an error
    // would be worse than one that reported success.
    let back = backend
        .get(&owner, "t-dup")
        .await
        .expect("get")
        .expect("some");
    assert_eq!(back.version, 1);
}

#[tokio::test]
async fn cas_conflicts_on_a_stale_version_and_succeeds_on_the_current_one() {
    let Some((backend, _)) = backend().await else {
        return skipped("cas_conflicts_on_a_stale_version_and_succeeds_on_the_current_one");
    };
    let owner = test_owner("cas");
    backend
        .put_new(&owner, record("t-cas", 1, 3600))
        .await
        .expect("insert");

    let mut next = record("t-cas", 2, 3600);
    next.task.status = TaskStatus::Completed;
    backend
        .cas(&owner, "t-cas", 1, next)
        .await
        .expect("CAS against the current version must succeed");

    // The store's retry policy is built entirely on this being Conflict.
    let err = backend
        .cas(&owner, "t-cas", 1, record("t-cas", 2, 3600))
        .await
        .expect_err("CAS against a stale version must fail");
    assert!(matches!(err, BackendError::Conflict), "{err:?}");

    let back = backend
        .get(&owner, "t-cas")
        .await
        .expect("get")
        .expect("some");
    assert_eq!(back.task.status, TaskStatus::Completed, "the winner stands");
}

#[tokio::test]
async fn cas_against_a_record_that_was_never_written_is_a_conflict() {
    let Some((backend, _)) = backend().await else {
        return skipped("cas_against_a_record_that_was_never_written_is_a_conflict");
    };
    let owner = test_owner("casghost");
    // This is the shape TTL reclamation produces mid-flight. It must NOT
    // resurrect the record: an unconditional write here would recreate a task
    // the client was already told had expired.
    let err = backend
        .cas(&owner, "t-ghost", 1, record("t-ghost", 2, 3600))
        .await
        .expect_err("CAS on a vanished record must fail");
    assert!(matches!(err, BackendError::Conflict), "{err:?}");
    assert!(
        backend.get(&owner, "t-ghost").await.expect("get").is_none(),
        "a failed CAS must not have created the item"
    );
}

#[tokio::test]
async fn the_store_contract_holds_over_dynamodb() {
    let Some((backend, _)) = backend().await else {
        return skipped("the_store_contract_holds_over_dynamodb");
    };
    let owner = test_owner("store");
    let store = AprenderTaskStore::new(backend);

    // The full request-side sequence: mint, stash the envelope, finish.
    let task = store
        .mint_for_request(&owner, Some(3_600_000))
        .await
        .expect("mint");
    store
        .put_envelope(
            &task.task_id,
            &owner,
            serde_json::json!({"config":{"shots":8}}),
        )
        .await
        .expect("put envelope");

    // What the worker reads on the other side of the wire.
    let envelope = store
        .get_envelope(&task.task_id, &owner)
        .await
        .expect("get envelope")
        .expect("some envelope");
    assert_eq!(envelope["config"]["shots"], 8);

    // The envelope must never reach a client.
    let served = store.get(&task.task_id, &owner).await.expect("get");
    let json = serde_json::to_value(&served).expect("serializes");
    assert!(json.get("envelope").is_none(), "{json}");

    // The guarded terminal write, across what would be two processes.
    store
        .finish(
            &task.task_id,
            &owner,
            TaskStatus::Completed,
            result("first"),
        )
        .await
        .expect("terminal write");
    store
        .finish(&task.task_id, &owner, TaskStatus::Failed, result("second"))
        .await
        .expect("a straggler is a no-op SUCCESS");
    let final_task = store.get(&task.task_id, &owner).await.expect("get");
    assert_eq!(
        final_task.status,
        TaskStatus::Completed,
        "the FIRST terminal write wins — this is what makes a retried or raced \
         worker safe, and it has to hold over the real backend's CAS"
    );
    match store
        .get_result(&task.task_id, &owner)
        .await
        .expect("result")
        .content
        .first()
        .expect("content")
    {
        Content::Text { text } => assert_eq!(text, "first"),
        other => panic!("expected text: {other:?}"),
    }
}

#[tokio::test]
async fn another_owner_cannot_read_the_task() {
    let Some((backend, _)) = backend().await else {
        return skipped("another_owner_cannot_read_the_task");
    };
    let owner = test_owner("scope");
    let intruder = test_owner("intruder");
    let store = AprenderTaskStore::new(backend);
    let task = store
        .mint_for_request(&owner, Some(3_600_000))
        .await
        .expect("mint");
    let err = store
        .get(&task.task_id, &intruder)
        .await
        .expect_err("another owner must not read it");
    assert!(
        matches!(err, TaskStoreError::NotFound { .. }),
        "a wrong owner must read as NotFound, never as a permission error that \
         confirms the task exists: {err:?}"
    );
}

#[tokio::test]
async fn an_expired_record_reads_as_gone_before_ttl_reclaims_it() {
    let Some((backend, _)) = backend().await else {
        return skipped("an_expired_record_reads_as_gone_before_ttl_reclaims_it");
    };
    let owner = test_owner("expiry");
    // DynamoDB's own TTL deletion runs on its own schedule — up to 48 hours —
    // so this record is still physically present. Reading it as gone is the
    // backend's filter doing the work, which is precisely why the filter exists
    // and is not redundant with the table's TimeToLiveSpecification.
    let mut expired = record("t-expired", 1, 3600);
    expired.expires_at_secs = Some(1);
    backend.put_new(&owner, expired).await.expect("insert");
    assert!(
        backend
            .get(&owner, "t-expired")
            .await
            .expect("get")
            .is_none(),
        "an expired record must not be served just because TTL has not \
         collected it yet"
    );
    let (listed, _) = backend.list(&owner, None).await.expect("list");
    assert!(
        listed.is_empty(),
        "list must filter expired records too, not only get: {listed:?}"
    );
}

#[tokio::test]
async fn list_returns_only_this_owners_records() {
    let Some((backend, _)) = backend().await else {
        return skipped("list_returns_only_this_owners_records");
    };
    let mine = test_owner("list-mine");
    let theirs = test_owner("list-theirs");
    for id in ["t-a", "t-b"] {
        backend
            .put_new(&mine, record(id, 1, 3600))
            .await
            .expect("insert");
    }
    backend
        .put_new(&theirs, record("t-c", 1, 3600))
        .await
        .expect("insert");
    let (listed, _) = backend.list(&mine, None).await.expect("list");
    let mut ids: Vec<&str> = listed.iter().map(|r| r.task.task_id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        ["t-a", "t-b"],
        "the query is partitioned by owner; another owner's task must not appear"
    );
}
