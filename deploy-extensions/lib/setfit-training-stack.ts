import * as cdk from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';

/**
 * Everything the SetFit training MCP server needs that the pmcp.run platform
 * does not provide.
 *
 * The platform owns the MCP REQUEST Lambda (deployed by `cargo pmcp deploy`
 * from `.pmcp/deploy-train.toml`). That function is deliberately small: it
 * mints a task, writes the envelope, and asynchronously invokes the worker
 * defined here. It carries no `apr` binary, no dataset and no encoder — the
 * pmcp.run package ceiling is 250 MB and the platform's own deployment backend
 * has already OOM-killed one oversized bootstrap.
 *
 * This stack owns the parts with a different shape: durable task state, the
 * artifact bucket, and the heavy worker (6 GB, 15 minutes) that actually runs
 * `apr setfit train`. Two functions because their configurations are
 * irreconcilable, not because the code wants splitting.
 *
 * # Why a separate CDK app rather than edits to `deploy/`
 *
 * `deploy/lib/stack.ts` is rendered by cargo-pmcp. Hand edits flip it to
 * `cdk synth`, whose hashed logical IDs read as resource replacements and
 * collide with the platform's fixed names (that is what produced an
 * UPDATE_ROLLBACK_COMPLETE on the predict server). Extensions live in their own
 * app — the same separation chess-mcp uses.
 */
export interface SetFitTrainingStackProps extends cdk.StackProps {
  /**
   * Deployment environment. Every resource name carries it, so dev and prod are
   * addressable by convention and the request Lambda can be handed their names
   * without cross-stack references or a lookup at runtime.
   */
  readonly environment: string;

  /**
   * Directory holding the worker's Lambda package: the `bootstrap` binary plus
   * `assets/` (the aarch64 `apr`, the attested dataset, the pinned encoder).
   * Produced by `just build-trainer-asset`; absent until that has run, which is
   * why `cdk synth` fails loudly rather than deploying an empty function.
   */
  readonly trainerAssetPath: string;

  /**
   * Worker memory in MB. Defaults to {@link TRAINER_MEMORY_MB}.
   *
   * A parameter and not a constant because Lambda's per-function memory ceiling
   * is a PER-ACCOUNT limit that does not appear in Service Quotas and is raised
   * only through an AWS Support case. A fresh account is capped at 3008 MB, and
   * this account is: the first deploy failed with `'MemorySize' value failed to
   * satisfy constraint: Member must have value less than or equal to 3008`.
   *
   * Below {@link TRAINER_MEMORY_MB} the reference train is expected to OOM —
   * see that constant. Setting this lower is a deliberate choice to measure
   * rather than a configuration knob, so it is passed explicitly:
   *
   *   just deploy-training dev 3008
   */
  readonly trainerMemoryMb?: number;
}

/**
 * What the worker needs, from measurement: the 8-shot reference train peaks at
 * **4.0 GB RSS** and runs 127 s wall on an M-series CPU. 6 GB leaves headroom
 * for a slower core and a larger shot count.
 *
 * This is the number the design is justified against. An account whose ceiling
 * is below 4 GB cannot run that train, and no amount of configuration changes
 * that — the fix is an AWS Support case to raise the account's Lambda memory
 * limit, not a smaller number here.
 */
export const TRAINER_MEMORY_MB = 6144;

/** Peak RSS measured for the 8-shot reference train, in MB. */
export const MEASURED_PEAK_RSS_MB = 4096;

export class SetFitTrainingStack extends cdk.Stack {
  public readonly tasksTable: dynamodb.Table;
  public readonly artifactBucket: s3.Bucket;
  public readonly trainer: lambda.Function;

  constructor(scope: Construct, id: string, props: SetFitTrainingStackProps) {
    super(scope, id, props);

    const env = props.environment;
    // Retain data in prod, destroy it in dev: a dev redeploy that leaves an
    // orphaned table behind is a slow-growing bill nobody reads.
    const isProd = env === 'prod';
    const removalPolicy = isProd
      ? cdk.RemovalPolicy.RETAIN
      : cdk.RemovalPolicy.DESTROY;

    // ---------------------------------------------------------------------
    // Task state
    // ---------------------------------------------------------------------
    // Keyed (owner_id, task_id) because that is exactly the scope the
    // TaskStore trait enforces: every read is owner-scoped, and a task
    // belonging to another owner must read as NotFound rather than as a
    // permission error that confirms it exists.
    this.tasksTable = new dynamodb.Table(this, 'TrainingTasksTable', {
      tableName: `aprender-setfit-training-tasks-${env}`,
      partitionKey: { name: 'owner_id', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'task_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      // The store writes an absolute epoch-seconds expiry; DynamoDB reclaims on
      // its own schedule. That deletion is asynchronous, which is why the store
      // ALSO filters expired records on read rather than trusting TTL alone.
      timeToLiveAttribute: 'expires_at',
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: isProd },
      removalPolicy,
    });

    // ---------------------------------------------------------------------
    // Artifacts
    // ---------------------------------------------------------------------
    this.artifactBucket = new s3.Bucket(this, 'ArtifactBucket', {
      bucketName: `aprender-setfit-artifacts-${env}-${this.account}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: isProd,
      removalPolicy,
      autoDeleteObjects: !isProd,
      lifecycleRules: [
        {
          // A trained artifact is ~87 MB and a superseded one has no readers:
          // the task that names it has a 1-hour TTL. Keeping them forever is
          // how an object store quietly becomes the largest line on the bill.
          id: 'expire-training-artifacts',
          expiration: cdk.Duration.days(isProd ? 90 : 7),
        },
      ],
    });

    // ---------------------------------------------------------------------
    // The worker
    // ---------------------------------------------------------------------
    const trainerLogs = new logs.LogGroup(this, 'TrainerLogGroup', {
      logGroupName: `/aws/lambda/aprender-setfit-trainer-${env}`,
      retention: isProd
        ? logs.RetentionDays.THREE_MONTHS
        : logs.RetentionDays.ONE_WEEK,
      removalPolicy,
    });

    // A worker below the measured peak is a deploy that succeeds and then fails
    // every training run with an OOM. That is worth stating at synth time, next
    // to the number, rather than leaving it to be discovered in CloudWatch.
    const memorySize = props.trainerMemoryMb ?? TRAINER_MEMORY_MB;
    if (memorySize < MEASURED_PEAK_RSS_MB) {
      cdk.Annotations.of(this).addWarning(
        `TrainerFunction memory is ${memorySize} MB, below the ${MEASURED_PEAK_RSS_MB} MB ` +
          `peak RSS measured for the 8-shot reference train. The plumbing will work and ` +
          `the training run is expected to OOM — CloudWatch's "Max Memory Used" is what ` +
          `settles it. The fix is an AWS Support case raising this account's Lambda ` +
          `memory limit (not in Service Quotas), then redeploy without the override.`,
      );
    }

    this.trainer = new lambda.Function(this, 'TrainerFunction', {
      functionName: `aprender-setfit-trainer-${env}`,
      // PROVIDED_AL2023 + arm64: the worker is a Rust custom runtime, and the
      // `apr` it spawns is cross-compiled aarch64-unknown-linux-gnu. A mismatch
      // here is an exec-format error at the first submit, not at deploy.
      runtime: lambda.Runtime.PROVIDED_AL2023,
      architecture: lambda.Architecture.ARM_64,
      handler: 'bootstrap',
      code: lambda.Code.fromAsset(props.trainerAssetPath),
      // MEASURED, not guessed — see TRAINER_MEMORY_MB. Overridable only because
      // the per-account ceiling is a support case, not a setting; the warning
      // below fires whenever the deployed value is under the measured peak.
      memorySize,
      // 900 s is Lambda's own ceiling and ~7x the measured 127 s run.
      timeout: cdk.Duration.seconds(900),
      // The trained artifact is ~87 MB and is written to /tmp before upload,
      // alongside the config. 512 MB (the default) would fit today and leaves
      // nothing for a larger model.
      ephemeralStorageSize: cdk.Size.mebibytes(2048),
      logGroup: trainerLogs,
      environment: {
        APRENDER_SETFIT_TASKS_TABLE: this.tasksTable.tableName,
        // NO artifact-bucket variable here, deliberately. The worker does not
        // choose where an artifact goes — it uploads to the `artifact_uri` the
        // request side wrote into the task's envelope, so that URI is the one
        // spelling of the destination and the client is shown the same string.
        // The bucket grant below is what enforces it: an envelope naming some
        // other bucket fails on permissions rather than writing there. A
        // variable here would look like the setting that decides, and would be
        // read by nobody.
        //
        // Paths inside the package, resolved against $LAMBDA_TASK_ROOT by the
        // worker. Named here so the layout is declared in ONE place rather than
        // agreed by convention between the build script and the Rust.
        APRENDER_SETFIT_TRAIN_APR_BIN: '/var/task/assets/apr',
        APRENDER_SETFIT_TRAIN_DATA: '/var/task/assets/data',
        APRENDER_SETFIT_TRAIN_SELECTION:
          '/var/task/assets/data/selection-manifest.json',
        APRENDER_SETFIT_TRAIN_MODEL_DIR: '/var/task/assets/encoder',
        APRENDER_SETFIT_TRAIN_OUTPUT_DIR: '/tmp/setfit-out',
        RUST_LOG: 'info',
      },
      // A retry re-runs a 127-second CPU-saturating job. The terminal write is
      // guarded against a second writer, so a retry cannot corrupt state — but
      // it also cannot help, because a failure here is a bad config or a broken
      // package, not a transient. Failures are visible as a `failed` task.
      retryAttempts: 0,
    });

    this.tasksTable.grantReadWriteData(this.trainer);
    this.artifactBucket.grantWrite(this.trainer);
    // Datasets a client uploaded live under their own prefix, and the worker
    // may only READ them. Scoping the grant to the prefix is what makes an
    // envelope naming any other key fail on permissions before a byte moves.
    this.artifactBucket.grantRead(this.trainer, 'datasets/*');

    // ---------------------------------------------------------------------
    // What the request Lambda needs to know
    // ---------------------------------------------------------------------
    // Published to SSM as well as CfnOutput: the request Lambda is deployed by
    // cargo-pmcp from a different app, so it cannot take a cross-stack
    // reference. SSM is the seam, and the names are predictable per
    // environment so `.pmcp/deploy-train.toml` can name them directly.
    const params: Record<string, string> = {
      'tasks-table': this.tasksTable.tableName,
      'artifact-bucket': this.artifactBucket.bucketName,
      'trainer-function-arn': this.trainer.functionArn,
      'trainer-function-name': this.trainer.functionName,
    };
    for (const [key, value] of Object.entries(params)) {
      new ssm.StringParameter(this, `Param-${key}`, {
        parameterName: `/aprender/setfit-train/${env}/${key}`,
        stringValue: value,
      });
      new cdk.CfnOutput(this, `Out-${key}`, { value, exportName: `aprender-setfit-train-${env}-${key}` });
    }

    // The policy the request Lambda's execution role needs. Published rather
    // than attached: that role belongs to the platform-owned stack, so this
    // stack states the requirement and the operator binds it, instead of
    // reaching into a stack it does not own.
    new cdk.CfnOutput(this, 'RequestLambdaPolicy', {
      description:
        'Attach to the pmcp.run request Lambda role: DynamoDB RW on the tasks table, lambda:InvokeFunction on the trainer, and the S3 rights it PRESIGNS with (a presigned URL carries the signer\'s permissions, so the request function needs PutObject on datasets/ and GetObject on tasks/ even though it never moves a byte itself)',
      value: JSON.stringify({
        Version: '2012-10-17',
        Statement: [
          {
            Effect: 'Allow',
            Action: [
              'dynamodb:GetItem',
              'dynamodb:PutItem',
              'dynamodb:UpdateItem',
              'dynamodb:DeleteItem',
              'dynamodb:Query',
            ],
            Resource: this.tasksTable.tableArn,
          },
          {
            Effect: 'Allow',
            Action: ['lambda:InvokeFunction'],
            Resource: this.trainer.functionArn,
          },
          {
            // The upload slot: the client PUTs with a URL this role signed.
            // HeadObject on the same prefix is how `train` refuses a
            // dataset_uri nothing was uploaded to, synchronously.
            Effect: 'Allow',
            Action: ['s3:PutObject', 's3:GetObject'],
            Resource: `${this.artifactBucket.bucketArn}/datasets/*`,
          },
          {
            // Without ListBucket, S3 answers HeadObject on a MISSING key with
            // 403 rather than 404 — it will not disclose whether the key
            // exists to a caller who could not list it. Measured on the first
            // live negative test: the "nothing uploaded yet" branch was
            // unreachable and the client got an AccessDenied dump instead.
            // Scoped to the prefix, so it discloses nothing about artifacts.
            Effect: 'Allow',
            Action: ['s3:ListBucket'],
            Resource: this.artifactBucket.bucketArn,
            Condition: { StringLike: { 's3:prefix': ['datasets/*'] } },
          },
          {
            // The download link `train_status` mints for a completed artifact.
            Effect: 'Allow',
            Action: ['s3:GetObject'],
            Resource: `${this.artifactBucket.bucketArn}/tasks/*`,
          },
        ],
      }),
    });
  }
}
