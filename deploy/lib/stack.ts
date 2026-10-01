import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

/**
 * MCP Server Stack for pmcp.run deployment
 *
 * This stack deploys only the Lambda function. The API Gateway is managed
 * by the shared pmcp.run infrastructure at https://api.pmcp.run/{serverId}/mcp
 */
export class McpServerStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ========================================================================
    // MCP METADATA (for pmcp.run platform enrichment)
    // Read from CDK context, passed by `cargo pmcp deploy`
    // Platforms read this metadata to provision secrets, add IAM, etc.
    // ========================================================================
    const mcpVersion = this.node.tryGetContext('mcp:version') || '1.0';
    const mcpServerType = this.node.tryGetContext('mcp:serverType') || 'custom';
    const mcpServerId = this.node.tryGetContext('mcp:serverId');
    const mcpTemplateId = this.node.tryGetContext('mcp:templateId');
    const mcpTemplateVersion = this.node.tryGetContext('mcp:templateVersion');
    const mcpResources = this.node.tryGetContext('mcp:resources');
    const mcpCapabilities = this.node.tryGetContext('mcp:capabilities');

    // Set CloudFormation template metadata
    // This is ignored by vanilla CloudFormation but read by pmcp.run
    const metadata: Record<string, any> = {
      'mcp:version': mcpVersion,
      'mcp:serverType': mcpServerType,
    };
    if (mcpServerId) metadata['mcp:serverId'] = mcpServerId;
    if (mcpTemplateId) metadata['mcp:templateId'] = mcpTemplateId;
    if (mcpTemplateVersion) metadata['mcp:templateVersion'] = mcpTemplateVersion;
    if (mcpResources) {
      try {
        metadata['mcp:resources'] = typeof mcpResources === 'string'
          ? JSON.parse(mcpResources)
          : mcpResources;
      } catch (e) {
        metadata['mcp:resources'] = mcpResources;
      }
    }
    if (mcpCapabilities) {
      try {
        metadata['mcp:capabilities'] = typeof mcpCapabilities === 'string'
          ? JSON.parse(mcpCapabilities)
          : mcpCapabilities;
      } catch (e) {
        metadata['mcp:capabilities'] = mcpCapabilities;
      }
    }
    this.templateOptions.metadata = metadata;

    // Get configuration from context or environment
    // These can be overridden via CDK context: -c serverId=myserver
    const serverId = this.node.tryGetContext('serverId') || 'aprender-setfit-predict';
    const organizationId = this.node.tryGetContext('organizationId') || process.env.PMCP_ORGANIZATION_ID || 'default-org';
    const mcpServersTable = this.node.tryGetContext('mcpServersTable') || process.env.MCP_SERVERS_TABLE || 'McpServer';

    // Cost allocation tags — propagate to all resources in this stack
    cdk.Tags.of(this).add('project', 'hosting');
    cdk.Tags.of(this).add('service', serverId);
    cdk.Tags.of(this).add('managed-by', 'pmcp');
    cdk.Tags.of(this).add('target', 'pmcp-run');

    // Lambda function (ARM64 for better price/performance)
    const mcpFunction = new lambda.Function(this, 'McpFunction', {
      functionName: serverId,
      runtime: lambda.Runtime.PROVIDED_AL2023,
      handler: 'bootstrap',
      code: lambda.Code.fromAsset('.build'),
      architecture: lambda.Architecture.ARM_64,
      // OPERATOR-CURATED (do not regenerate): `cargo pmcp deploy init` renders
      // these as hardcoded literals and never reads deploy.toml's memory_mb /
      // timeout_seconds (CLI defect, filed with the SDK team 2026-08-19).
      // Deploys PRESERVE an existing stack.ts, so THESE values are what ships.
      // 2048MB is load-bearing: the 87MB SetFit model OOMs the function at 256.
      memorySize: 2048,
      timeout: cdk.Duration.seconds(60),
      environment: {
        RUST_LOG: 'info',
        // Composition configuration for domain servers calling foundation servers
        PMCP_ORGANIZATION_ID: organizationId,
        PMCP_SERVER_ID: serverId,
        MCP_SERVERS_TABLE: mcpServersTable,
      },
      tracing: lambda.Tracing.ACTIVE,
      // Structured JSON logging so CloudWatch correctly parses log levels
      loggingFormat: lambda.LoggingFormat.JSON,
    });

    // OPERATOR-CURATED: pin ALL logical IDs to the ones the platform's own
    // template renderer emitted when this stack was CREATED (verified against
    // the live template: McpFunction, LogGroup, ExecutionRole,
    // ExecutionRoleDefaultPolicy). A hand-modified stack.ts makes cargo-pmcp
    // fall back to `cdk synth`, whose hashed IDs (McpFunctionF370A1F8, …) read
    // as RESOURCE REPLACEMENTS to CloudFormation — the replacements collide
    // with the existing functionName/logGroupName and the update rolls back
    // (UPDATE_ROLLBACK_COMPLETE, dep_1787188942880_2a3ab5b3, 2026-08-19).
    (mcpFunction.node.defaultChild as cdk.CfnResource).overrideLogicalId('McpFunction');
    if (mcpFunction.role) {
      (mcpFunction.role.node.defaultChild as cdk.CfnResource).overrideLogicalId('ExecutionRole');
      const defaultPolicy = mcpFunction.role.node.tryFindChild('DefaultPolicy');
      if (defaultPolicy) {
        (defaultPolicy.node.defaultChild as cdk.CfnResource).overrideLogicalId('ExecutionRoleDefaultPolicy');
      }
    }

    // Log group with 7-day retention (cost optimization)
    const logGroup = new logs.LogGroup(this, 'LogGroup', {
      logGroupName: `/aws/lambda/${mcpFunction.functionName}`,
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    (logGroup.node.defaultChild as cdk.CfnResource).overrideLogicalId('LogGroup');

    // IAM permissions for domain server composition
    // These permissions allow domain servers to call foundation servers via Lambda
    // 1. Read from DynamoDB McpServer table to discover foundation servers
    mcpFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:Query',
      ],
      resources: [
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${mcpServersTable}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${mcpServersTable}/*`,
      ],
    }));

    // 2. Invoke other Lambda functions (foundation servers)
    mcpFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['lambda:InvokeFunction'],
      resources: [
        `arn:aws:lambda:${this.region}:${this.account}:function:*`,
      ],
    }));

    // Outputs
    new cdk.CfnOutput(this, 'LambdaArn', {
      value: mcpFunction.functionArn,
      description: 'MCP Server Lambda ARN',
    });

    new cdk.CfnOutput(this, 'LambdaName', {
      value: mcpFunction.functionName,
      description: 'MCP Server Lambda Name',
    });

    // ApiUrl output for backward compatibility with pmcp.run workflow
    // The actual URL is constructed from serverId: https://api.pmcp.run/{serverId}/mcp
    // This placeholder is used until pmcp.run workflow is updated to use LambdaArn
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: 'https://api.pmcp.run/{use-deployment-id}/mcp',
      description: 'MCP endpoint (construct from deployment ID)',
    });

    new cdk.CfnOutput(this, 'DashboardUrl', {
      value: `https://console.aws.amazon.com/cloudwatch/home?region=${this.region}`,
      description: 'CloudWatch Console',
    });

    new cdk.CfnOutput(this, 'McpRoleArn', {
      value: mcpFunction.role!.roleArn,
      description: 'MCP Server Lambda execution role ARN (stable export for downstream stacks)',
      exportName: `pmcp-${serverId}-McpRoleArn`,
    });
  }
}
