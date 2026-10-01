#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { McpServerStack } from '../lib/stack';

const app = new cdk.App();

// Stack name is hardcoded from config
const serverName = 'aprender';
const region = process.env.AWS_REGION || process.env.CDK_DEFAULT_REGION || 'us-east-1';

new McpServerStack(app, `${serverName}-stack`, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: region,
  },
  description: `MCP Server: ${serverName}`,
});

app.synth();
