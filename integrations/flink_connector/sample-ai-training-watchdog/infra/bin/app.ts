#!/usr/bin/env node
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as cdk from 'aws-cdk-lib';
import { TrainingWatchdogStack } from '../lib/training-watchdog-stack';

const app = new cdk.App();

const jarPath = path.resolve(
  __dirname,
  '..',
  app.node.tryGetContext('flinkJarPath') ??
    '../flink-job/target/training-watchdog-flink-0.1.0-SNAPSHOT-all.jar',
);
if (!fs.existsSync(jarPath)) {
  throw new Error(`Flink application JAR not found: ${jarPath}; run 'mvn package' in flink-job/`);
}

// Settings from the environment (or .env, which the scripts load), unless given with -c: the
// Availability Zones, which must stay the same on every deploy (see the stack's network section),
// and the Trainium host's AMI (see the stack's Trainium host section).
for (const [key, variable] of [
  ['availabilityZones', 'DEPLOY_AZS'],
  ['trainiumAz', 'DEPLOY_TRAINIUM_AZ'],
  ['trainiumAmi', 'DEPLOY_TRAINIUM_AMI'],
]) {
  const value = process.env[variable];
  if (value && app.node.tryGetContext(key) === undefined) {
    app.node.setContext(key, value);
  }
}

new TrainingWatchdogStack(app, 'TrainingWatchdogStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  flinkJarPath: jarPath,
  repoRoot: path.resolve(__dirname, '..', '..'),
});
cdk.Tags.of(app).add('project', 'training-watchdog');
// Optional tags for every resource, such as tags your account's policies require:
// DEPLOY_TAGS='Key1=value1,Key2=value2'. scripts/preflight.sh checks the Trainium host has them.
for (const pair of (process.env.DEPLOY_TAGS ?? '').split(',').map((p) => p.trim()).filter((p) => p !== '')) {
  const eq = pair.indexOf('=');
  const key = pair.slice(0, eq).trim();
  if (eq <= 0 || key === '') {
    throw new Error(`DEPLOY_TAGS entry '${pair}' is not Key=value`);
  }
  cdk.Tags.of(app).add(key, pair.slice(eq + 1).trim());
}
