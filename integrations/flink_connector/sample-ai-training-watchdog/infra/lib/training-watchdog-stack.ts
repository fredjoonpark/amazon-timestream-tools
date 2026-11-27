// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import * as path from 'node:path';
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecrAssets from 'aws-cdk-lib/aws-ecr-assets';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kinesis from 'aws-cdk-lib/aws-kinesis';
import * as kinesisanalytics from 'aws-cdk-lib/aws-kinesisanalyticsv2';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3assets from 'aws-cdk-lib/aws-s3-assets';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as timestream from 'aws-cdk-lib/aws-timestream';
import * as cr from 'aws-cdk-lib/custom-resources';
import { Construct } from 'constructs';
import { FlinkApplicationLifecycle } from './flink-application-lifecycle';

const INFLUX_PORT = 8181;
const DATABASE = 'watchdog';
const TOKEN_FIELD = 'token';
const APPLICATION_NAME = 'training-watchdog';
const SIM_TELEMETRY_MODES = ['light', 'realistic', 'production'] as const;
type SimTelemetry = (typeof SIM_TELEMETRY_MODES)[number];
/** The simulator's ranks per SFT run when -c simRanks is not set (watchdog_sim.scenarios.DEFAULT_RANKS). */
const SIM_DEFAULT_RANKS: Record<SimTelemetry, number> = { light: 8, realistic: 2, production: 64 };
/**
 * Production profile (watchdog_sim.scenarios): per rank, 32 layers x 8 LoRA matrices of
 * layer_stats and 32 layers x 32 heads of attn_stats, every `detailEvery` steps, in SFT and GRPO;
 * GRPO runs add `promptsPerStep` rl_prompt records a step and 512 eval_question records an
 * evaluation (every 20 steps).
 */
const PROD_DETAIL_PER_RANK = 32 * 8 + 32 * 32;
/**
 * Events per step of one simulated run (README, Load; simulator/tests/test_telemetry.py and
 * test_production.py): SFT runs by telemetry mode and ranks; GRPO runs the same in light and
 * realistic.
 */
const simSftEventsPerStep = (telemetry: SimTelemetry, ranks: number, detailEvery: number) =>
  telemetry === 'production'
    ? 1.2 * ranks + (ranks * PROD_DETAIL_PER_RANK) / detailEvery + 0.06
    : telemetry === 'realistic' ? 225.2 * ranks + 0.06 : 1.2 * ranks + 2.46;
const simGrpoEventsPerStep = (telemetry: SimTelemetry, ranks: number, detailEvery: number, promptsPerStep: number) =>
  telemetry === 'production'
    ? 1.07 + promptsPerStep + (ranks * PROD_DETAIL_PER_RANK) / detailEvery + 512 / 20
    : 1.07;
/**
 * SFT runs in a simulator task of `runs` runs: each task assigns scenarios round-robin (healthy,
 * bad_lr, healthy_grpo, reward_hacking), so 10 runs are 6 SFT and 4 GRPO.
 */
const simSftRuns = (runs: number) => 2 * Math.floor(runs / 4) + Math.min(runs % 4, 2);
/** A comma-separated setting's entries, trimmed, without empty ones. */
const splitList = (value: string) => value.split(',').map((entry) => entry.trim()).filter((entry) => entry !== '');

export interface TrainingWatchdogStackProps extends cdk.StackProps {
  readonly flinkJarPath: string;
  /** Directory containing simulator/, grafana/, and infra/. */
  readonly repoRoot: string;
}

/**
 * Trainer (Trainium) and simulator (Fargate) -> Kinesis -> Managed Flink -> Timestream for
 * InfluxDB 3 Enterprise -> Grafana (Fargate). InfluxDB is also the control plane: Flink writes
 * each incident, with its action, to the `incident` table, and every trainer polls that table
 * for its own run through the cluster's reader endpoint.
 * Nothing is publicly reachable: InfluxDB is private, and Grafana is reached through an
 * ECS Exec port-forwarding session.
 */
export class TrainingWatchdogStack extends cdk.Stack {
  public constructor(scope: Construct, id: string, props: TrainingWatchdogStackProps) {
    super(scope, id, props);

    // ------------------------------------------------------------------ network
    // trn1.2xlarge is offered only in some Availability Zones, and zone names map to different
    // zones in each account, so with the Trainium host the deployer names the VPC's two zones
    // (-c availabilityZones=a,b, or DEPLOY_AZS): the host goes in the first, or in -c trainiumAz.
    // Keep them the same on every deploy; changing them replaces the VPC's subnets.
    const withTrainium =
      this.node.tryGetContext('trainium') !== 'false' && this.node.tryGetContext('trainium') !== false;
    const azContext: string | undefined = this.node.tryGetContext('availabilityZones');
    const availabilityZones = azContext ? splitList(azContext) : undefined;
    if (withTrainium && !availabilityZones) {
      throw new Error(
        'Name two Availability Zones, one offering trn1.2xlarge, with DEPLOY_AZS=<az-a>,<az-b> in .env or ' +
          "-c availabilityZones=<az-a>,<az-b> (or deploy with -c trainium=false). List the zones offering it: aws ec2 " +
          'describe-instance-type-offerings --location-type availability-zone --filters ' +
          "Name=instance-type,Values=trn1.2xlarge --query 'InstanceTypeOfferings[].Location'",
      );
    }
    const vpc = new ec2.Vpc(this, 'Vpc', {
      ...(availabilityZones ? { availabilityZones } : { maxAzs: 2 }),
      natGateways: 1,
      subnetConfiguration: [
        { name: 'public', subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        { name: 'private', subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 22 },
      ],
    });
    const privateSubnets = vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS });
    // Required by InfluxDB 3 in private subnets; also keeps S3 traffic off the NAT gateway.
    const s3Endpoint = vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
      subnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
    });

    const securityGroup = (name: string, description: string) =>
      new ec2.SecurityGroup(this, name, { vpc, description, allowAllOutbound: true });
    const influxSg = securityGroup('InfluxDbSecurityGroup', 'InfluxDB 3 cluster; clients listed explicitly');
    const flinkSg = securityGroup('FlinkSecurityGroup', 'Managed Flink application; no ingress');
    const grafanaSg = securityGroup('GrafanaSecurityGroup', 'Grafana task; no ingress (ECS Exec port forwarding)');
    const simulatorSg = securityGroup('SimulatorSecurityGroup', 'Simulator task; no ingress');
    const dbInitSg = securityGroup('DatabaseInitSecurityGroup', 'Database bootstrap Lambda; no ingress');
    // The Trainium host's group is created here (not with the host) so InfluxDB can name it.
    const trainiumSg = securityGroup('TrainiumSecurityGroup', 'Trainium host; no ingress (SSM only)');
    for (const [client, label] of [
      [flinkSg, 'Flink'],
      [grafanaSg, 'Grafana'],
      [dbInitSg, 'database bootstrap'],
      [simulatorSg, 'simulator control polls'],
      [trainiumSg, 'Trainium trainer control polls'],
    ] as const) {
      influxSg.addIngressRule(client, ec2.Port.tcp(INFLUX_PORT), `InfluxDB 3 from ${label}`);
    }

    // ------------------------------------------------------------------ InfluxDB 3 Enterprise
    // 3 nodes: ingest+query (Flink's writes, writer endpoint), query-only (control polls and
    // Grafana, reader endpoint), dedicated compactor. The roles live in a parameter group that
    // scripts/deploy.sh creates with the CLI (-c influxParameterGroupId): CloudFormation can't
    // provision AWS::Timestream::InfluxDBParameterGroup, and parameter groups can't be deleted.
    const influxParameterGroupId: string | undefined = this.node.tryGetContext('influxParameterGroupId');
    if (!influxParameterGroupId) {
      throw new Error('Set -c influxParameterGroupId=<id> (scripts/deploy.sh creates the parameter group and passes it)');
    }
    const influx = new timestream.CfnInfluxDBCluster(this, 'InfluxDbEnterpriseCluster', {
      name: 'training-watchdog-ent',
      dbInstanceType: 'db.influx.medium',
      dbParameterGroupIdentifier: influxParameterGroupId,
      networkType: 'IPV4',
      port: INFLUX_PORT,
      publiclyAccessible: false,
      vpcSubnetIds: privateSubnets.subnetIds,
      vpcSecurityGroupIds: [influxSg.securityGroupId],
    });
    // The cluster validates the endpoint route at creation time.
    influx.node.addDependency(s3Endpoint);
    influx.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);
    const influxEndpoint = `https://${influx.attrEndpoint}:${INFLUX_PORT}`;
    const influxReaderEndpoint = `https://${influx.attrReaderEndpoint}:${INFLUX_PORT}`;
    const influxSecret = secretsmanager.Secret.fromSecretCompleteArn(
      this,
      'InfluxDbAuthSecret',
      influx.attrInfluxAuthParametersSecretArn,
    );
    // A read-only and a write-only token for the watchdog database, created by the bootstrap below.
    // Everything that only reads (Grafana, trainer and simulator control polls) uses the first, and
    // Flink writes with the second; only the bootstrap holds the admin token.
    const readTokenSecret = new secretsmanager.Secret(this, 'InfluxDbReadTokenSecret', {
      description: `Read-only InfluxDB 3 token for the ${DATABASE} database (written by the database bootstrap)`,
    });
    readTokenSecret.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);
    const writeTokenSecret = new secretsmanager.Secret(this, 'InfluxDbWriteTokenSecret', {
      description: `Write-only InfluxDB 3 token for the ${DATABASE} database (written by the database bootstrap)`,
    });
    writeTokenSecret.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);

    const databaseInit = new lambda.Function(this, 'DatabaseInit', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'handler.on_event',
      code: lambda.Code.fromAsset(path.join(__dirname, '..', 'lambda', 'influxdb-database')),
      // Up to six InfluxDB calls in a row, each allowed 30 s (handler.py), plus Secrets Manager.
      timeout: cdk.Duration.minutes(5),
      vpc,
      vpcSubnets: privateSubnets,
      securityGroups: [dbInitSg],
      logGroup: new logs.LogGroup(this, 'DatabaseInitLogs', {
        retention: logs.RetentionDays.ONE_WEEK,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      }),
      description: `Creates the ${DATABASE} InfluxDB 3 database, its measurements, and its read and write tokens`,
    });
    influxSecret.grantRead(databaseInit);
    for (const secret of [readTokenSecret, writeTokenSecret]) {
      secret.grantRead(databaseInit);
      secret.grantWrite(databaseInit);
    }
    const databaseProvider = new cr.Provider(this, 'DatabaseProvider', {
      onEventHandler: databaseInit,
      // Without its own log group, the provider's Lambda logs to one CloudFormation retains.
      logGroup: new logs.LogGroup(this, 'DatabaseProviderLogs', {
        retention: logs.RetentionDays.ONE_WEEK,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      }),
    });
    const database = new cdk.CustomResource(this, 'Database', {
      serviceToken: databaseProvider.serviceToken,
      resourceType: 'Custom::InfluxDBDatabase',
      properties: {
        Endpoint: influxEndpoint,
        ReaderEndpoint: influxReaderEndpoint,
        SecretArn: influx.attrInfluxAuthParametersSecretArn,
        SecretJsonField: TOKEN_FIELD,
        ReadTokenSecretArn: readTokenSecret.secretArn,
        WriteTokenSecretArn: writeTokenSecret.secretArn,
        Database: DATABASE,
        // Bump to rerun the bootstrap (database, every measurement, tokens) on deploy.
        SchemaVersion: '8',
      },
    });

    // ------------------------------------------------------------------ load: simulated fleet
    // Context flags that size the simulated fleet and the pipeline under it (README, Load). With
    // none of them set: one simulator task of simRuns light runs, max(4, ceil(simRuns / 80))
    // shards, Flink parallelism 1.
    //   -c simRuns=N           runs in the fleet (default 4, one per scenario)
    //   -c simTelemetry=M      light (default), realistic, or production: WATCHDOG_SIM_TELEMETRY
    //   -c simRanks=R          ranks per SFT run: WATCHDOG_SIM_RANKS (default: the simulator's,
    //                          8 light, 2 realistic, 64 production)
    //   -c simStepSeconds=S    seconds per simulated step: STEP_SECONDS (default 1); rates scale 1/S
    //   -c simMaxSteps=N       steps per simulated run before it restarts under a new run ID:
    //                          SIM_MAX_STEPS (default 360). Shorter runs churn run IDs, so distinct
    //                          series accumulate faster at the same ingest rate.
    //   -c simDetailEvery=N    production: steps between each rank's per-layer and per-head stats,
    //                          SIM_DETAIL_EVERY (default 10)
    //   -c simPromptsPerStep=N production: GRPO prompts a step, SIM_PROMPTS_PER_STEP (default 128)
    //   -c simRunsPerTask=K    ceil(N / K) tasks of SIM_RUNS = ceil(N / tasks) <= K runs each, so the
    //                          total rounds up to a multiple of the task count (N=45, K=10: 5 x 9;
    //                          N=41: 5 x 9 = 45, with a synth warning). Each task tags its run IDs
    //                          with its ECS task ID (SIM_TASKS). A task sends one PutRecords call at
    //                          a time unless it has several senders (SIM_SENDERS): realistic tasks
    //                          get one per ~1,000 events/s, at most 8, each run on one sender.
    //   -c kinesisShards=N     shard count (default automatic, below)
    //   -c flinkParallelism=P  Managed Flink parallelism (default 1); parallelismPerKpu stays 1
    const contextNumber = (key: string, integer = true): number | undefined => {
      const raw = this.node.tryGetContext(key);
      if (raw === undefined || raw === null || raw === '') return undefined;
      const value = Number(raw);
      if (!Number.isFinite(value) || value <= 0 || (integer && !Number.isInteger(value))) {
        throw new Error(`-c ${key} must be a positive ${integer ? 'integer' : 'number'}, not '${raw}'`);
      }
      return value;
    };
    const simRuns = contextNumber('simRuns') ?? 4;
    const simTelemetryContext: string | undefined = this.node.tryGetContext('simTelemetry');
    if (simTelemetryContext !== undefined && !(SIM_TELEMETRY_MODES as readonly string[]).includes(simTelemetryContext)) {
      throw new Error(`-c simTelemetry must be one of ${SIM_TELEMETRY_MODES.join(', ')}, not '${simTelemetryContext}'`);
    }
    const simTelemetry = (simTelemetryContext ?? 'light') as SimTelemetry;
    const simRanksContext = contextNumber('simRanks');
    const simRanks = simRanksContext ?? SIM_DEFAULT_RANKS[simTelemetry];
    const simStepSecondsContext = contextNumber('simStepSeconds', false);
    const simStepSeconds = simStepSecondsContext ?? 1;
    const simMaxStepsContext = contextNumber('simMaxSteps');
    const simDetailEveryContext = contextNumber('simDetailEvery');
    const simDetailEvery = simDetailEveryContext ?? 10;
    const simPromptsPerStepContext = contextNumber('simPromptsPerStep');
    const simPromptsPerStep = simPromptsPerStepContext ?? 128;
    const simRunsPerTaskContext = contextNumber('simRunsPerTask');
    const simTasks = simRunsPerTaskContext ? Math.ceil(simRuns / simRunsPerTaskContext) : 1;
    const simRunsPerTask = Math.ceil(simRuns / simTasks);
    if (simTasks * simRunsPerTask !== simRuns) {
      cdk.Annotations.of(this).addWarningV2(
        'watchdog:simRunsRoundedUp',
        `-c simRuns=${simRuns} -c simRunsPerTask=${simRunsPerTaskContext}: the fleet runs ` +
          `${simTasks} tasks x ${simRunsPerTask} = ${simTasks * simRunsPerTask} runs (every task runs the same count)`,
      );
    }
    const simTaskSftRuns = simSftRuns(simRunsPerTask);
    const sftEventsPerStep = simSftEventsPerStep(simTelemetry, simRanks, simDetailEvery);
    const simTaskEventsPerSecond =
      (simTaskSftRuns * sftEventsPerStep +
        (simRunsPerTask - simTaskSftRuns) * simGrpoEventsPerStep(simTelemetry, simRanks, simDetailEvery, simPromptsPerStep)) /
      simStepSeconds;
    // Kinesis takes 1,000 records/s per shard, and SFT runs send nearly every record, so the
    // automatic count is max(4, ceil(SFT runs / SFT runs per shard)): light SFT runs up to about
    // 480 records/s a shard (40 runs at 8 ranks), or 2 realistic ones at 2 ranks (about 900
    // records/s), scaled by the step time and the ranks. At the fleet's half-SFT mix and the default
    // ranks that is one shard per 80 light or 4 realistic runs; counting SFT runs matters for small
    // tasks (4 tasks of 10 runs: 24 SFT runs, 12 shards, where 40 / 4 = 10 shards would each take
    // about 1,080 records/s).
    const sftRunsPerShard = (simTelemetry === 'realistic' ? 4 / simRanks : 482.4 / sftEventsPerStep) * simStepSeconds;
    // Production: SFT and GRPO runs alike send thousands of records a step, so size by the total
    // rate, about 900 records/s a shard.
    const autoShards =
      simTelemetry === 'production'
        ? Math.max(4, Math.ceil((simTasks * simTaskEventsPerSecond) / 900 - 1e-9))
        : Math.max(4, Math.ceil((simTasks * simTaskSftRuns) / sftRunsPerShard - 1e-9));
    const shardCount = contextNumber('kinesisShards') ?? autoShards;
    const flinkParallelism = contextNumber('flinkParallelism') ?? 1;

    // ------------------------------------------------------------------ Kinesis
    const telemetry = new kinesis.Stream(this, 'TelemetryStream', {
      // At least 4 shards: a real SFT run writes about 1,300 points/s (per-module layer stats).
      shardCount,
      encryption: kinesis.StreamEncryption.MANAGED,
      retentionPeriod: cdk.Duration.hours(24),
    });
    telemetry.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);

    // ------------------------------------------------------------------ Managed Flink
    const jar = new s3assets.Asset(this, 'FlinkApplicationJar', { path: props.flinkJarPath });
    const flinkRole = new iam.Role(this, 'FlinkExecutionRole', {
      assumedBy: new iam.ServicePrincipal('kinesisanalytics.amazonaws.com'),
      description: 'Execution role for the AI Training Watchdog Managed Flink application',
    });
    // Only the JAR's own object. Asset.grantRead would grant read on the whole CDK bootstrap
    // bucket, which holds every stack's assets in the account and Region.
    flinkRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['s3:GetObject', 's3:GetObjectVersion'],
        resources: [jar.bucket.arnForObjects(jar.s3ObjectKey)],
      }),
    );
    telemetry.grantRead(flinkRole);
    writeTokenSecret.grantRead(flinkRole);
    flinkRole.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'ec2:CreateNetworkInterface',
          'ec2:CreateNetworkInterfacePermission',
          'ec2:DeleteNetworkInterface',
          'ec2:DescribeDhcpOptions',
          'ec2:DescribeNetworkInterfaces',
          'ec2:DescribeSecurityGroups',
          'ec2:DescribeSubnets',
          'ec2:DescribeVpcs',
        ],
        resources: ['*'],
      }),
    );
    const flinkLogGroup = new logs.LogGroup(this, 'FlinkLogGroup', {
      logGroupName: `/aws/kinesis-analytics/${APPLICATION_NAME}`,
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const flinkLogStream = new logs.LogStream(this, 'FlinkLogStream', {
      logGroup: flinkLogGroup,
      logStreamName: 'application',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const flinkLogStreamArn = this.formatArn({
      service: 'logs',
      resource: 'log-group',
      resourceName: `${flinkLogGroup.logGroupName}:log-stream:${flinkLogStream.logStreamName}`,
      arnFormat: cdk.ArnFormat.COLON_RESOURCE_NAME,
    });
    flinkRole.addToPolicy(new iam.PolicyStatement({ actions: ['logs:DescribeLogGroups'], resources: ['*'] }));
    flinkRole.addToPolicy(
      new iam.PolicyStatement({ actions: ['logs:DescribeLogStreams'], resources: [flinkLogGroup.logGroupArn] }),
    );
    flinkRole.addToPolicy(new iam.PolicyStatement({ actions: ['logs:PutLogEvents'], resources: [flinkLogStreamArn] }));

    // Snapshots keep detector state (moving averages, open incidents, Kinesis positions) across
    // application stops, starts, and updates. -c flinkRestore=skip starts without state, e.g.
    // when a state change in the job is incompatible with the latest snapshot, or when a stateful
    // operator was removed or renamed: allowNonRestoredState is false, so a snapshot that can't be
    // restored in full fails the start instead of silently dropping that state.
    const restoreType =
      this.node.tryGetContext('flinkRestore') === 'skip' ? 'SKIP_RESTORE_FROM_SNAPSHOT' : 'RESTORE_FROM_LATEST_SNAPSHOT';
    const application = new kinesisanalytics.CfnApplication(this, 'FlinkApplication', {
      applicationName: APPLICATION_NAME,
      applicationDescription: 'Detects training failures and reward hacking; writes points and incidents to InfluxDB 3',
      applicationMode: 'STREAMING',
      runtimeEnvironment: 'FLINK-2_3',
      serviceExecutionRole: flinkRole.roleArn,
      runConfiguration: {
        applicationRestoreConfiguration: { applicationRestoreType: restoreType },
        flinkRunConfiguration: { allowNonRestoredState: false },
      },
      applicationConfiguration: {
        applicationCodeConfiguration: {
          codeContentType: 'ZIPFILE',
          codeContent: { s3ContentLocation: { bucketArn: jar.bucket.bucketArn, fileKey: jar.s3ObjectKey } },
        },
        applicationSnapshotConfiguration: { snapshotsEnabled: true },
        environmentProperties: {
          propertyGroups: [
            {
              propertyGroupId: 'Telemetry',
              propertyMap: { streamArn: telemetry.streamArn, region: this.region, initialPosition: 'LATEST' },
            },
            {
              propertyGroupId: 'InfluxDB3',
              propertyMap: {
                endpoint: influxEndpoint,
                database: DATABASE,
                secretId: writeTokenSecret.secretArn,
                secretJsonField: TOKEN_FIELD,
                region: this.region,
              },
            },
          ],
        },
        flinkApplicationConfiguration: {
          checkpointConfiguration: {
            configurationType: 'CUSTOM',
            checkpointingEnabled: true,
            checkpointInterval: 30_000,
          },
          monitoringConfiguration: { configurationType: 'CUSTOM', logLevel: 'INFO', metricsLevel: 'APPLICATION' },
          // -c flinkParallelism=P (default 1): P KPUs. Changing it restarts the job from a snapshot.
          parallelismConfiguration: {
            configurationType: 'CUSTOM',
            autoScalingEnabled: false,
            parallelism: flinkParallelism,
            parallelismPerKpu: 1,
          },
        },
        vpcConfigurations: [
          { subnetIds: privateSubnets.subnetIds, securityGroupIds: [flinkSg.securityGroupId] },
        ],
      },
    });
    // The write token exists before the application is created or updated to use it.
    application.node.addDependency(flinkRole, database);
    const flinkLogging = new kinesisanalytics.CfnApplicationCloudWatchLoggingOption(this, 'FlinkLogging', {
      applicationName: application.ref,
      cloudWatchLoggingOption: { logStreamArn: flinkLogStreamArn },
    });
    const lifecycle = new FlinkApplicationLifecycle(this, 'FlinkApplicationLifecycle', {
      applicationArn: kinesisanalytics.CfnApplication.arnForApplication(application),
      applicationName: application.ref,
      codePath: path.join(__dirname, '..', 'lambda', 'flink-application-lifecycle'),
      deploymentFingerprint: cdk.Fn.join(':', [jar.assetHash, influxEndpoint, telemetry.streamArn]),
      restoreType,
    });
    lifecycle.resource.node.addDependency(flinkLogging, database);

    // ------------------------------------------------------------------ ECS services
    // ECS Exec (Grafana's Session Manager tunnel) logs its sessions to a group of their own. Without
    // one, CDK grants the task role logs:PutLogEvents on every log group in the account, including
    // the trainer's, whose lines demo.sh reads to judge each run.
    const cluster = new ecs.Cluster(this, 'Cluster', {
      vpc,
      executeCommandConfiguration: {
        logging: ecs.ExecuteCommandLogging.OVERRIDE,
        logConfiguration: {
          cloudWatchLogGroup: new logs.LogGroup(this, 'ExecSessionLogs', {
            retention: logs.RetentionDays.ONE_WEEK,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
          }),
        },
      },
    });
    const taskLogs = (name: string) =>
      ecs.LogDrivers.awsLogs({
        streamPrefix: name,
        logGroup: new logs.LogGroup(this, `${name}Logs`, {
          retention: logs.RetentionDays.ONE_WEEK,
          removalPolicy: cdk.RemovalPolicy.DESTROY,
        }),
      });
    const arm64 = { cpuArchitecture: ecs.CpuArchitecture.ARM64, operatingSystemFamily: ecs.OperatingSystemFamily.LINUX };

    // A large fleet serializes and sends thousands of events per second: one Python thread builds,
    // serializes, and signs every event, so 1 vCPU, or 2 above 3,000 events/s per task, with room
    // for the sender threads' TLS and the control poller. Light tasks of up to 160 events/s (24 runs
    // at 8 ranks and one step per second) get 0.25 vCPU. One sender per ~1,000 events/s, at most 8.
    const [simulatorCpu, simulatorMemory] =
      simTaskEventsPerSecond > 3000
        ? [2048, 4096]
        : simTelemetry !== 'light' || simTaskEventsPerSecond > 160 ? [1024, 2048] : [256, 512];
    const simSenders = Math.min(8, simRunsPerTask, Math.max(1, Math.ceil(simTaskEventsPerSecond / 1000)));
    if ([simTelemetryContext, simRanksContext, simStepSecondsContext, simRunsPerTaskContext, simMaxStepsContext,
      simDetailEveryContext, simPromptsPerStepContext,
      this.node.tryGetContext('kinesisShards'), this.node.tryGetContext('flinkParallelism')].some((v) => v !== undefined)) {
      cdk.Annotations.of(this).addInfo(
        `Load: ${simTasks} simulator task(s) x ${simRunsPerTask} ${simTelemetry} runs (${simRanks} ranks, ` +
          `${simStepSeconds} s/step, about ${Math.round(simTaskEventsPerSecond)} events/s per task), ` +
          `${simulatorCpu} CPU / ${simulatorMemory} MiB, ${simSenders} sender(s); ${shardCount} Kinesis shards; ` +
          `Flink parallelism ${flinkParallelism}`,
      );
    }
    const simulatorTask = new ecs.FargateTaskDefinition(this, 'SimulatorTask', {
      cpu: simulatorCpu,
      memoryLimitMiB: simulatorMemory,
      runtimePlatform: arm64,
    });
    simulatorTask.addContainer('simulator', {
      image: ecs.ContainerImage.fromAsset(path.join(props.repoRoot, 'simulator'), {
        platform: ecrAssets.Platform.LINUX_ARM64,
      }),
      environment: {
        AWS_REGION: this.region,
        TELEMETRY_STREAM: telemetry.streamName,
        SIM_RUNS: String(simRunsPerTask),
        // Control plane: each simulated run polls the incident table for its own run.
        INFLUX_URL: influxReaderEndpoint,
        INFLUX_DB: DATABASE,
        // Load flags, set only when given, so that without them the task definition is unchanged.
        // scripts/simulator.sh on restores SIM_TASKS tasks.
        ...(simTelemetryContext !== undefined ? { WATCHDOG_SIM_TELEMETRY: simTelemetry } : {}),
        ...(simRanksContext !== undefined ? { WATCHDOG_SIM_RANKS: String(simRanks) } : {}),
        ...(simStepSecondsContext !== undefined ? { STEP_SECONDS: String(simStepSeconds) } : {}),
        ...(simMaxStepsContext !== undefined ? { SIM_MAX_STEPS: String(simMaxStepsContext) } : {}),
        ...(simDetailEveryContext !== undefined ? { SIM_DETAIL_EVERY: String(simDetailEvery) } : {}),
        ...(simPromptsPerStepContext !== undefined ? { SIM_PROMPTS_PER_STEP: String(simPromptsPerStep) } : {}),
        ...(simRunsPerTaskContext !== undefined ? { SIM_TASKS: String(simTasks) } : {}),
        ...(simSenders > 1 ? { SIM_SENDERS: String(simSenders) } : {}),
      },
      secrets: { INFLUX_TOKEN: ecs.Secret.fromSecretsManager(readTokenSecret, TOKEN_FIELD) },
      logging: taskLogs('Simulator'),
    });
    telemetry.grantWrite(simulatorTask.taskRole);
    // -c simulator=false scales the simulator to zero, for clean real-Trainium sessions.
    const simulatorContext = this.node.tryGetContext('simulator');
    const simulatorEnabled = simulatorContext !== 'false' && simulatorContext !== false;
    const simulator = new ecs.FargateService(this, 'SimulatorService', {
      cluster,
      taskDefinition: simulatorTask,
      desiredCount: simulatorEnabled ? simTasks : 0,
      assignPublicIp: false,
      vpcSubnets: privateSubnets,
      securityGroups: [simulatorSg],
      // Never more tasks than desired, even during a deployment, so the load never exceeds the
      // configured fleet. Tasks emit distinct runs: run IDs carry the task's start time and, with
      // several tasks (-c simRunsPerTask), its ECS task ID.
      minHealthyPercent: 0,
      maxHealthyPercent: 100,
      circuitBreaker: { rollback: true },
    });
    // Flink reads from LATEST, so start producing after Flink runs (and the read token exists).
    simulator.node.addDependency(lifecycle.resource);

    const grafanaAdmin = new secretsmanager.Secret(this, 'GrafanaAdminPassword', {
      description: 'Grafana admin password for the AI Training Watchdog demo',
      generateSecretString: { passwordLength: 24, excludePunctuation: true },
    });
    grafanaAdmin.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);
    const grafanaTask = new ecs.FargateTaskDefinition(this, 'GrafanaTask', {
      cpu: 512,
      memoryLimitMiB: 1024,
      runtimePlatform: arm64,
    });
    grafanaTask.addContainer('grafana', {
      image: ecs.ContainerImage.fromAsset(path.join(props.repoRoot, 'grafana'), {
        platform: ecrAssets.Platform.LINUX_ARM64,
      }),
      // Grafana only reads: the reader endpoint (query-only node) and the read-only token.
      environment: { INFLUX_URL: influxReaderEndpoint, INFLUX_DB: DATABASE, AWS_REGION: this.region },
      secrets: {
        INFLUX_TOKEN: ecs.Secret.fromSecretsManager(readTokenSecret, TOKEN_FIELD),
        GF_SECURITY_ADMIN_PASSWORD: ecs.Secret.fromSecretsManager(grafanaAdmin),
      },
      logging: taskLogs('Grafana'),
    });
    // Read-only CloudWatch access for the dashboard's data-plane row: the InfluxDB connector's
    // write metrics (calls/s, latency percentiles) that Managed Flink publishes to CloudWatch.
    grafanaTask.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: ['cloudwatch:GetMetricData', 'cloudwatch:ListMetrics', 'cloudwatch:GetMetricStatistics'],
        resources: ['*'], // these CloudWatch read actions do not support resource-level scoping
      }),
    );
    const grafana = new ecs.FargateService(this, 'GrafanaService', {
      cluster,
      taskDefinition: grafanaTask,
      desiredCount: 1,
      assignPublicIp: false,
      vpcSubnets: privateSubnets,
      securityGroups: [grafanaSg],
      enableExecuteCommand: true,
      minHealthyPercent: 0,
      maxHealthyPercent: 100,
      circuitBreaker: { rollback: true },
    });
    grafana.node.addDependency(database);

    // ------------------------------------------------------------------ Trainium host
    // A trn1.2xlarge for real training runs (scripts/train.sh). Private subnet, no ingress, managed
    // through SSM only. Disable with -c trainium=false.
    if (withTrainium) {
      const trainerCode = new s3assets.Asset(this, 'TrainerCode', {
        path: path.join(props.repoRoot, 'trainer'),
        exclude: ['tests', '**/__pycache__', '*.egg-info', '.venv'],
      });
      const trainerLogs = new logs.LogGroup(this, 'TrainerLogs', {
        retention: logs.RetentionDays.ONE_WEEK,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      });
      const trainiumRole = new iam.Role(this, 'TrainiumHostRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        description: 'AI Training Watchdog Trainium host: SSM, telemetry, control reads, trainer code',
        managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore')],
      });
      telemetry.grantWrite(trainiumRole);
      readTokenSecret.grantRead(trainiumRole);
      // Only the trainer zip (train.sh copies this one key), not the whole bootstrap bucket.
      trainiumRole.addToPolicy(
        new iam.PolicyStatement({
          actions: ['s3:GetObject'],
          resources: [trainerCode.bucket.arnForObjects(trainerCode.s3ObjectKey)],
        }),
      );
      trainerLogs.grantWrite(trainiumRole);
      trainiumRole.addToPolicy(
        new iam.PolicyStatement({ actions: ['logs:DescribeLogGroups'], resources: ['*'] }),
      );
      const trainiumAz: string | undefined =
        this.node.tryGetContext('trainiumAz')?.trim() || availabilityZones?.[0];
      if (trainiumAz && availabilityZones && !availabilityZones.includes(trainiumAz)) {
        throw new Error(`trainiumAz ${trainiumAz} is not one of the VPC's zones, ${availabilityZones.join(', ')}`);
      }
      // The host's AMI: -c trainiumAmi (or DEPLOY_TRAINIUM_AMI), which scripts/deploy.sh sets to the
      // running host's AMI, so a new DLAMI release never replaces the host and its compile cache,
      // checkpoints, and saved adapters. 'latest' takes the newest DLAMI, replacing the host. Unset
      // (the first deploy, or plain cdk deploy), the lookup is cached in cdk.context.json, which git
      // ignores, so that cache holds only for this checkout. Ubuntu, not AL2023: the AL2023 DLAMI's
      // torch-xla 2.9 needs glibc 2.35 and AL2023 has 2.34.
      const trainiumAmi: string | undefined = this.node.tryGetContext('trainiumAmi')?.trim() || undefined;
      if (trainiumAmi && trainiumAmi !== 'latest' && !/^ami-[0-9a-f]{8,17}$/.test(trainiumAmi)) {
        throw new Error(`-c trainiumAmi (DEPLOY_TRAINIUM_AMI) must be an AMI ID or 'latest', not '${trainiumAmi}'`);
      }
      const dlami = '/aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id';
      const machineImage =
        trainiumAmi && trainiumAmi !== 'latest'
          ? { getImage: () => ({ imageId: trainiumAmi, osType: ec2.OperatingSystemType.LINUX, userData: ec2.UserData.forLinux() }) }
          : ec2.MachineImage.fromSsmParameter(dlami, {
              os: ec2.OperatingSystemType.LINUX,
              cachedInContext: trainiumAmi === undefined,
            });
      const userData = ec2.UserData.forLinux();
      userData.addCommands('mkdir -p /opt/watchdog/code /opt/watchdog/logs /opt/watchdog/checkpoints');
      const trainium = new ec2.Instance(this, 'TrainiumHost', {
        vpc,
        vpcSubnets: {
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          ...(trainiumAz ? { availabilityZones: [trainiumAz] } : {}),
        },
        instanceType: new ec2.InstanceType('trn1.2xlarge'),
        machineImage,
        securityGroup: trainiumSg,
        role: trainiumRole,
        requireImdsv2: true,
        userData,
        blockDevices: [
          {
            deviceName: '/dev/sda1',
            volume: ec2.BlockDeviceVolume.ebs(200, { volumeType: ec2.EbsDeviceVolumeType.GP3, encrypted: true }),
          },
        ],
      });
      cdk.Tags.of(trainium).add('Name', 'training-watchdog-trainium');
      new cdk.CfnOutput(this, 'TrainiumInstanceId', { value: trainium.instanceId });
      new cdk.CfnOutput(this, 'TrainerCodeS3Url', { value: trainerCode.s3ObjectUrl });
      new cdk.CfnOutput(this, 'TrainerLogGroupName', { value: trainerLogs.logGroupName });
    }

    // ------------------------------------------------------------------ outputs
    new cdk.CfnOutput(this, 'EcsClusterName', { value: cluster.clusterName });
    // For one-shot simulated runs (scripts/simulator.sh once <scenario>): an ECS RunTask of the
    // simulator task definition with a command override, in the private subnets. RunTask needs
    // iam:PassRole on the task's roles, so run it with credentials that allow it.
    new cdk.CfnOutput(this, 'SimulatorTaskDefinitionArn', { value: simulatorTask.taskDefinitionArn });
    new cdk.CfnOutput(this, 'SimulatorSecurityGroupId', { value: simulatorSg.securityGroupId });
    new cdk.CfnOutput(this, 'PrivateSubnetIds', { value: cdk.Fn.join(',', privateSubnets.subnetIds) });
    new cdk.CfnOutput(this, 'SimulatorLogGroupName', {
      value: (this.node.findChild('SimulatorLogs') as logs.LogGroup).logGroupName,
    });
    new cdk.CfnOutput(this, 'GrafanaServiceName', { value: grafana.serviceName });
    new cdk.CfnOutput(this, 'GrafanaAdminSecretArn', { value: grafanaAdmin.secretArn });
    new cdk.CfnOutput(this, 'FlinkApplicationName', { value: application.ref });
    new cdk.CfnOutput(this, 'InfluxDbEndpoint', { value: influxEndpoint });
    new cdk.CfnOutput(this, 'InfluxDbReaderEndpoint', { value: influxReaderEndpoint });
    new cdk.CfnOutput(this, 'InfluxDbClusterId', { value: influx.ref });
    new cdk.CfnOutput(this, 'InfluxDbDatabase', { value: DATABASE });
    new cdk.CfnOutput(this, 'InfluxDbAuthSecretArn', { value: influx.attrInfluxAuthParametersSecretArn });
    new cdk.CfnOutput(this, 'InfluxDbReadTokenSecretArn', { value: readTokenSecret.secretArn });
    new cdk.CfnOutput(this, 'TelemetryStreamName', { value: telemetry.streamName });
  }
}
