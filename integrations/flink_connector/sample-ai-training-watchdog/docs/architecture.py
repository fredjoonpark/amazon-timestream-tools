#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The solution architecture in the README, drawn with the `diagrams` package
(https://diagrams.mingrammer.com/) and the AWS icons it ships. Needs Graphviz
(`brew install graphviz`). `diagrams` has no Managed Service for Apache Flink icon (only the
retired Kinesis Data Analytics one), so that icon is
icons/Arch_Amazon-Managed-Service-for-Apache-Flink_64@5x.png, from the AWS Architecture Icons
package of 2026-07-31 (https://aws.amazon.com/architecture/icons/). From the sample's directory:

    python3 -m venv .venv && .venv/bin/pip install diagrams matplotlib
    .venv/bin/python docs/architecture.py    # writes docs/architecture.png

Sized to 800 pixels wide: short service names (allowed in graphics), arrows labeled with step
numbers only (the numbered list under the figure in the README explains each step), and a
compact layout, so the text stays legible at 800 px. You reach Grafana through a Session Manager
port-forwarding session (scripts/grafana-tunnel.sh); nothing in the stack is publicly reachable.
`diagrams` has no Session Manager icon, so that node uses the Systems Manager one. The PNG is
rendered large, then resampled to exactly 800 px wide.
"""

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import KinesisDataStreams
from diagrams.aws.compute import EC2Instance, Fargate
from diagrams.aws.database import Timestream
from diagrams.aws.general import User
from diagrams.aws.management import SystemsManager
from diagrams.aws.security import SecretsManager
from diagrams.custom import Custom
from PIL import Image

HERE = Path(__file__).resolve().parent
FLINK_ICON = HERE / "icons" / "Arch_Amazon-Managed-Service-for-Apache-Flink_64@5x.png"


def ManagedServiceForApacheFlink(label: str, **attrs) -> Custom:  # noqa: N802 (reads like the icon classes)
    return Custom(label, str(FLINK_ICON), **attrs)
OUT = HERE / "architecture"
WIDTH_PX = 800
FONT = "Helvetica"

DATA = {"color": "#232F3E", "penwidth": "2.2", "fontcolor": "#232F3E"}
CONTROL = {"color": "#E7157B", "penwidth": "2.6", "fontcolor": "#E7157B"}

GROUP = {"labeljust": "l", "penwidth": "1.6", "fontsize": "30", "margin": "14"}
AWS_CLOUD = {**GROUP, "bgcolor": "white", "pencolor": "#232F3E", "fontname": f"{FONT} Bold"}
VPC = {**GROUP, "bgcolor": "white", "pencolor": "#8C4FFF", "fontcolor": "#8C4FFF"}
PRIVATE = {**GROUP, "bgcolor": "#E6F6F7", "pencolor": "#00A4A6", "fontcolor": "#007A7C"}
INFLUX = {**GROUP, "bgcolor": "#FFFFFF", "pencolor": "#C925D1", "style": "dashed",
          "fontcolor": "#7A1782", "labeljust": "c"}


def node(cls, label: str):
    """An icon node tall enough for its label at this font size (diagrams sizes nodes for 13 pt)."""
    lines = label.count("\n") + 1
    return cls(label, height=str(round(1.55 + 0.5 * lines, 2)), imagepos="tc")


def step(number: int, look: dict, **extra) -> Edge:
    return Edge(label=f" {number} ", fontname=f"{FONT} Bold", fontsize="34", **{**look, **extra})


with Diagram(
    "",
    filename=str(OUT),
    outformat="png",
    show=False,
    direction="LR",
    graph_attr={"dpi": "200", "pad": "0.25", "nodesep": "0.3", "ranksep": "0.55",
                "fontname": FONT, "splines": "spline", "newrank": "true"},
    node_attr={"fontname": FONT, "fontsize": "28"},
):
    you = node(User, "You")

    with Cluster("AWS Cloud", graph_attr=AWS_CLOUD):
        secrets = node(SecretsManager, "Secrets\nManager")
        session = node(SystemsManager, "Session\nManager")
        kinesis = node(KinesisDataStreams, "Kinesis Data\nStreams")

        with Cluster("VPC", graph_attr=VPC):
            with Cluster("Private subnets", graph_attr=PRIVATE):
                trainer = node(EC2Instance, "Trainer\n(trn1.2xlarge)")
                flink = node(ManagedServiceForApacheFlink, "Managed Service\nfor Apache Flink")
                with Cluster("Timestream for InfluxDB 3", graph_attr=INFLUX):
                    # Graphviz stacks this rank in reverse creation order: compactor on top,
                    # then ingest, then query next to the poll.
                    query = node(Timestream, "Query")
                    ingest = node(Timestream, "Ingest")
                    compactor = node(Timestream, "Compactor")
                grafana = node(Fargate, "Grafana\n(Fargate)")

    trainer >> step(1, DATA) >> kinesis
    kinesis >> step(2, DATA) >> flink
    flink >> step(3, DATA) >> ingest
    query >> step(4, CONTROL, constraint="false") >> trainer
    query >> step(5, DATA) >> grafana
    # The incident that Flink writes to the ingest node reaches the trainer's polls on the query node.
    ingest >> Edge(constraint="false", color=DATA["color"], penwidth="1.6") >> query
    # The connector in the Flink job reads the InfluxDB token from Secrets Manager.
    secrets >> Edge(style="dashed", constraint="false", weight="0", **DATA) >> flink
    # You reach Grafana only through Session Manager, which sits outside the VPC.
    session << Edge(style="dashed", **DATA) << you
    session >> Edge(style="dashed", constraint="false", **DATA) >> grafana

    # Put the three InfluxDB nodes in one rank, so they stack vertically.
    flink >> Edge(style="invis") >> query
    flink >> Edge(style="invis") >> compactor
    # Secrets Manager sits above the InfluxDB cluster, outside the VPC.
    # (Placed from Kinesis, not Flink: an edge from Flink pulls Flink up into arrow 3's path.)
    kinesis >> Edge(style="invis", minlen="2") >> secrets
    # Session Manager sits top right, outside the VPC, in Grafana's rank.
    ingest >> Edge(style="invis") >> session

png = OUT.with_suffix(".png")
with Image.open(png) as image:
    height = round(image.height * WIDTH_PX / image.width)
    image.resize((WIDTH_PX, height), Image.LANCZOS).save(png, optimize=True)
print(png, f"{WIDTH_PX} x {height}")
