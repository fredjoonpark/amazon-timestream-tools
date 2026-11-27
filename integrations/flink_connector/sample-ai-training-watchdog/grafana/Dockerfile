# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM grafana/grafana-oss:12.4.3
ENV GF_AUTH_ANONYMOUS_ENABLED=false \
    GF_USERS_ALLOW_SIGN_UP=false \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_SERVER_HTTP_ADDR=127.0.0.1 \
    GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/watchdog.json
COPY provisioning /etc/grafana/provisioning
COPY dashboards /var/lib/grafana/dashboards
