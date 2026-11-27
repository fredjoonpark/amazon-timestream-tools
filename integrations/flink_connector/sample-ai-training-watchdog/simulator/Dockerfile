# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM public.ecr.aws/docker/library/python:3.12-slim
RUN useradd --create-home --uid 10001 sim
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY watchdog_sim ./watchdog_sim
# The exact, hash-checked dependency versions in uv.lock (--locked fails if it is out of date
# with pyproject.toml), then the simulator itself. uv is only needed to read the lock.
RUN pip install --no-cache-dir uv==0.12.3 \
 && uv export --locked --no-dev --no-emit-project --no-header --quiet -o /tmp/requirements.txt \
 && pip uninstall --yes uv \
 && pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
 && pip install --no-cache-dir --no-deps . \
 && rm /tmp/requirements.txt
USER sim
ENTRYPOINT ["python", "-m", "watchdog_sim.main"]
CMD ["--sink", "kinesis"]
