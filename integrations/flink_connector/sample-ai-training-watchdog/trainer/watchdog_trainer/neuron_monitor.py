# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Forwards neuron-monitor reports as `neuron_hw` events, one per NeuronCore in use."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from typing import Callable

LOG = logging.getLogger(__name__)


def parse_report(report: dict) -> list[dict]:
    """Returns {core, util_pct, mem_used_gb, exec_latency_ms} per NeuronCore in a report."""
    samples: list[dict] = []
    for runtime in report.get("neuron_runtime_data") or []:
        body = runtime.get("report") or {}
        cores = (body.get("neuroncore_counters") or {}).get("neuroncores_in_use") or {}
        memory = (
            ((body.get("memory_used") or {}).get("neuron_runtime_used_bytes") or {})
            .get("usage_breakdown", {})
            .get("neuroncore_memory_usage", {})
        )
        latency = (
            ((body.get("execution_stats") or {}).get("latency_stats") or {}).get("total_latency") or {}
        ).get("p50")
        for core, counters in cores.items():
            sample = {"core": int(core), "util_pct": float(counters.get("neuroncore_utilization", 0.0))}
            core_memory = memory.get(core)
            if isinstance(core_memory, dict):
                sample["mem_used_gb"] = sum(v for v in core_memory.values() if isinstance(v, (int, float))) / 1e9
                for part, value in core_memory.items():
                    if isinstance(value, (int, float)):
                        sample[f"mem_{part}_gb"] = value / 1e9
            if isinstance(latency, (int, float)):
                sample["exec_latency_ms"] = latency * 1000.0
            samples.append(sample)
    return samples


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def parse_extra(report: dict) -> list[dict]:
    """Runtime and host telemetry: [{"kind", "tags", "metrics"}]. Kinds: neuron_runtime (execution
    counts, errors, latency percentiles, runtime vCPU), host_cpu (one per vCPU), host_mem."""
    out: list[dict] = []
    for index, runtime in enumerate(report.get("neuron_runtime_data") or []):
        body = runtime.get("report") or {}
        stats = body.get("execution_stats") or {}
        summary = stats.get("execution_summary") or {}
        errors = stats.get("error_summary") or {}
        latency = stats.get("latency_stats") or {}
        metrics: dict[str, float] = {}
        for key in ("completed", "completed_with_err", "completed_with_num_err", "timed_out"):
            if _num(summary.get(key)) is not None:
                metrics[key] = _num(summary.get(key))
        if errors:
            metrics["errors"] = sum(v for v in errors.values() if isinstance(v, (int, float)))
        for group, prefix in (("total_latency", "latency"), ("device_latency", "device_latency")):
            for pct in ("p50", "p99"):
                value = _num((latency.get(group) or {}).get(pct))
                if value is not None:
                    metrics[f"{prefix}_{pct}_ms"] = value * 1000.0
        vcpu = (body.get("neuron_runtime_vcpu_usage") or {}).get("vcpu_usage") or {}
        for key in ("user", "system"):
            if _num(vcpu.get(key)) is not None:
                metrics[f"vcpu_{key}_pct"] = _num(vcpu.get(key))
        if metrics:
            out.append({"kind": "neuron_runtime", "tags": {"runtime": str(runtime.get("pid", index))}, "metrics": metrics})
    system = report.get("system_data") or {}
    for cpu, usage in ((system.get("vcpu_usage") or {}).get("usage_data") or {}).items():
        metrics = {f"{k}_pct": _num(v) for k, v in (usage or {}).items() if _num(v) is not None}
        if metrics:
            out.append({"kind": "host_cpu", "tags": {"cpu": f"cpu{int(cpu):02d}"}, "metrics": metrics})
    memory = system.get("memory_info") or {}
    if _num(memory.get("memory_used_bytes")) is not None:
        out.append({"kind": "host_mem", "tags": {}, "metrics": {
            "used_gb": memory["memory_used_bytes"] / 1e9,
            "total_gb": (_num(memory.get("memory_total_bytes")) or 0.0) / 1e9,
        }})
    return out


def monitor_config(period: str) -> dict:
    """neuron-monitor config: the per-runtime metric groups parse_report reads, at `period`."""
    return {
        "period": period,
        "neuron_runtimes": [
            {
                "tag_filter": ".*",
                "metrics": [
                    {"type": "neuroncore_counters"},
                    {"type": "memory_used"},
                    {"type": "execution_stats"},
                    {"type": "neuron_runtime_vcpu_usage"},
                ],
            }
        ],
        "system_metrics": [{"type": "vcpu_usage"}, {"type": "memory_info"}],
    }


class NeuronMonitorForwarder(threading.Thread):
    """Runs `neuron-monitor` at `period` and calls `on_sample(sample)` for each per-core sample.
    If neuron-monitor rejects the config, falls back to its default settings (5 s period)."""

    def __init__(self, on_sample: Callable[[dict], None], command: str = "neuron-monitor", period: str = "1s",
                 on_extra: Callable[[dict], None] | None = None) -> None:
        super().__init__(name="neuron-monitor", daemon=True)
        self.on_sample = on_sample
        self.on_extra = on_extra
        self.command = command
        self.period = period
        self.process: subprocess.Popen | None = None
        self.stopped = False

    def _stream(self, args: list[str]) -> int:
        """Forwards reports until the process exits; returns the number of reports parsed."""
        try:
            self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            LOG.warning("%s not found; Neuron hardware telemetry disabled", self.command)
            self.stopped = True
            return 0
        assert self.process.stdout is not None
        reports = 0
        for line in self.process.stdout:
            try:
                report = json.loads(line)
            except ValueError:
                continue
            reports += 1
            for sample in parse_report(report):
                self.on_sample(sample)
            if self.on_extra is not None:
                for extra in parse_extra(report):
                    self.on_extra(extra)
        return reports

    def run(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as config:
            json.dump(monitor_config(self.period), config)
        if self._stream([self.command, "-c", config.name]) == 0 and not self.stopped:
            LOG.warning("neuron-monitor rejected the %s config; using its defaults", self.period)
            self._stream([self.command])

    def stop(self) -> None:
        self.stopped = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
