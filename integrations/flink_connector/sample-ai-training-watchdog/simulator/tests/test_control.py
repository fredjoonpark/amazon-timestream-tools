# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from watchdog_sim.control import InfluxControlPlane
from watchdog_sim.scenarios import Run

ROW = {"incident_id": "bad_lr-t-001-divergence-1", "type": "DIVERGENCE", "action": "ROLLBACK", "action_arg": "100",
       "attempt": 1, "step": 127, "onset_step": 124, "message": "loss spiked", "time_ns": 1_762_800_000_123_000_000}


class FakeInflux(BaseHTTPRequestHandler):
    """Answers /api/v3/query_sql like InfluxDB 3 (format=json): one incident for one run."""

    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeInflux.requests.append({"path": self.path, "auth": self.headers["Authorization"], "body": body})
        rows = [ROW] if body["params"]["run_id"] == "bad_lr-t-001" else []
        payload = json.dumps(rows).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class ControlPlaneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeInflux)  # loopback only
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeInflux.requests.clear()

    def test_polls_each_run_and_queues_each_incident_once(self):
        control = InfluxControlPlane(self.url, "watchdog", "tok", lambda: ["bad_lr-t-001", "healthy-t-001"],
                                     workers=2, clock=lambda: 1_762_799_060.0)
        control.poll_once()
        control.poll_once()
        self.assertEqual(len(FakeInflux.requests), 4)
        request = FakeInflux.requests[0]
        self.assertEqual(request["path"], "/api/v3/query_sql")
        self.assertEqual(request["auth"], "Bearer tok")
        self.assertEqual(request["body"]["db"], "watchdog")
        # The run ID travels as a parameter, never inside the SQL text.
        self.assertIn("$run_id", request["body"]["q"])
        self.assertNotIn("bad_lr-t-001", request["body"]["q"])
        self.assertIn("to_timestamp_millis(1762799000000)", request["body"]["q"])

        self.assertEqual(len(control.commands), 1)
        command = control.commands[0]
        self.assertEqual(command["run_id"], "bad_lr-t-001")
        self.assertEqual(command["action"], "ROLLBACK")
        self.assertEqual(command["arg"], "100")
        self.assertEqual(command["attempt"], 1)
        self.assertEqual(command["ts"], 1_762_800_000_123)
        stats = control.stats()
        self.assertEqual((stats["polls"], stats["errors"], stats["rows"]), (4.0, 0.0, 2.0))
        self.assertEqual(control.stats()["polls"], 0.0)  # stats reset each window

    def test_each_run_reads_from_when_it_was_first_polled(self):
        runs, now = ["bad_lr-t-001"], [1_762_800_000.0]
        control = InfluxControlPlane(self.url, "watchdog", "tok", lambda: runs, workers=2, clock=lambda: now[0])
        control.poll_once()
        now[0] += 3_600  # an hour later, a new run starts
        runs = ["bad_lr-t-001", "healthy-t-002"]
        control.poll_once()
        windows = {(r["body"]["params"]["run_id"], r["body"]["q"].split("to_timestamp_millis(")[1].split(")")[0])
                   for r in FakeInflux.requests}
        self.assertEqual(windows, {("bad_lr-t-001", "1762799940000"), ("healthy-t-002", "1762803540000")})
        self.assertEqual(len(control.commands), 1)
        # A finished run drops its window and its seen incidents; the others keep theirs.
        runs = ["healthy-t-002"]
        control.poll_once()
        self.assertEqual(control.since, {"healthy-t-002": 1_762_803_540_000})
        self.assertEqual(set(control.seen), {"healthy-t-002"})

    def test_errors_are_counted_not_raised(self):
        control = InfluxControlPlane("http://127.0.0.1:1", "watchdog", "tok", lambda: ["r"])
        control.poll_once()
        self.assertEqual(control.stats()["errors"], 1.0)
        self.assertFalse(control.commands)

    def test_a_bad_row_is_skipped_once_and_polling_continues(self):
        control = InfluxControlPlane(self.url, "watchdog", "tok", lambda: ["r"], workers=1)
        bad = {**ROW, "incident_id": "r-divergence-1", "time_ns": "not a time"}
        good = {**ROW, "incident_id": "r-divergence-2"}
        control._fetch = lambda run_id, since_ms: [bad, "not a row", good]
        with self.assertLogs("watchdog_sim", "ERROR") as logs:
            control.poll_once()
            control.poll_once()
        self.assertEqual(len(logs.records), 1)
        self.assertEqual([c["incident_id"] for c in control.commands], ["r-divergence-2"])
        self.assertEqual(control.stats()["errors"], 1.0)

    def test_the_poller_thread_survives_an_unexpected_failure(self):
        calls = []

        def run_ids():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("unexpected")
            raise SystemExit  # ends run() on the second poll

        control = InfluxControlPlane(self.url, "watchdog", "tok", run_ids, interval=0.0)
        with self.assertLogs("watchdog_sim", "ERROR"), self.assertRaises(SystemExit):
            control.run()
        self.assertEqual(len(calls), 2)

    def test_ack_event_reports_outcome_and_latency(self):
        from watchdog_sim.control import ack_event, incident_command

        run = Run("bad_lr", 1, prefix="t-")
        command = incident_command(run.run_id, ROW, seen_ms=1_762_800_002_123)
        event = ack_event(run, command, "rolled back", now_ms=1_762_800_003_123)
        self.assertEqual(event["type"], "control_ack")
        self.assertEqual(event["metrics"], {"apply_latency_ms": 3000.0, "seen_latency_ms": 2000.0})
        self.assertEqual(event["text"], {"incident_id": ROW["incident_id"], "outcome": "rolled back"})


if __name__ == "__main__":
    unittest.main()
