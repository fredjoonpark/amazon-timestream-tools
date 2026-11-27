# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Executor.run_many when a worker dies. The workers run stand-ins for DataFusion, defined here so
the spawned workers can import them, so these tests need no SQL dependencies."""

import os
import time
import unittest
from unittest import mock

from watchdog_trainer import sqlexec


def _no_tables(anchor_s):
    pass


def _fake_run(sql):
    if sql == "crash":
        os._exit(1)  # the worker dies, as on a segfault or an out-of-memory kill
    if sql == "slow":
        time.sleep(1.0)
    return "ok", [(sql,)]


class WorkerCrashTest(unittest.TestCase):
    def run_many(self, *batches):
        with mock.patch.object(sqlexec, "_init_worker", _no_tables), mock.patch.object(sqlexec, "_run", _fake_run):
            executor = sqlexec.Executor(workers=3)
            try:
                return [executor.run_many(queries) for queries in batches]
            finally:
                executor.close()

    def test_only_a_query_that_kills_its_worker_fails(self):
        # "slow" is still running when the first "crash" breaks the pool; it must not be blamed.
        results, after = self.run_many(["slow", "crash", "a", "crash", "b"], ["c", "d"])
        self.assertEqual(results[0], ("ok", [("slow",)]))
        self.assertEqual([status for status, _ in results], ["ok", "error", "ok", "error", "ok"])
        self.assertTrue(results[1][1].startswith("worker failed"), results[1])
        # The next call runs on a working pool.
        self.assertEqual(after, [("ok", [("c",)]), ("ok", [("d",)])])


if __name__ == "__main__":
    unittest.main()
