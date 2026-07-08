import json
import tempfile
import time
import unittest
from pathlib import Path

from train_sae import PeriodicDeltaLmEvaluator


class _LiveProc:
    returncode = None

    def poll(self):
        return None


class EvalWorkerStateTest(unittest.TestCase):
    def _manager(self, td: str):
        root = Path(td)
        mgr = PeriodicDeltaLmEvaluator.__new__(PeriodicDeltaLmEvaluator)
        mgr.log_path = root / "eval_worker.log"
        mgr.log_path.write_text("worker log\n", encoding="utf-8")
        mgr.ready_file = root / "READY.json"
        mgr.fatal_file = root / "FATAL.json"
        mgr.heartbeat_file = root / "HEARTBEAT"
        mgr.stop_file = root / "STOP"
        mgr.eval_startup_timeout_sec = 0.01
        mgr.eval_request_timeout_sec = 0.01
        mgr.pending = {}
        mgr.proc = _LiveProc()
        return mgr

    def test_wait_until_ready_reads_status_file(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = self._manager(td)
            mgr.ready_file.write_text(json.dumps({"status": "ready", "n_eval_texts": 2}), encoding="utf-8")
            mgr._wait_until_ready()

    def test_fatal_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = self._manager(td)
            mgr.fatal_file.write_text(json.dumps({"error": "boom"}), encoding="utf-8")
            with self.assertRaises(RuntimeError) as ctx:
                mgr._check_worker_alive()
            self.assertIn("boom", str(ctx.exception))

    def test_request_timeout_raises(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = self._manager(td)
            mgr.pending = {"step_000001": {"created_at": time.time() - 10}}
            with self.assertRaises(TimeoutError):
                mgr._check_request_timeouts()


if __name__ == "__main__":
    unittest.main()
