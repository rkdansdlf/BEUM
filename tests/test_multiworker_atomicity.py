"""
Tests for SQLite multi-worker sequence atomicity and concurrency control.
Verifies that BEGIN IMMEDIATE + conditional UPSERT in SQLite completely prevents
TOCTOU race conditions under high concurrent load.
"""

from __future__ import annotations

import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from receiver_server import ReceiverApp


class MultiWorkerAtomicityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.receiver = ReceiverApp(data_dir=self.data_dir, token="test-token")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_concurrent_identical_sequence_atomicity(self) -> None:
        """
        Simulate 50 concurrent worker threads attempting to accept the EXACT same
        (vehicle_code='TRUCK-ATOMIC', sequence=1) packet at the exact same millisecond.
        Must result in EXACTLY 1 ACCEPT and 49 DUPLICATE REJECTIONS.
        """
        vehicle = "TRUCK-ATOMIC"
        target_seq = 1

        def attempt_update(seq: int) -> tuple[bool, str | None]:
            return self.receiver._update_vehicle_telemetry_atomic(
                vehicle_code=vehicle,
                sequence=seq,
                timestamp=1725514800.0,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(attempt_update, target_seq) for _ in range(50)]
            results = [f.result() for f in futures]

        accepted = [r for r in results if r[0] is True]
        rejected = [r for r in results if r[0] is False]

        self.assertEqual(len(accepted), 1, "Exactly ONE request must be accepted")
        self.assertEqual(len(rejected), 49, "All remaining 49 concurrent requests must be rejected")
        for rej in rejected:
            self.assertEqual(rej[1], "duplicate", "Rejection reason must be 'duplicate'")

        # Verify DB state
        with self.receiver._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT last_sequence, last_timestamp FROM vehicle_telemetry_states WHERE vehicle_code = ?",
                (vehicle,),
            )
            row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], target_seq)

    def test_concurrent_interleaved_sequences(self) -> None:
        """
        Simulate concurrent arrival of newer sequences (10..15) and older sequences (1..9).
        Verifies that older sequences are never accepted once a newer sequence is committed.
        """
        vehicle = "CAR-INTERLEAVED"

        # First commit sequence 10
        ok, reason = self.receiver._update_vehicle_telemetry_atomic(vehicle, 10, 100.0)
        self.assertTrue(ok)

        # Now send 20 requests with seq < 10 and 10 requests with seq = 10
        def try_seq(seq: int) -> tuple[bool, str | None]:
            return self.receiver._update_vehicle_telemetry_atomic(vehicle, seq, 100.0 + seq)

        older_seqs = [5] * 20
        duplicate_seqs = [10] * 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            old_futures = [executor.submit(try_seq, s) for s in older_seqs]
            dup_futures = [executor.submit(try_seq, s) for s in duplicate_seqs]

            old_results = [f.result() for f in old_futures]
            dup_results = [f.result() for f in dup_futures]

        # None of the older sequences should be accepted
        for ok, reason in old_results:
            self.assertFalse(ok)
            self.assertEqual(reason, "out_of_order")

        # None of the duplicate sequences should be accepted
        for ok, reason in dup_results:
            self.assertFalse(ok)
            self.assertEqual(reason, "duplicate")

        # Now send sequence 11
        ok_next, _ = self.receiver._update_vehicle_telemetry_atomic(vehicle, 11, 120.0)
        self.assertTrue(ok_next)


if __name__ == "__main__":
    unittest.main()
