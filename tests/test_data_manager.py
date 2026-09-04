from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from gully_system.data_manager import StorageQueue

class StorageQueueTest(unittest.TestCase):
    def test_enqueue_and_mark_sent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = StorageQueue(Path(directory) / 'spool')
            event_id = queue.enqueue({'hello': 'world'}, evidence_bytes=b'image-bytes')
            self.assertEqual(queue.pending_count(), 1)
            paths = queue.pending_paths()
            self.assertEqual(len(paths), 1)
            event = queue.load(paths[0])
            self.assertEqual(event['event_id'], event_id)
            ev = queue.evidence_path(paths[0], event)
            self.assertIsNotNone(ev)
            self.assertTrue(ev.exists())
            queue.mark_sent(paths[0])
            self.assertEqual(queue.pending_count(), 0)
            self.assertFalse(ev.exists())

    def test_limit_prunes_oldest_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = StorageQueue(Path(directory) / 'spool', max_bytes=350)
            queue.enqueue({'value': 'a' * 120})
            queue.enqueue({'value': 'b' * 120})
            self.assertLessEqual(queue.usage_bytes(), 350)
            self.assertLess(queue.pending_count(), 2)

if __name__ == '__main__':
    unittest.main()
