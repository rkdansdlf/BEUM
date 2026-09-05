from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from starlette.testclient import TestClient
from receiver_server import create_app
from gully_system.uploader import HttpUploader


class ReceiverServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.token = 'test-secret-token'
        self.app = create_app(data_dir=self.data_dir, token=self.token)
        self.client = TestClient(self.app)
        self.app_no_auth = create_app(data_dir=self.data_dir, token=None)
        self.client_no_auth = TestClient(self.app_no_auth)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_health_check(self) -> None:
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'ok')
        self.assertTrue(data['auth_required'])

    def test_upload_json_success(self) -> None:
        payload = {
            'event_id': 'evt-json-001',
            'event_type': 'gully_blockage',
            'source': 'pi5-curb-cam',
            'created_at': '2026-09-03T07:00:00Z',
            'blockage': {'status': 'blocked', 'coverage_percent': 85.5},
            'policy': {'mode': 'high', 'roi_profile': 'expanded'},
            'sensors': {'battery_pct': 82.0, 'rain_level': 2},
            'gps': {'lat': 37.5665, 'lon': 126.9780, 'speed': 1.2},
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        response = self.client.post('/upload', json=payload, headers=headers)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['event_id'], 'evt-json-001')
        self.assertFalse(data['image_stored'])

        events_res = self.client.get('/events')
        self.assertEqual(events_res.status_code, 200)
        events_data = events_res.json()
        self.assertEqual(events_data['total'], 1)
        item = events_data['events'][0]
        self.assertEqual(item['event_id'], 'evt-json-001')
        self.assertEqual(item['blockage_status'], 'blocked')
        self.assertEqual(item['coverage_percent'], 85.5)
        self.assertEqual(item['gps']['lat'], 37.5665)
        self.assertFalse(item['has_image'])

    def test_upload_multipart_with_image_success(self) -> None:
        payload = {
            'event_id': 'evt-img-002',
            'event_type': 'gully_blockage',
            'source': 'pi5-camera-rear',
            'created_at': '2026-09-03T07:05:00Z',
            'blockage': {'status': 'warning', 'coverage_percent': 55.0},
        }
        image_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb'
        files = {
            'metadata': (None, json.dumps(payload), 'application/json'),
            'image': ('evidence.jpg', image_bytes, 'image/jpeg'),
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        response = self.client.post('/upload', files=files, headers=headers)
        self.assertEqual(response.status_code, 201)
        res_data = response.json()
        self.assertEqual(res_data['status'], 'success')
        self.assertEqual(res_data['event_id'], 'evt-img-002')
        self.assertTrue(res_data['image_stored'])

        img_res = self.client.get('/events/evt-img-002/image')
        self.assertEqual(img_res.status_code, 200)
        self.assertEqual(img_res.content, image_bytes)
        self.assertEqual(img_res.headers['content-type'], 'image/jpeg')

    def test_auth_failure_missing_and_invalid_token(self) -> None:
        payload = {'event_type': 'test'}
        res1 = self.client.post('/upload', json=payload)
        self.assertEqual(res1.status_code, 401)
        res2 = self.client.post('/upload', json=payload, headers={'Authorization': 'Bearer wrong-token'})
        self.assertEqual(res2.status_code, 401)

    def test_no_auth_mode(self) -> None:
        payload = {'event_type': 'test', 'event_id': 'no-auth-01'}
        res = self.client_no_auth.post('/upload', json=payload)
        self.assertEqual(res.status_code, 201)

    def test_invalid_payload_error_handling(self) -> None:
        headers = {'Authorization': f'Bearer {self.token}'}
        files = {'metadata': (None, 'not-valid-json', 'application/json')}
        res = self.client.post('/upload', files=files, headers=headers)
        self.assertEqual(res.status_code, 400)

    def test_get_event_detail_and_not_found(self) -> None:
        res = self.client.get('/events/non-existent-event')
        self.assertEqual(res.status_code, 404)

    def test_e2e_http_uploader_integration(self) -> None:
        import uvicorn
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        config = uvicorn.Config(self.app, host='127.0.0.1', port=port, log_level='warning')
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.05)
        try:
            uploader = HttpUploader(f'http://127.0.0.1:{port}/upload', token=self.token, timeout_s=5.0)
            event1 = {'event_id': 'e2e-json', 'event_type': 'gully_blockage', 'source': 'pi'}
            try:
                uploader.upload(event1)
            except RuntimeError as exc:
                if 'Operation not permitted' in str(exc) or 'Errno 1' in str(exc):
                    self.skipTest('Sandbox restricts local socket HTTP connections')
                raise

            evidence_file = self.data_dir / 'e2e_sample.jpg'
            evidence_file.write_bytes(b'\xff\xd8\xff\xe0JFIF_SAMPLE_IMAGE_DATA')
            event2 = {'event_id': 'e2e-multipart', 'event_type': 'gully_blockage', 'blockage': {'status': 'blocked'}}
            uploader.upload(event2, evidence_path=evidence_file)

            events_res = self.client.get('/events')
            self.assertEqual(events_res.json()['total'], 2)
            img_res = self.client.get('/events/e2e-multipart/image')
            self.assertEqual(img_res.status_code, 200)
            self.assertEqual(img_res.content, b'\xff\xd8\xff\xe0JFIF_SAMPLE_IMAGE_DATA')
        finally:
            server.should_exit = True
            thread.join(timeout=2.0)

    def test_idempotent_duplicate_upload(self) -> None:
        payload = {
            'event_id': 'evt-idempotent-001',
            'event_type': 'gully_blockage',
            'blockage': {'status': 'warning', 'coverage_percent': 30.0},
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        res1 = self.client.post('/upload', json=payload, headers=headers)
        self.assertEqual(res1.status_code, 201)

        payload['blockage']['coverage_percent'] = 55.0
        res2 = self.client.post('/upload', json=payload, headers=headers)
        self.assertEqual(res2.status_code, 201)

        events = self.client.get('/events').json()
        self.assertEqual(events['total'], 1)
        self.assertEqual(events['events'][0]['event_id'], 'evt-idempotent-001')
        self.assertEqual(events['events'][0]['coverage_percent'], 55.0)

    def test_upload_fallback_event_id_generation(self) -> None:
        payload = {
            'event_type': 'gully_blockage',
            'source': 'pi5-curb-cam',
            'sensors': {'timestamp': 1725350400.0},
            'blockage': {'status': 'blocked', 'coverage_percent': 90.0},
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        res1 = self.client.post('/upload', json=payload, headers=headers)
        self.assertEqual(res1.status_code, 201)
        expected_id = 'pi5-curb-cam_1725350400'
        self.assertEqual(res1.json()['event_id'], expected_id)

        res2 = self.client.post('/upload', json=payload, headers=headers)
        self.assertEqual(res2.status_code, 201)
        self.assertEqual(res2.json()['event_id'], expected_id)

        events = self.client.get('/events').json()
        self.assertEqual(events['total'], 1)


if __name__ == '__main__':
    unittest.main()
