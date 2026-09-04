from __future__ import annotations
import tempfile, time, unittest
from pathlib import Path
from gully_system.gps import GPSFix, ReplayGPSProvider, parse_nmea_sentence

class GPSTest(unittest.TestCase):
    def test_parse_rmc_sentence(self) -> None:
        fix = parse_nmea_sentence('$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A')
        self.assertIsNotNone(fix)
        self.assertAlmostEqual(fix.latitude, 48.1173, places=4)
        self.assertAlmostEqual(fix.longitude, 11.516666, places=4)
        self.assertAlmostEqual(fix.speed_mps, 11.5235, places=3)

    def test_invalid_rmc_is_ignored(self) -> None:
        self.assertIsNone(parse_nmea_sentence('$GPRMC,123519,V,,,,,,,'))

    def test_replay_updates_latest_fix_in_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gps.csv'
            path.write_text('latitude,longitude,speed\n37.1,127.1,0\n37.2,127.2,1\n', encoding='utf-8')
            provider = ReplayGPSProvider(path, sample_period_s=0.02, replay_speed=10.0)
            provider.start()
            time.sleep(0.01)
            first = provider.latest()
            time.sleep(0.03)
            second = provider.latest()
            provider.stop()
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertAlmostEqual(second.latitude, 37.2)
            self.assertEqual(second.source, 'csv')

    def test_fix_serializes_age(self) -> None:
        value = GPSFix(1.0, 2.0, received_at=time.time() - 2).to_dict()
        self.assertIn('age_s', value)
        self.assertGreaterEqual(value['age_s'], 2.0)

if __name__ == '__main__':
    unittest.main()
