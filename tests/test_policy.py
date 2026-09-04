from __future__ import annotations
import unittest
from gully_system.config import PolicyConfig
from gully_system.policy import PolicyDecision, RuleBasedPolicy, SafePolicy
from gully_system.sensors import SensorSnapshot

class PolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PolicyConfig()
        self.baseline = RuleBasedPolicy(self.config)

    def test_emergency_rain_uses_high_mode_and_expanded_roi(self) -> None:
        decision = self.baseline.decide(SensorSnapshot(rain_level=2, timestamp=1))
        self.assertEqual(decision.mode, 'high')
        self.assertEqual(decision.roi_profile, 'expanded')

    def test_critical_battery_uses_low_mode(self) -> None:
        decision = self.baseline.decide(SensorSnapshot(battery_pct=5, timestamp=1))
        self.assertEqual(decision.mode, 'low')

    def test_safety_override_beats_failing_policy(self) -> None:
        class FailingPolicy:
            def decide(self, snapshot: SensorSnapshot) -> PolicyDecision:
                raise RuntimeError('bad policy')

        safe = SafePolicy(FailingPolicy(), self.baseline, self.config)
        decision = safe.decide(SensorSnapshot(rain_level=2, timestamp=1))
        self.assertEqual(decision.reason, 'safety override')

if __name__ == '__main__':
    unittest.main()
