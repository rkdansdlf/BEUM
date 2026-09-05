"""Tests for automatic ROI scaling and multi-resolution camera support."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from gully_system.camera import CameraSource
from gully_system.config import SystemConfig
from gully_system.roi import ROI, is_normalized, scale_points
from gully_system.runtime import GullyRuntime


class TestROIScaling:
    def test_is_normalized(self) -> None:
        assert is_normalized(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))) is True
        assert is_normalized(((0.1, 0.2), (0.8, 0.2), (0.9, 0.9), (0.2, 0.9))) is True
        assert is_normalized(((100.0, 400.0), (540.0, 400.0), (600.0, 480.0), (40.0, 480.0))) is False
        assert is_normalized(()) is False
        assert is_normalized(((-0.1, 0.5), (1.0, 1.0))) is False
        assert is_normalized(((0.0, 0.0), (1.5, 1.0))) is False

    def test_roi_scale_to_resolution_normalized(self) -> None:
        norm_points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        roi = ROI(norm_points)
        assert roi.is_normalized() is True

        scaled_1080x1920 = roi.scale_to_resolution(1080, 1920)
        assert scaled_1080x1920.points == (
            (0.0, 0.0),
            (1080.0, 0.0),
            (1080.0, 1920.0),
            (0.0, 1920.0),
        )

        scaled_640x480 = roi.scale_to_resolution(640, 480)
        assert scaled_640x480.points == (
            (0.0, 0.0),
            (640.0, 0.0),
            (640.0, 480.0),
            (0.0, 480.0),
        )

    def test_roi_scale_to_resolution_pixel_auto_detection(self) -> None:
        # 640x480 base points
        base_640 = ((100.0, 400.0), (540.0, 400.0), (600.0, 480.0), (40.0, 480.0))
        roi = ROI(base_640)
        assert roi.is_normalized() is False

        # Scale to 1280x960 (2x)
        scaled_2x = roi.scale_to_resolution(1280, 960)
        assert scaled_2x.points == (
            (200.0, 800.0),
            (1080.0, 800.0),
            (1200.0, 960.0),
            (80.0, 960.0),
        )

        # Scale to 1080x1920 (vertical video)
        scaled_vert = roi.scale_to_resolution(1080, 1920)
        scale_x = 1080.0 / 640.0
        scale_y = 1920.0 / 480.0
        assert scaled_vert.points[0] == (100.0 * scale_x, 400.0 * scale_y)
        assert scaled_vert.points[1] == (540.0 * scale_x, 400.0 * scale_y)
        assert scaled_vert.points[2] == (600.0 * scale_x, 1920.0)
        assert scaled_vert.points[3] == (40.0 * scale_x, 1920.0)

    def test_roi_scale_to_resolution_with_base_resolution(self) -> None:
        points = ((500.0, 500.0), (1000.0, 1000.0))
        roi = ROI(points)

        scaled = roi.scale_to_resolution(2000, 3000, base_resolution=(1000, 1000))
        assert scaled.points == (
            (1000.0, 1500.0),
            (2000.0, 3000.0),
        )

    def test_roi_scale_clamping(self) -> None:
        # Points exceeding bounds
        points = ((-50.0, -20.0), (800.0, 600.0))
        roi = ROI(points)
        scaled = roi.scale_to_resolution(640, 480, base_resolution=(640, 480))
        assert scaled.points == (
            (0.0, 0.0),
            (640.0, 480.0),
        )

    def test_system_config_roi_base_resolution(self) -> None:
        data = {
            "source": "test.mp4",
            "roi_points": [[0, 0], [100, 100]],
            "roi_base_resolution": [1920, 1080],
        }
        config = SystemConfig.from_dict(data)
        assert config.roi_base_resolution == (1920, 1080)
        assert config.roi_points == ((0.0, 0.0), (100.0, 100.0))


class TestCameraSourceResolutionDetection:
    def test_camera_source_detects_frame_size_on_open(self) -> None:
        camera = CameraSource(source="0")
        assert camera.frame_size is None

        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        # cv2.CAP_PROP_FRAME_WIDTH = 3, cv2.CAP_PROP_FRAME_HEIGHT = 4, CAP_PROP_FPS = 5
        def get_prop(prop_id: int) -> float:
            if prop_id == 3:  # FRAME_WIDTH
                return 1920.0
            if prop_id == 4:  # FRAME_HEIGHT
                return 1080.0
            if prop_id == 5:  # FPS
                return 30.0
            return 0.0

        mock_capture.get.side_effect = get_prop

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            camera.open()

        assert camera.width == 1920
        assert camera.height == 1080
        assert camera.frame_size == (1920, 1080)
        assert camera.fps == 30.0

    def test_camera_source_fallback_on_read(self) -> None:
        camera = CameraSource(source="0")
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        # properties return 0
        mock_capture.get.return_value = 0.0

        # read returns frame of size (720, 1280, 3) (height, width, channels)
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_capture.read.return_value = (True, fake_frame)

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            camera.open()
            assert camera.width is None
            assert camera.height is None

            ok, frame = camera.read()
            assert ok is True
            assert camera.width == 1280
            assert camera.height == 720
            assert camera.frame_size == (1280, 720)


class TestGullyRuntimeROIScaling:
    def test_runtime_updates_roi_on_resolution(self) -> None:
        config = SystemConfig(
            source="0",
            roi_points=((100.0, 400.0), (540.0, 400.0), (600.0, 480.0), (40.0, 480.0)),
            roi_base_resolution=(640, 480),
            run_preflight=False,
        )
        mock_detector = MagicMock()
        runtime = GullyRuntime(config, detector=mock_detector)

        # Initial ROI is 640x480
        assert runtime.roi.points[0] == (100.0, 400.0)

        # Update for 1280x960
        runtime.update_roi_for_resolution(1280, 960)
        assert runtime.roi.points[0] == (200.0, 800.0)
        assert runtime.expanded_roi.points != runtime.roi.points

    def test_runtime_run_uses_detected_camera_resolution(self) -> None:
        config = SystemConfig(
            source="test.mp4",
            output_path="out.mp4",
            roi_points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            run_preflight=False,
        )
        mock_detector = MagicMock()
        runtime = GullyRuntime(config, detector=mock_detector)

        mock_cv2 = MagicMock()
        mock_writer = MagicMock()
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 0x34504D

        fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Mock camera behavior
        runtime.camera.open = MagicMock()
        runtime.camera.width = 1920
        runtime.camera.height = 1080
        runtime.camera.fps = 25.0
        runtime.camera.read = MagicMock(side_effect=[(True, fake_frame), (False, None)])
        runtime.detector.predict = MagicMock(return_value=[])

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = runtime.run(max_frames=1)

        # Camera was opened
        runtime.camera.open.assert_called_once()
        # ROI was scaled to 1920x1080
        assert runtime.roi.points == (
            (0.0, 0.0),
            (1920.0, 0.0),
            (1920.0, 1080.0),
            (0.0, 1080.0),
        )
        # VideoWriter was initialized with (1920, 1080) rather than hardcoded (1080, 1920)
        mock_cv2.VideoWriter.assert_called_once_with(
            "out.mp4",
            0x34504D,
            25,
            (1920, 1080),
        )
