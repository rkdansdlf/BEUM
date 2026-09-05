"""Camera capture abstraction for PiCamera2, USB camera, or video file."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CameraSource:
    source: str = "0"
    width: int | None = None
    height: int | None = None
    realtime: bool = False
    buffer_size: int = 1

    def __post_init__(self) -> None:
        self._use_picamera2 = str(self.source).lower() == "picamera2"
        try:
            self._source: int | str = int(self.source)
        except ValueError:
            self._source = self.source
        self._capture: Any | None = None
        self._picamera: Any | None = None
        self._replay_queue: queue.Queue | None = None
        self._replay_stop = threading.Event()
        self._replay_thread: threading.Thread | None = None
        self.dropped_frames: int = 0
        self.fps: float = 30.0
        self._replay_end = object()

    @property
    def frame_size(self) -> tuple[int, int] | None:
        if self.width and self.height:
            return (self.width, self.height)
        return None

    def open(self) -> None:
        if self._use_picamera2:
            try:
                from picamera2 import Picamera2
                self._picamera = Picamera2()
                width = self.width or 640
                height = self.height or 480
                config = self._picamera.create_video_configuration(
                    main={"size": (width, height)}
                )
                self._picamera.configure(config)
                self._picamera.start()
                self.width = width
                self.height = height
            except ImportError as exc:
                raise RuntimeError("python3-picamera2 is required for picamera2 source") from exc
        else:
            try:
                import cv2
                self._capture = cv2.VideoCapture(self._source)
                if not self._capture.isOpened():
                    raise RuntimeError(f"Failed to open video source: {self._source}")
                if self.width:
                    self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                if self.height:
                    self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_w > 0 and actual_h > 0:
                    self.width = actual_w
                    self.height = actual_h
                fps = self._capture.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    self.fps = fps
                if self.realtime and not isinstance(self._source, int):
                    self._start_replay()
            except ImportError as exc:
                raise RuntimeError("opencv-python or python3-opencv is required") from exc

    def _start_replay(self) -> None:
        self._replay_queue = queue.Queue(maxsize=max(1, self.buffer_size))
        self._replay_stop.clear()
        self._replay_thread = threading.Thread(target=self._replay_worker, daemon=True)
        self._replay_thread.start()

    def _replay_worker(self) -> None:
        frame_interval = 1.0 / max(1.0, self.fps)
        next_time = time.time()
        while not self._replay_stop.is_set():
            ret, frame = self._capture.read()
            if not ret:
                try:
                    self._replay_queue.put(self._replay_end, timeout=1.0)
                except queue.Full:
                    pass
                break
            try:
                self._replay_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._replay_queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
                try:
                    self._replay_queue.put_nowait(frame)
                except queue.Full:
                    pass
            next_time += frame_interval
            sleep_s = next_time - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_time = time.time()

    def read(self) -> tuple[bool, Any | None]:
        if self._use_picamera2:
            if self._picamera is None:
                raise RuntimeError("CameraSource.open() must be called first")
            import cv2
            frame = self._picamera.capture_array()
            ok, frame = True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self._replay_queue is not None:
            try:
                item = self._replay_queue.get(timeout=max(1.0, 2.0 / max(1.0, self.fps)))
                if item is self._replay_end:
                    return False, None
                ok, frame = True, item
            except queue.Empty:
                return False, None
        else:
            if self._capture is None:
                raise RuntimeError("CameraSource.open() must be called first")
            ok, frame = self._capture.read()

        if ok and frame is not None and (self.width is None or self.height is None):
            if hasattr(frame, "shape") and len(frame.shape) >= 2:
                self.height = int(frame.shape[0])
                self.width = int(frame.shape[1])

        return ok, frame

    def release(self) -> None:
        self._replay_stop.set()
        if self._replay_thread:
            self._replay_thread.join(timeout=2.0)
            self._replay_thread = None
        self._replay_queue = None
        if self._picamera is not None:
            self._picamera.stop()
            self._picamera.close()
            self._picamera = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
