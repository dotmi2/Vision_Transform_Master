"""Simple camera manager module"""
import cv2
import numpy as np
from typing import Optional, Tuple


class CameraManager:
    """Manages camera operations"""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_index = 0
        # Camera parameters
        self.width = 640
        self.height = 480
        self.fps = 30
        self.exposure_mode = 'auto'
        self.exposure_value = -6
        # Calibration data
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

    def discover_cameras(self) -> list:
        """Find available cameras"""
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(f"Camera {i}")
                cap.release()
            else:
                break
        return cameras if cameras else ["No cameras found"]

    def open_camera(self, index: int) -> str:
        """Open camera by index"""
        if self.cap is not None:
            self.cap.release()

        self.current_index = index
        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            return f"Failed to open camera {index}"

        # Apply settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return f"Camera {index} opened: {actual_w}x{actual_h}"

    def update_params(self, width: int = None, height: int = None,
                     fps: int = None, exposure_mode: str = None,
                     exposure_value: float = None) -> str:
        """Update camera parameters"""
        if width: self.width = width
        if height: self.height = height
        if fps: self.fps = fps
        if exposure_mode: self.exposure_mode = exposure_mode
        if exposure_value is not None: self.exposure_value = exposure_value

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if self.exposure_mode == 'auto':
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_value)

        return "Parameters updated"

    def get_frame(self, board_manager) -> Optional[np.ndarray]:
        """Get camera frame with overlay"""
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Detect corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = board_manager.detect_corners(gray)

        # Draw corners
        if board_manager.board_type == "chessboard" and corners is not None:
            cv2.drawChessboardCorners(frame, board_manager.chessboard_size, corners, True)
        elif board_manager.board_type == "charuco" and corners is not None and ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)

        # Apply undistortion if calibrated
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            frame = self._undistort(frame)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        """Apply distortion correction"""
        try:
            h, w = frame.shape[:2]
            new_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_matrix)
        except:
            return frame

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture single frame"""
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        """Close camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
