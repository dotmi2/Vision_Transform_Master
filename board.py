"""Board manager for calibration patterns"""
import cv2
import numpy as np
from typing import Optional, Tuple


class BoardManager:
    """Manages calibration boards"""

    def __init__(self):
        self.board_type = "charuco"

        # Chessboard config
        self.chessboard_size = (9, 6)
        self.square_size = 25.0

        # ChArUco config
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.board = cv2.aruco.CharucoBoard((7, 10), 25.0, 17.5, self.dictionary)

    def set_board_type(self, board_type: str, **kwargs) -> str:
        """Set board type and parameters"""
        self.board_type = board_type

        if board_type == "chessboard":
            if 'chessboard_size' in kwargs:
                self.chessboard_size = tuple(kwargs['chessboard_size'])
            if 'square_size' in kwargs:
                self.square_size = float(kwargs['square_size'])
        elif board_type == "charuco":
            if 'dict_name' in kwargs:
                dict_id = getattr(cv2.aruco, kwargs['dict_name'])
                self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

            squares_x = kwargs.get('squares_x', 7)
            squares_y = kwargs.get('squares_y', 10)
            square_len = kwargs.get('square_len', 25.0)
            marker_len = kwargs.get('marker_len', 17.5)

            self.board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y), square_len, marker_len, self.dictionary
            )

        return f"Board type set to {board_type}"

    def detect_corners(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect board corners"""
        if self.board_type == "chessboard":
            corners = self._detect_chessboard(gray)
            return corners, None
        else:
            return self._detect_charuco(gray)

    def _detect_chessboard(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect chessboard corners"""
        if hasattr(cv2, "findChessboardCornersSB"):
            ok, corners = cv2.findChessboardCornersSB(gray, self.chessboard_size)
        else:
            ok, corners = cv2.findChessboardCorners(gray, self.chessboard_size)

        if not ok:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners

    def _detect_charuco(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect ChArUco corners"""
        # Try new API first
        if hasattr(cv2.aruco, "CharucoDetector"):
            detector = cv2.aruco.CharucoDetector(self.board)
            corners, ids, _, _ = detector.detectBoard(gray)
        else:
            # Old API
            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(self.dictionary)
                marker_corners, marker_ids, _ = detector.detectMarkers(gray)
            else:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)

            if marker_ids is None or len(marker_ids) < 4:
                return None, None

            ok, corners, ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board
            )
            if not ok:
                return None, None

        if ids is None or len(ids) < 6:
            return None, None

        return corners, ids

    def get_object_points(self, corners: np.ndarray, ids: Optional[np.ndarray]) -> np.ndarray:
        """Get 3D object points for corners"""
        if self.board_type == "chessboard":
            objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
            objp *= self.square_size
            return objp
        else:
            chess_corners_3d = self.board.getChessboardCorners()
            ids_flat = ids.flatten().astype(int)
            return chess_corners_3d[ids_flat, :].reshape(-1, 1, 3).astype(np.float64)

    def generate_image(self, width: int = 600, height: int = 800) -> np.ndarray:
        """Generate board image"""
        if self.board_type == "chessboard":
            return self._generate_chessboard(width, height)
        else:
            board_img = self.board.generateImage((width, height), marginSize=20, borderBits=1)
            return cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)

    def _generate_chessboard(self, width: int, height: int) -> np.ndarray:
        """Generate chessboard image"""
        cols, rows = self.chessboard_size
        sq = 50
        board_w, board_h = cols * sq, rows * sq
        img = np.zeros((board_h, board_w, 3), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                color = (255, 255, 255) if (x + y) % 2 == 0 else (0, 0, 0)
                cv2.rectangle(img, (x*sq, y*sq), ((x+1)*sq, (y+1)*sq), color, -1)

        img = cv2.resize(img, (width, height))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
