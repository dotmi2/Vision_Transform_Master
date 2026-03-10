"""Calibration engine"""
import cv2
import numpy as np
from typing import List, Optional, Tuple


class CalibrationEngine:
    """Camera calibration engine"""

    def __init__(self):
        self.objpoints: List[np.ndarray] = []
        self.imgpoints: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.reprojection_error: float = 0.0

    def add_image(self, frame: np.ndarray, board_manager) -> Tuple[bool, str]:
        """Add calibration image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        self.image_size = (w, h)

        # Detect corners
        corners, ids = board_manager.detect_corners(gray)

        if corners is None:
            return False, "Could not detect board corners"

        min_corners = 6 if board_manager.board_type == "charuco" else 4
        if len(corners) < min_corners:
            return False, f"Not enough corners: {len(corners)} < {min_corners}"

        # Get object points
        objp = board_manager.get_object_points(corners, ids)

        # Format points
        if board_manager.board_type == "chessboard":
            imgp = corners.reshape(-1, 1, 2).astype(np.float64)
        else:
            imgp = corners.reshape(-1, 1, 2).astype(np.float64)

        self.objpoints.append(objp)
        self.imgpoints.append(imgp)

        return True, f"Image captured. Total: {len(self.objpoints)}"

    def calibrate(self, model: str = "pinhole") -> Tuple[bool, str]:
        """Perform calibration"""
        if len(self.objpoints) < 3:
            return False, f"Need at least 3 images, got {len(self.objpoints)}"

        if model == "pinhole":
            return self._calibrate_pinhole()
        elif model == "fisheye":
            return self._calibrate_fisheye()
        else:
            return False, f"Unknown model: {model}"

    def _calibrate_pinhole(self) -> Tuple[bool, str]:
        """Pinhole calibration"""
        objp_list = []
        imgp_list = []

        for obj, img in zip(self.objpoints, self.imgpoints):
            objp_list.append(obj.reshape(-1, 3).astype(np.float32))
            imgp_list.append(img.reshape(-1, 1, 2).astype(np.float32))

        K = np.eye(3, dtype=np.float64)
        D = np.zeros((5, 1), dtype=np.float64)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objp_list, imgp_list, self.image_size, K, D, criteria=criteria
        )

        # Calculate error
        error = self._calc_error_pinhole(objp_list, imgp_list, rvecs, tvecs, K, D)

        self.camera_matrix = K
        self.dist_coeffs = D
        self.reprojection_error = error

        return True, f"Calibration successful! Error: {error:.4f}px"

    def _calibrate_fisheye(self) -> Tuple[bool, str]:
        """Fisheye calibration"""
        objp_list = []
        imgp_list = []

        for obj, img in zip(self.objpoints, self.imgpoints):
            objp_list.append(obj.reshape(-1, 1, 3).astype(np.float64))
            imgp_list.append(img.reshape(-1, 1, 2).astype(np.float64))

        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objp_list, imgp_list, self.image_size, K, D, None, None,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC, criteria=criteria
        )

        error = self._calc_error_fisheye(objp_list, imgp_list, rvecs, tvecs, K, D)

        self.camera_matrix = K
        self.dist_coeffs = D
        self.reprojection_error = error

        return True, f"Fisheye calibration successful! Error: {error:.4f}px"

    def _calc_error_pinhole(self, objp, imgp, rvecs, tvecs, K, D) -> float:
        """Calculate reprojection error"""
        total_error = 0
        total_points = 0
        for i in range(len(objp)):
            proj, _ = cv2.projectPoints(objp[i], rvecs[i], tvecs[i], K, D)
            error = cv2.norm(imgp[i], proj, cv2.NORM_L2) / len(proj)
            total_error += error * len(objp[i])
            total_points += len(objp[i])
        return total_error / total_points

    def _calc_error_fisheye(self, objp, imgp, rvecs, tvecs, K, D) -> float:
        """Calculate fisheye reprojection error"""
        total_error = 0
        total_points = 0
        for i in range(len(objp)):
            proj, _ = cv2.fisheye.projectPoints(objp[i], rvecs[i], tvecs[i], K, D)
            error = cv2.norm(imgp[i], proj, cv2.NORM_L2) / len(proj)
            total_error += error * len(objp[i])
            total_points += len(objp[i])
        return total_error / total_points

    def save_yaml(self, filepath: str, model: str):
        """Save calibration to YAML"""
        if self.camera_matrix is None:
            return

        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        fs.write("image_width", self.image_size[0])
        fs.write("image_height", self.image_size[1])
        fs.write("model", model)
        fs.write("camera_matrix", self.camera_matrix)
        fs.write("distortion_coefficients", self.dist_coeffs)
        fs.write("reprojection_error", float(self.reprojection_error))
        fs.release()

    def generate_cpp(self, camera_params: dict) -> str:
        """Generate C++ code"""
        if self.camera_matrix is None:
            return "// No calibration data"

        K = self.camera_matrix
        D = self.dist_coeffs.flatten()

        return f"""#include <opencv2/opencv.hpp>

int main() {{
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, {camera_params.get('width', 640)});
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, {camera_params.get('height', 480)});

    cv::Mat K = (cv::Mat_<double>(3,3) <<
        {K[0,0]:.6f}, {K[0,1]:.6f}, {K[0,2]:.6f},
        {K[1,0]:.6f}, {K[1,1]:.6f}, {K[1,2]:.6f},
        {K[2,0]:.6f}, {K[2,1]:.6f}, {K[2,2]:.6f});

    cv::Mat D = (cv::Mat_<double>(1,{len(D)}) << {', '.join(f'{d:.6f}' for d in D)});

    cv::Mat frame, undistorted;
    while(true) {{
        cap >> frame;
        cv::undistort(frame, undistorted, K, D);
        cv::imshow("Undistorted", undistorted);
        if(cv::waitKey(1) == 'q') break;
    }}
    return 0;
}}"""

    def reset(self):
        """Clear calibration data"""
        self.objpoints.clear()
        self.imgpoints.clear()
        self.image_size = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.reprojection_error = 0.0
