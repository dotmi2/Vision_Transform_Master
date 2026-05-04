"""
历史记录管理器
使用 .vision_history.json 进行本地轻量级持久化存储。
提供标定记录和逆透视记录的 CRUD 操作，以及基于 cv2.FileStorage 的 YAML 导入导出。
"""
import os
import json
import time
import uuid
from typing import Dict, List, Optional

import cv2
import numpy as np


class HistoryManager:
    """管理标定记录和逆透视记录的轻量级本地数据库"""

    _DEFAULT_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".vision_history.json",
    )

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or self._DEFAULT_PATH
        self._data = self._load()

    # ================================================================
    # 内部：读写 JSON
    # ================================================================
    def _load(self) -> dict:
        if os.path.isfile(self._db_path):
            try:
                with open(self._db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"calibration_records": [], "perspective_records": []}

    def _save(self):
        with open(self._db_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _gen_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    # ================================================================
    # 标定记录 CRUD
    # ================================================================
    def add_calibration_record(
        self,
        name: str,
        image_count: int,
        avg_error: float,
        K: np.ndarray,
        D: np.ndarray,
        model: str = "pinhole",
        image_size: Optional[tuple] = None,
    ) -> Dict:
        """新增一条标定记录并持久化。如果同名则自动加后缀。"""
        # --- 新增：防重名逻辑 ---
        existing_names = [r["name"] for r in self._data.get("calibration_records", [])]
        original_name = name
        counter = 1
        while name in existing_names:
            name = f"{original_name}_{counter}"
            counter += 1
        # -----------------------
        
        record = {
            "id": self._gen_id(),
            "name": name,
            "date": self._now(),
            "image_count": image_count,
            "avg_error": round(avg_error, 6),
            "model": model,
            "K": K.tolist() if isinstance(K, np.ndarray) else K,
            "D": D.flatten().tolist() if isinstance(D, np.ndarray) else D,
        }
        if image_size is not None:
            record["image_width"] = int(image_size[0])
            record["image_height"] = int(image_size[1])
        self._data["calibration_records"].append(record)
        self._save()
        return record

    def get_calibration_records(self) -> List[Dict]:
        """获取全部标定记录列表（按时间倒序）。"""
        return list(reversed(self._data.get("calibration_records", [])))

    def get_calibration_record_by_id(self, record_id: str) -> Optional[Dict]:
        """按 ID 获取单条标定记录。"""
        for r in self._data.get("calibration_records", []):
            if r["id"] == record_id:
                return r
        return None

    def rename_record(self, collection: str, record_id: str, new_name: str) -> bool:
        """通用重命名方法。包含防重名检查。"""
        records = self._data.get(collection, [])
        # 检查是否已有同名（且不是自己）
        if any(r["name"] == new_name and r["id"] != record_id for r in records):
            raise ValueError(f"名称 '{new_name}' 已存在。")
            
        for r in records:
            if r["id"] == record_id:
                r["name"] = new_name
                self._save()
                return True
        return False

    def delete_calibration_record(self, record_id: str) -> bool:
        """按 ID 删除一条标定记录。返回是否成功。"""
        records = self._data.get("calibration_records", [])
        before = len(records)
        self._data["calibration_records"] = [
            r for r in records if r["id"] != record_id
        ]
        if len(self._data["calibration_records"]) < before:
            self._save()
            return True
        return False

    # ================================================================
    # 逆透视记录 CRUD
    # ================================================================
    def add_perspective_record(
        self,
        name: str,
        objdx: float,
        objdy: float,
        imgdx: float,
        imgdy: float,
        out_w: int,
        out_h: int,
    ) -> Dict:
        """新增一条逆透视参数记录并持久化。如果同名则自动加后缀。"""
        # --- 新增：防重名逻辑 ---
        existing_names = [r["name"] for r in self._data.get("perspective_records", [])]
        original_name = name
        counter = 1
        while name in existing_names:
            name = f"{original_name}_{counter}"
            counter += 1
        # -----------------------

        record = {
            "id": self._gen_id(),
            "name": name,
            "date": self._now(),
            "objdx": objdx,
            "objdy": objdy,
            "imgdx": imgdx,
            "imgdy": imgdy,
            "out_w": out_w,
            "out_h": out_h,
        }
        self._data["perspective_records"].append(record)
        self._save()
        return record

    def get_perspective_records(self) -> List[Dict]:
        """获取全部逆透视记录列表（按时间倒序）。"""
        return list(reversed(self._data.get("perspective_records", [])))

    def get_perspective_record_by_id(self, record_id: str) -> Optional[Dict]:
        """按 ID 获取单条逆透视记录。"""
        for r in self._data.get("perspective_records", []):
            if r["id"] == record_id:
                return r
        return None

    def delete_perspective_record(self, record_id: str) -> bool:
        """按 ID 删除一条逆透视记录。"""
        records = self._data.get("perspective_records", [])
        before = len(records)
        self._data["perspective_records"] = [
            r for r in records if r["id"] != record_id
        ]
        if len(self._data["perspective_records"]) < before:
            self._save()
            return True
        return False

    # ================================================================
    # YAML 导出/导入（基于 cv2.FileStorage）
    # ================================================================
    @staticmethod
    def export_calibration_to_yaml(
        filepath: str,
        K: np.ndarray,
        D: np.ndarray,
        model: str = "pinhole",
        image_size: Optional[tuple] = None,
        reprojection_error: Optional[float] = None,
    ):
        """将内参矩阵 K 和畸变系数 D 导出为 YAML 文件。"""
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        if image_size is not None:
            fs.write("image_width", int(image_size[0]))
            fs.write("image_height", int(image_size[1]))
        fs.write("model", model)
        fs.write("camera_matrix", K.astype(np.float64))
        fs.write("dist_coeffs", D.reshape(-1, 1).astype(np.float64))
        fs.write("distortion_coefficients", D.reshape(-1, 1).astype(np.float64))
        if reprojection_error is not None:
            fs.write("reprojection_error", float(reprojection_error))
        fs.release()

    @staticmethod
    def import_calibration_bundle_from_yaml(filepath: str) -> Dict:
        """
        从 YAML 文件读取内参矩阵和畸变系数。
        返回包含 K/D/model/image_size 的字典，读取失败时抛出 ValueError。
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        try:
            K = fs.getNode("camera_matrix").mat()
            D = fs.getNode("dist_coeffs").mat()
            if D is None:
                D = fs.getNode("distortion_coefficients").mat()

            model_node = fs.getNode("model")
            model = model_node.string() if not model_node.empty() else "pinhole"

            width_node = fs.getNode("image_width")
            height_node = fs.getNode("image_height")
            image_width = int(width_node.real()) if not width_node.empty() else None
            image_height = int(height_node.real()) if not height_node.empty() else None

            err_node = fs.getNode("reprojection_error")
            reprojection_error = float(err_node.real()) if not err_node.empty() else None
        finally:
            fs.release()

        if K is None or D is None:
            raise ValueError(
                "YAML 文件中未找到 camera_matrix 或 dist_coeffs/distortion_coefficients 节点"
            )

        image_size = None
        if image_width is not None and image_height is not None:
            image_size = (image_width, image_height)

        return {
            "K": K,
            "D": D,
            "model": model or "pinhole",
            "image_size": image_size,
            "reprojection_error": reprojection_error,
        }

    @staticmethod
    def import_calibration_from_yaml(filepath: str):
        bundle = HistoryManager.import_calibration_bundle_from_yaml(filepath)
        return bundle["K"], bundle["D"]
