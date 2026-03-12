import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from qfluentwidgets import (
    SimpleCardWidget, CaptionLabel, BodyLabel, PushButton, PrimaryPushButton, 
    TransparentPushButton, ComboBox, ProgressBar
)

from image_process import (
    undistort, compute_backward_remap, compute_forward_point_map,
    export_table_python, export_table_c,
    K as DEFAULT_K, D as DEFAULT_D
)


# ============================================================
# 可点击图像标签 (鼠标选 4 点)
# ============================================================
class ClickableImageLabel(QLabel):
    """可点击的图像标签，支持在图像上选点并绘制标记。"""
    point_added = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "background-color: #191919; border-radius: 8px;"
        )
        self._cv_image = None
        self._display_image = None
        self._points = []
        self._clickable = False

    def set_image(self, cv_img: np.ndarray):
        self._cv_image = cv_img.copy()
        self._display_image = cv_img.copy()
        self._points = []
        self._update_pixmap()

    def set_clickable(self, enabled: bool):
        self._clickable = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def get_points(self):
        return list(self._points)

    def clear_points(self):
        self._points = []
        if self._cv_image is not None:
            self._display_image = self._cv_image.copy()
            self._update_pixmap()

    def mousePressEvent(self, event):
        if not self._clickable or self._cv_image is None:
            return
        if len(self._points) >= 4 or event.button() != Qt.LeftButton:
            return
        pixmap = self.pixmap()
        if pixmap is None:
            return

        lw, lh = self.width(), self.height()
        pw, ph = pixmap.width(), pixmap.height()
        ox, oy = (lw - pw) / 2, (lh - ph) / 2
        px, py = event.x() - ox, event.y() - oy
        if px < 0 or py < 0 or px >= pw or py >= ph:
            return

        ih, iw = self._cv_image.shape[:2]
        rx = max(0, min(int(px * iw / pw), iw - 1))
        ry = max(0, min(int(py * ih / ph), ih - 1))
        self._points.append((rx, ry))

        cv2.circle(self._display_image, (rx, ry), 5, (0, 140, 255), -1)
        cv2.putText(self._display_image, f"({rx},{ry})",
                    (rx + 6, ry - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 200, 255), 1)
        self._update_pixmap()
        self.point_added.emit(rx, ry)

    def _update_pixmap(self):
        if self._display_image is None:
            return
        img = self._display_image
        h, w = img.shape[:2]
        if img.ndim == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        else:
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        scaled = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()


# ============================================================
# 打表多线程
# ============================================================
class MappingTaskThread(QThread):
    """将映射计算 + export 放入后台线程"""
    progress = pyqtSignal(int)       # 0-100
    finished_ok = pyqtSignal(str)    # 导出目录
    finished_err = pyqtSignal(str)   # 错误信息

    def __init__(self, K, D, M, out_size, img_size, out_dir, export_format="python", mode="backward", parent=None):
        super().__init__(parent)
        self._K = K
        self._D = D
        self._M = M
        self._out_size = out_size
        self._img_size = img_size
        self._out_dir = out_dir
        self._fmt = export_format   # "python" | "c"
        self._mode = mode           # "forward" | "backward"

    def run(self):
        try:
            self.progress.emit(10)
            
            if self._mode == "forward":
                map_h, map_w = compute_forward_point_map(
                    camera_matrix=self._K,
                    dist_coeffs=self._D,
                    M=self._M,
                    in_size=(160, 120),
                    img_size=self._img_size,
                    bev_size=self._out_size
                )
                name_h, name_w = "ForwardMapH", "ForwardMapW"
            else:
                map_h, map_w = compute_backward_remap(
                    camera_matrix=self._K,
                    dist_coeffs=self._D,
                    M=self._M,
                    out_size=self._out_size,
                    img_size=self._img_size
                )
                name_h, name_w = "BackwardMapH", "BackwardMapW"
                
            self.progress.emit(60)
            
            if self._fmt == "python":
                export_table_python(
                    map_h, map_w,
                    name_h, name_w,
                    self._out_dir,
                )
            else:
                export_table_c(
                    map_h, map_w,
                    name_h, name_w,
                    self._out_dir,
                )
            self.progress.emit(100)
            self.finished_ok.emit(os.path.abspath(self._out_dir))
        except Exception as e:
            self.finished_err.emit(str(e))


# ============================================================
# 逆透视交互页面 (核心业务逻辑)
# ============================================================
class PerspectiveInterface(QFrame):
    """ 逆透视交互页面 """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("PerspectiveInterface")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#PerspectiveInterface { background: transparent; }")

        # 内部状态
        self._original_img = None
        self._undistorted_img = None
        self._perspective_M = None
        self._mapping_thread = None

        self._init_ui()

    # ---------- 辅助：获取 MainWindow ----------
    def _main_window(self):
        w = self.parent()
        while w is not None:
            if hasattr(w, 'settings_interface'):
                return w
            w = w.parent()
        return None

    # ---------- 辅助：获取当前 K / D ----------
    def _get_KD(self):
        mw = self._main_window()
        if mw and hasattr(mw, 'calibration_interface'):
            ci = mw.calibration_interface
            if hasattr(ci, '_K'):
                return ci._K, ci._D
        return DEFAULT_K.copy(), DEFAULT_D.copy()

    # ---------- UI ----------
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 60, 20, 20)
        root.setSpacing(12)

        # 顶部操作提示
        tip = CaptionLabel(
            "提示：请在去畸变图上按【左上 → 右上 → 左下 → 右下】的顺序依次点击 4 个顶点。"
        )
        tip.setStyleSheet("color: rgba(255, 255, 255, 0.4);")
        root.addWidget(tip)

        # 上半部分：两个图像卡片并排
        img_row = QHBoxLayout()
        img_row.setSpacing(16)

        # 卡片 A — 去畸变交互图
        card_a = SimpleCardWidget()
        card_a_layout = QVBoxLayout(card_a)
        card_a_layout.setContentsMargins(12, 12, 12, 12)
        card_a_layout.setSpacing(6)
        card_a_layout.addWidget(CaptionLabel("去畸变交互图 — 左键选 4 点"))
        self.image_label_a = ClickableImageLabel()
        self.image_label_a.point_added.connect(self._on_point_added)
        card_a_layout.addWidget(self.image_label_a, stretch=1)
        img_row.addWidget(card_a, stretch=1)

        # 卡片 B — 逆透视结果
        card_b = SimpleCardWidget()
        card_b_layout = QVBoxLayout(card_b)
        card_b_layout.setContentsMargins(12, 12, 12, 12)
        card_b_layout.setSpacing(6)
        card_b_layout.addWidget(CaptionLabel("逆透视结果图"))
        self.image_label_b = ClickableImageLabel()
        card_b_layout.addWidget(self.image_label_b, stretch=1)
        img_row.addWidget(card_b, stretch=1)

        root.addLayout(img_row, stretch=1)

        # 下半部分：操作栏
        ctrl_card = SimpleCardWidget()
        ctrl_layout = QHBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(16, 12, 16, 12)
        ctrl_layout.setSpacing(12)

        self.btn_load = PushButton("加载测试图片")
        self.btn_load.clicked.connect(self._on_load_image)

        self.btn_perspective = PrimaryPushButton("执行逆透视")
        self.btn_perspective.setEnabled(False)
        self.btn_perspective.clicked.connect(self._on_run_perspective)

        # 优化点 1：精简文本，减小固定宽度
        self.export_mode_combo = ComboBox()
        self.export_mode_combo.addItems([
            "正向点表 (C++巡线)",
            "反向映射 (GUI预览)",
        ])
        self.export_mode_combo.setCurrentIndex(0)
        self.export_mode_combo.setFixedWidth(160)

        # 优化点 2：统一宽度，保持视觉对齐
        self.export_format_combo = ComboBox()
        self.export_format_combo.addItems([
            "Python 格式 (.py)",
            "C/C++ 格式 (.txt)",
        ])
        self.export_format_combo.setCurrentIndex(0)
        self.export_format_combo.setFixedWidth(160)

        self.btn_export = PushButton("导出映射表")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)

        self.btn_clear = TransparentPushButton("清除选点")
        self.btn_clear.clicked.connect(self._on_clear_points)

        ctrl_layout.addWidget(self.btn_load)
        ctrl_layout.addWidget(self.btn_perspective)
        ctrl_layout.addWidget(self.export_mode_combo)
        ctrl_layout.addWidget(self.export_format_combo)
        ctrl_layout.addWidget(self.btn_export)
        ctrl_layout.addWidget(self.btn_clear)
        root.addWidget(ctrl_card)

        # 状态栏 + 进度条
        status_row = QHBoxLayout()
        status_row.setSpacing(16)
        self.status_label = BodyLabel("等待操作...")
        self.status_label.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        status_row.addWidget(self.status_label, stretch=1)

        self.progress_bar = ProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        status_row.addWidget(self.progress_bar)
        root.addLayout(status_row)

    # ---------- 槽函数 ----------
    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", ".",
            "Images (*.jpg *.jpeg *.png *.bmp);;All (*)",
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法读取:\n{path}")
            return

        self._original_img = img
        K, D = self._get_KD()
        self._undistorted_img = undistort(img, K, D)
        self._perspective_M = None

        self.image_label_a.set_image(self._undistorted_img)
        self.image_label_a.set_clickable(True)
        self.image_label_b.clear()

        self.btn_perspective.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        h, w = img.shape[:2]
        self.status_label.setText(f"已加载: {os.path.basename(path)} ({w}×{h})  |  选点进度: 0/4")

    def _on_point_added(self, x: int, y: int):
        n = len(self.image_label_a.get_points())
        self.status_label.setText(f"选点进度: {n}/4  |  当前坐标: ({x}, {y})")
        if n >= 4:
            self.btn_perspective.setEnabled(True)
            self.status_label.setText("✔ 已选 4 点，可执行逆透视")

    def _on_clear_points(self):
        self.image_label_a.clear_points()
        self.btn_perspective.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._perspective_M = None
        self.image_label_b.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.status_label.setText("已清除选点")

    def _on_run_perspective(self):
        points = self.image_label_a.get_points()
        if len(points) < 4 or self._undistorted_img is None:
            return

        mw = self._main_window()
        if mw is None:
            QMessageBox.warning(self, "错误", "无法访问设置页面")
            return
        si = mw.settings_interface
        try:
            objdx = float(si.input_objdx.text())
            objdy = float(si.input_objdy.text())
            imgdx = float(si.input_imgdx.text())
            imgdy = float(si.input_imgdy.text())
            result_w = int(si.input_result_w.text())
            result_h = int(si.input_result_h.text())
        except ValueError:
            QMessageBox.warning(self, "参数错误", "请在设置页面填入有效数字")
            return

        pts_src = np.float32(points)
        pts_dst = np.float32([
            [imgdx,         imgdy],
            [imgdx + objdx, imgdy],
            [imgdx,         imgdy + objdy],
            [imgdx + objdx, imgdy + objdy],
        ])

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        self._perspective_M = M

        result = cv2.warpPerspective(self._undistorted_img, M, (result_w, result_h))
        self.image_label_b.set_image(result)

        self.btn_export.setEnabled(True)
        self.status_label.setText(f"逆透视完成: {result_w}×{result_h}")

    def _on_export(self):
        if self._perspective_M is None or self._original_img is None:
            return
        if self._mapping_thread is not None and self._mapping_thread.isRunning():
            return

        # 弹出目录选择
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", "."
        )
        if not export_dir:
            return

        # 读取格式
        fmt = "python" if self.export_format_combo.currentIndex() == 0 else "c"

        K, D = self._get_KD()
        img_h, img_w = self._original_img.shape[:2]
        img_size = (img_w, img_h)
        
        # 获取目标输出尺寸
        mw = self._main_window()
        if mw is None:
            QMessageBox.warning(self, "错误", "无法访问设置页面")
            return
        si = mw.settings_interface
        try:
            out_w = int(si.input_result_w.text())
            out_h = int(si.input_result_h.text())
        except ValueError:
            QMessageBox.warning(self, "参数错误", "请在设置页面填入有效数字")
            return
        out_size = (out_w, out_h)

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_export.setEnabled(False)
        
        mode = "forward" if self.export_mode_combo.currentIndex() == 0 else "backward"
        fmt_label = "Python (.py)" if fmt == "python" else "C/C++ (.txt)"
        mode_label = "正向" if mode == "forward" else "反向"
        self.status_label.setText(f"正在打表 [{mode_label} | {fmt_label}] → {export_dir}")

        self._mapping_thread = MappingTaskThread(
            K, D, self._perspective_M, out_size, img_size,
            export_dir, export_format=fmt, mode=mode, parent=self
        )
        self._mapping_thread.progress.connect(self._on_mapping_progress)
        self._mapping_thread.finished_ok.connect(self._on_mapping_done)
        self._mapping_thread.finished_err.connect(self._on_mapping_error)
        self._mapping_thread.start()

    def _on_mapping_progress(self, val: int):
        self.progress_bar.setValue(val)

    def _on_mapping_done(self, abs_dir: str):
        self.btn_export.setEnabled(True)
        self.status_label.setText(f"✅ 导出完成: {abs_dir}")
        QMessageBox.information(self, "导出完成", f"映射表已导出到:\n{abs_dir}")

    def _on_mapping_error(self, err: str):
        self.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"❌ 导出失败")
        QMessageBox.warning(self, "导出失败", err)
