# 图像处理 GUI 应用 - QFluentWidgets 侧边栏重构 (完整版)
import os
import sys
import shutil
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QFrame, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QSizePolicy, QFileDialog, QMessageBox, QInputDialog,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QStackedWidget,
    QDialog
)
from PyQt5.QtGui import QFont, QImage, QPixmap, QIcon, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from qfluentwidgets import (
    FluentWindow,
    NavigationItemPosition,
    setTheme,
    Theme,
    FluentIcon as FIF,
    TitleLabel,
    SubtitleLabel,
    BodyLabel,
    CaptionLabel,
    LineEdit,
    PushButton,
    PrimaryPushButton,
    TransparentPushButton,
    SimpleCardWidget,
    ScrollArea,
    TextEdit,
    ProgressBar,
    ComboBox,
    SegmentedWidget,
    TableWidget,
    MessageBox,
)

from image_process import (
    undistort, compute_undist_inverse_map,
    export_table_python, export_table_c,
    K as DEFAULT_K, D as DEFAULT_D,
)
from board import BoardManager
from calibration import CalibrationEngine
from history_manager import HistoryManager

OUTPUT_DIR = "PrintTable"


# ============================================================
# 资源路径
# ============================================================
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# ============================================================
# 深色输入对话框
# ============================================================
class CustomInputDialog(QDialog):
    """无边框深色命名对话框，替代原生 QInputDialog"""
    def __init__(self, title: str, prompt: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setFixedSize(400, 180)
        self.setStyleSheet(
            "QDialog { background: #2B2B2B; border: 1px solid #444; border-radius: 10px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        lbl = QLabel(prompt)
        lbl.setStyleSheet("color: #E0E0E0; font-size: 13px;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self.line_edit = LineEdit()
        self.line_edit.setPlaceholderText("请输入名称...")
        layout.addWidget(self.line_edit)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch(1)
        self.btn_cancel = PushButton("取消")
        self.btn_cancel.setFixedWidth(80)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = PrimaryPushButton("确定")
        self.btn_ok.setFixedWidth(80)
        self.btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)

        self.line_edit.returnPressed.connect(self.accept)

    def get_text(self) -> str:
        return self.line_edit.text().strip()


# ============================================================
# 首页欢迎界面
# ============================================================
class HomeInterface(QFrame):
    """ 首页 / 欢迎界面 """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("HomeInterface")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#HomeInterface { background: transparent; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 60, 30, 30)
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.setSpacing(24)

        # Logo
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(ASSETS_DIR, "logo.png")
        if os.path.isfile(logo_path):
            pixmap = QPixmap(logo_path).scaled(
                512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("logo.png 未找到")
            logo_label.setFixedSize(512, 512)
            logo_label.setStyleSheet(
                "color: rgba(255,255,255,0.3); font-size: 12px;"
                "border: 2px dashed rgba(255,255,255,0.15); border-radius: 12px;"
            )
        layout.addWidget(logo_label, 0, Qt.AlignHCenter)

        # 标题
        title = TitleLabel("Vision Transform Master")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title, 0, Qt.AlignHCenter)

        # 副标题
        desc = BodyLabel("一款专业的相机标定与逆透视变换系统")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        layout.addWidget(desc, 0, Qt.AlignHCenter)

        # 使用说明
        guide = CaptionLabel(
            "使用流程:  相机标定 → 加载图片 → 选点逆透视 → 导出映射表"
        )
        guide.setAlignment(Qt.AlignCenter)
        guide.setStyleSheet("color: rgba(255, 255, 255, 0.4);")
        layout.addWidget(guide, 0, Qt.AlignHCenter)

        # 下载标定板 PDF
        btn_download = PushButton("下载 ChArUco 标定板 (PDF)")
        btn_download.setFixedWidth(280)
        btn_download.clicked.connect(self._on_download_pdf)
        layout.addWidget(btn_download, 0, Qt.AlignHCenter)

    def _on_download_pdf(self):
        src = os.path.join(ASSETS_DIR, "Charuco_A4.pdf")
        if not os.path.isfile(src):
            QMessageBox.warning(self, "文件缺失", f"未找到:\n{src}")
            return
        dst, _ = QFileDialog.getSaveFileName(
            self, "保存标定板 PDF", "Charuco_A4.pdf",
            "PDF (*.pdf);;All (*)",
        )
        if not dst:
            return
        try:
            shutil.copy(src, dst)
            QMessageBox.information(self, "导出成功", f"已保存到:\n{dst}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))


# ============================================================
# 相机标定页面
# ============================================================
class CalibrationInterface(QFrame):
    """ 相机标定页面 """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("CalibrationInterface")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#CalibrationInterface { background: transparent; }")

        # 标定引擎
        self._board_manager = BoardManager()
        self._calib_engine = CalibrationEngine()

        # 标定结果：默认使用硬编码参数
        self._K = DEFAULT_K.copy()
        self._D = DEFAULT_D.copy()

        self.scroll_area = ScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("ScrollArea { background: transparent; border: none; }")

        self.scroll_widget = QWidget()
        self.scroll_widget.setObjectName("calibrationScrollWidget")
        self.scroll_widget.setStyleSheet("#calibrationScrollWidget { background: transparent; }")
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(24, 24, 24, 24)
        self.scroll_layout.setSpacing(24)
        self.scroll_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self._init_ui()
        self.scroll_area.setWidget(self.scroll_widget)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 60, 30, 30)
        main_layout.addWidget(self.scroll_area)

    def _init_ui(self):
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # 左侧控制面板
        left_card = SimpleCardWidget(self.scroll_widget)
        left_card.setMaximumWidth(400)
        calib_layout = QVBoxLayout(left_card)
        calib_layout.setContentsMargins(24, 24, 24, 24)
        calib_layout.setSpacing(16)

        calib_layout.addWidget(SubtitleLabel("相机标定"))

        # 新手引导提示
        guide = CaptionLabel(
            "提示：建议拍摄至少 15-20 张标定图，标定板需包含大角度的俯仰、\n"
            "侧倾以及位于画面极度边缘的姿态，以保证高阶畸变参数准确收敛。"
        )
        guide.setWordWrap(True)
        guide.setStyleSheet("color: rgba(255, 255, 255, 0.4);")
        calib_layout.addWidget(guide)

        info_label = CaptionLabel(
            "标定板: ChArUco 7×10\n"
            "方块: 25.0mm  标记: 17.5mm\n"
            "字典: DICT_4X4_1000"
        )
        info_label.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        calib_layout.addWidget(info_label)

        self.calib_status_label = BodyLabel("状态: 未标定 (使用默认参数)")
        calib_layout.addWidget(self.calib_status_label)

        # 批量导入标定图片
        self.btn_import_calib = PushButton("批量导入标定图片")
        self.btn_import_calib.clicked.connect(self._on_import_images)
        calib_layout.addWidget(self.btn_import_calib)

        # 导入现有内参 YAML
        self.btn_import_yaml = PushButton("导入现有内参 (YAML)")
        self.btn_import_yaml.clicked.connect(self._on_import_yaml)
        calib_layout.addWidget(self.btn_import_yaml)

        calib_btn_row = QHBoxLayout()
        calib_btn_row.setContentsMargins(0, 0, 0, 0)
        self.btn_calibrate = PrimaryPushButton("开始标定")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.clicked.connect(self._on_calibrate)
        self.btn_reset_calib = TransparentPushButton("重置标定")
        self.btn_reset_calib.clicked.connect(self._on_reset)
        calib_btn_row.addWidget(self.btn_calibrate)
        calib_btn_row.addWidget(self.btn_reset_calib)
        calib_layout.addLayout(calib_btn_row)
        
        calib_layout.addStretch(1)

        # 右侧预览面板
        right_card = SimpleCardWidget(self.scroll_widget)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(24, 24, 24, 24)
        right_layout.setSpacing(16)
        right_layout.addWidget(SubtitleLabel("图片预览"), 0, Qt.AlignHCenter)
        
        self.photo_wall = PhotoWallWidget()
        right_layout.addWidget(self.photo_wall, 1, Qt.AlignCenter)

        content_layout.addWidget(left_card)
        content_layout.addWidget(right_card, stretch=1)

        self.scroll_layout.addLayout(content_layout)
        self.scroll_layout.addStretch(1)

    # ---------- 槽函数 ----------
    def _on_import_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择标定图片", ".",
            "Images (*.jpg *.jpeg *.png *.bmp);;All (*)",
        )
        if not files:
            return

        # 重置引擎后重新导入
        self._calib_engine.reset()
        success = 0
        valid_paths = []
        for path in files:
            frame = cv2.imread(path)
            if frame is None:
                continue
            ok, msg = self._calib_engine.add_image(frame, self._board_manager)
            if ok:
                success += 1
                valid_paths.append(path)

        self.photo_wall.set_images(valid_paths)

        total = len(self._calib_engine.objpoints)
        self.calib_status_label.setText(
            f"状态: 导入 {success}/{len(files)} 张成功\n"
            f"累计有效图片: {total}"
        )
        self.btn_calibrate.setEnabled(total >= 3)

    def _on_calibrate(self):
        total = len(self._calib_engine.objpoints)
        if total < 3:
            QMessageBox.warning(self, "提示", "至少需要 3 张有效标定图片")
            return

        ok, msg = self._calib_engine.calibrate("pinhole")
        if ok:
            self._K = self._calib_engine.camera_matrix
            self._D = self._calib_engine.dist_coeffs
            err = self._calib_engine.reprojection_error
            self.calib_status_label.setText(
                f"✅ 标定成功 ({total} 张)\n"
                f"重投影误差: {err:.4f} px"
            )
            w = MessageBox(
                "标定成功",
                f"使用 {total} 张图片\n重投影误差: {err:.4f} px\n\nK 和 D 已更新。",
                self.window()
            )
            w.yesButton.setText("确定")
            w.cancelButton.setText("取消")
            w.exec()
            # 保存到历史记录
            dlg = CustomInputDialog("保存记录", "标定成功！请为本次标定参数命名:", self)
            if dlg.exec_() == QDialog.Accepted and dlg.get_text():
                HistoryManager().add_calibration_record(
                    name=dlg.get_text(),
                    image_count=total,
                    avg_error=err,
                    K=self._K,
                    D=self._D,
                )
                # 刷新历史页面
                mw = self._main_window()
                if mw:
                    mw.history_interface.load_data()
        else:
            self.calib_status_label.setText(f"❌ 标定失败")
            w = MessageBox("标定失败", msg, self.window())
            w.yesButton.setText("确定")
            w.cancelButton.setText("取消")
            w.exec()

    def _main_window(self):
        w = self.parent()
        while w is not None:
            if hasattr(w, 'history_interface'):
                return w
            w = w.parent()
        return None

    def _on_import_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择内参文件", ".",
            "YAML/XML (*.yaml *.yml *.xml);;All (*)",
        )
        if not path:
            return
        try:
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            K = fs.getNode("camera_matrix").mat()
            D = fs.getNode("dist_coeffs").mat()
            fs.release()
            if K is None or D is None:
                raise ValueError("文件中未找到 camera_matrix 或 dist_coeffs")
            self._K = K
            self._D = D
            self.calib_status_label.setText(
                f"✅ 已加载现有参数\n来源: {os.path.basename(path)}"
            )
        except Exception as e:
            QMessageBox.warning(self, "读取失败", str(e))

    def _on_reset(self):
        self._calib_engine.reset()
        self._K = DEFAULT_K.copy()
        self._D = DEFAULT_D.copy()
        self.btn_calibrate.setEnabled(False)
        self.photo_wall.clear()
        self.calib_status_label.setText("状态: 未标定 (使用默认参数)")


# ============================================================
# 相册墙预览组件
# ============================================================
class PhotoWallWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 350)
        self.setStyleSheet("QFrame { background-color: rgba(255, 255, 255, 0.02); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); }")
        self.pixmaps = []

    def set_images(self, paths: list):
        self.pixmaps.clear()
        for p in paths[-3:]:
            img = QImage(p)
            if not img.isNull():
                pix = QPixmap.fromImage(img).scaled(280, 200, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                self.pixmaps.append(pix)
        self.update()

    def clear(self):
        self.pixmaps.clear()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if not self.pixmaps:
            painter.setPen(QColor(255, 255, 255, 80))
            font = painter.font()
            font.setPixelSize(14)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "暂无图片\n请导入标定图以预览")
            return
        cx, cy = self.width() / 2, self.height() / 2
        total = len(self.pixmaps)
        angles = [-8, 6, 0] # 赋予图片错落有致的旋转角度
        
        for i, pix in enumerate(self.pixmaps):
            layer = total - 1 - i 
            painter.save()
            painter.translate(cx, cy)
            
            # 偏移和旋转
            painter.translate(layer * 20, -layer * 15) 
            if layer != 0: painter.rotate(angles[i % 3])
            
            pw, ph = pix.width(), pix.height()
            
            # 绘制深色投影
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 100))
            painter.drawRoundedRect(-pw // 2 + 6, -ph // 2 + 6, pw, ph, 4, 4)
            
            # 绘制图片本体与高光边框
            painter.drawPixmap(-pw // 2, -ph // 2, pix)
            painter.setPen(QPen(QColor(255, 255, 255, 80), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(-pw // 2, -ph // 2, pw, ph)
            painter.restore()


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
    """将 compute_undist_inverse_map + export 放入后台线程"""
    progress = pyqtSignal(int)       # 0-100
    finished_ok = pyqtSignal(str)    # 导出目录
    finished_err = pyqtSignal(str)   # 错误信息

    def __init__(self, K, D, M, img_size, out_dir, export_format="python", parent=None):
        super().__init__(parent)
        self._K = K
        self._D = D
        self._M = M
        self._img_size = img_size
        self._out_dir = out_dir
        self._fmt = export_format   # "python" | "c"

    def run(self):
        try:
            self.progress.emit(10)
            map_h, map_w = compute_undist_inverse_map(
                self._K, self._D, self._M, self._img_size
            )
            self.progress.emit(60)
            if self._fmt == "python":
                export_table_python(
                    map_h, map_w,
                    "UndistInverseMapH", "UndistInverseMapW",
                    self._out_dir,
                )
            else:
                export_table_c(
                    map_h, map_w,
                    "UndistInverseMapH", "UndistInverseMapW",
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

        self.export_format_combo = ComboBox()
        self.export_format_combo.addItems([
            "导出为 Python 格式 (.py)",
            "导出为 C/C++ 格式 (.txt)",
        ])
        self.export_format_combo.setCurrentIndex(0)
        self.export_format_combo.setFixedWidth(220)

        self.btn_export = PushButton("导出映射表")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)

        self.btn_clear = TransparentPushButton("清除选点")
        self.btn_clear.clicked.connect(self._on_clear_points)

        ctrl_layout.addWidget(self.btn_load)
        ctrl_layout.addWidget(self.btn_perspective)
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
        img_size = (self._original_img.shape[0], self._original_img.shape[1])

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_export.setEnabled(False)
        fmt_label = "Python (.py)" if fmt == "python" else "C/C++ (.txt)"
        self.status_label.setText(f"正在打表 [{fmt_label}] → {export_dir}")

        self._mapping_thread = MappingTaskThread(
            K, D, self._perspective_M, img_size,
            export_dir, export_format=fmt, parent=self
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


# ============================================================
# 参数设置页面
# ============================================================
class SettingsInterface(QFrame):
    """ 参数设置页面 """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("SettingsInterface")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#SettingsInterface { background: transparent; }")

        self.scroll_area = ScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("ScrollArea { background: transparent; border: none; }")

        self.scroll_widget = QWidget()
        self.scroll_widget.setObjectName("settingsScrollWidget")
        self.scroll_widget.setStyleSheet("#settingsScrollWidget { background: transparent; }")
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(24, 24, 24, 24)
        self.scroll_layout.setSpacing(24)
        self.scroll_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self._init_ui()

        self.scroll_area.setWidget(self.scroll_widget)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 60, 30, 30)
        main_layout.addWidget(self.scroll_area)

    def _init_ui(self):
        param_card = SimpleCardWidget(self.scroll_widget)
        param_card.setMaximumWidth(600)
        param_layout = QVBoxLayout(param_card)
        param_layout.setContentsMargins(24, 24, 24, 24)
        param_layout.setSpacing(16)

        param_title = SubtitleLabel("逆透视参数配置")
        param_layout.addWidget(param_title)

        self.input_objdx = self._make_param_row(param_layout, "物理宽 (objdx)", "45")
        self.input_objdx.setToolTip("目标区域的物理宽度 (mm)，即透视变换后的实际宽度")

        self.input_objdy = self._make_param_row(param_layout, "物理高 (objdy)", "45")
        self.input_objdy.setToolTip("目标区域的物理高度 (mm)，即透视变换后的实际高度")

        self.input_imgdx = self._make_param_row(param_layout, "映射X (imgdx)", "57.5")
        self.input_imgdx.setToolTip("输出图像中目标区域左上角的 X 坐标偏移 (px)")

        self.input_imgdy = self._make_param_row(param_layout, "映射Y (imgdy)", "55")
        self.input_imgdy.setToolTip("输出图像中目标区域左上角的 Y 坐标偏移 (px)")

        self.input_result_w = self._make_param_row(param_layout, "输出宽度", "160")
        self.input_result_w.setToolTip("最终逆透视结果图的像素宽度")

        self.input_result_h = self._make_param_row(param_layout, "输出高度", "120")
        self.input_result_h.setToolTip("最终逆透视结果图的像素高度")

        self.scroll_layout.addWidget(param_card, 0, Qt.AlignHCenter)

        # 保存/恢复按钮栏
        btn_row_widget = QWidget()
        btn_row_layout = QHBoxLayout(btn_row_widget)
        btn_row_layout.setContentsMargins(0, 0, 0, 0)
        btn_row_layout.setSpacing(12)
        self.btn_save = PrimaryPushButton("保存当前配置为记录")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_reset = TransparentPushButton("恢复默认")
        self.btn_reset.clicked.connect(self._on_reset)
        btn_row_layout.addWidget(self.btn_save)
        btn_row_layout.addWidget(self.btn_reset)
        btn_row_layout.addStretch(1)
        self.scroll_layout.addWidget(btn_row_widget, 0, Qt.AlignHCenter)

        self.scroll_layout.addStretch(1)

    def _make_param_row(self, layout: QVBoxLayout, label_text: str, default: str) -> LineEdit:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        label = BodyLabel(label_text)
        edit = LineEdit()
        edit.setText(default)

        row_layout.addWidget(label, stretch=4)
        row_layout.addWidget(edit, stretch=6)
        layout.addWidget(row_widget)
        return edit

    def _main_window(self):
        w = self.parent()
        while w is not None:
            if hasattr(w, 'history_interface'):
                return w
            w = w.parent()
        return None

    def _on_save(self):
        dlg = CustomInputDialog("保存配置", "请为当前逆透视参数配置命名:", self)
        if dlg.exec_() != QDialog.Accepted or not dlg.get_text():
            return
        try:
            objdx = float(self.input_objdx.text())
            objdy = float(self.input_objdy.text())
            imgdx = float(self.input_imgdx.text())
            imgdy = float(self.input_imgdy.text())
            out_w = int(self.input_result_w.text())
            out_h = int(self.input_result_h.text())
        except ValueError:
            w = MessageBox("参数错误", "请确保所有输入框均为有效数字", self.window())
            w.yesButton.setText("确定")
            w.cancelButton.setText("取消")
            w.exec()
            return
        HistoryManager().add_perspective_record(
            name=dlg.get_text(),
            objdx=objdx, objdy=objdy,
            imgdx=imgdx, imgdy=imgdy,
            out_w=out_w, out_h=out_h,
        )
        mw = self._main_window()
        if mw:
            mw.history_interface.load_data()
        w = MessageBox("保存成功", f"配置「{dlg.get_text()}」已保存到历史记录。", self.window())
        w.yesButton.setText("确定")
        w.cancelButton.setText("取消")
        w.exec()

    def _on_reset(self):
        self.input_objdx.setText("45")
        self.input_objdy.setText("45")
        self.input_imgdx.setText("57.5")
        self.input_imgdy.setText("55")
        self.input_result_w.setText("160")
        self.input_result_h.setText("120")


# ============================================================
# 历史记录页面
# ============================================================
class HistoryInterface(QFrame):
    """ 历史记录管理页面 """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("HistoryInterface")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#HistoryInterface { background: transparent; }")

        self.history_mgr = HistoryManager()
        self._calib_ids = []   # 当前表格行对应的记录 ID
        self._persp_ids = []

        self._init_ui()
        self.load_data()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 60, 30, 20)
        root.setSpacing(16)

        root.addWidget(SubtitleLabel("历史记录"))

        self.segment = SegmentedWidget()
        self.segment.addItem("calib", "标定记录")
        self.segment.addItem("persp", "逆透视记录")
        self.segment.setCurrentItem("calib")
        self.segment.currentItemChanged.connect(self._on_tab_changed)
        root.addWidget(self.segment)

        self.stacked = QStackedWidget()
        self.stacked.setStyleSheet("QStackedWidget { background: transparent; }")

        self.calib_table = self._make_table(["名称", "日期", "图片数", "平均误差"])
        self.stacked.addWidget(self.calib_table)

        self.persp_table = self._make_table(["名称", "日期", "参数摘要"])
        self.stacked.addWidget(self.persp_table)

        root.addWidget(self.stacked, stretch=1)

        btn_card = SimpleCardWidget()
        btn_layout = QHBoxLayout(btn_card)
        btn_layout.setContentsMargins(16, 10, 16, 10)
        btn_layout.setSpacing(12)

        self.btn_apply = PrimaryPushButton("应用此记录")
        self.btn_apply.clicked.connect(self._on_apply_record)

        self.btn_rename = PushButton("重命名")
        self.btn_rename.clicked.connect(self._on_rename_record)

        self.btn_export_yaml = PushButton("导出为 YAML")
        self.btn_export_yaml.clicked.connect(self._on_export_yaml)

        self.btn_refresh = TransparentPushButton("刷新记录")
        self.btn_refresh.clicked.connect(self.load_data)

        self.btn_delete = TransparentPushButton("删除记录")
        self.btn_delete.setStyleSheet(
            "QPushButton { background-color: rgba(255, 69, 58, 0.15);"
            " color: #ff453a; border: 1px solid rgba(255, 69, 58, 0.3);"
            " border-radius: 5px; padding: 6px 12px; }"
            "QPushButton:hover { background-color: rgba(255, 69, 58, 0.25); }"
            "QPushButton:pressed { background-color: rgba(255, 69, 58, 0.1); }"
        )
        self.btn_delete.clicked.connect(self._on_delete_record)

        btn_layout.addWidget(self.btn_apply)
        btn_layout.addWidget(self.btn_rename)
        btn_layout.addWidget(self.btn_export_yaml)
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_delete)
        root.addWidget(btn_card)

    # ---------- 表格工厂 ----------
    def _make_table(self, headers: list) -> TableWidget:
        table = TableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setFocusPolicy(Qt.NoFocus)
        table.setAlternatingRowColors(True)
        table.setStyleSheet('''
            QTableWidget {
                background: transparent;
                alternate-background-color: rgba(255, 255, 255, 8); /* 极弱的透明白，完美融入深色 */
                border: none;
            }
            QHeaderView::section {
                background-color: #272727;
                color: #cccccc;
                border: none;
                border-bottom: 1px solid #333333;
                padding: 4px;
                font-weight: bold;
            }
            QTableWidget::item {
                background: transparent;
                color: #E0E0E0;
                border-bottom: 1px solid rgba(255,255,255,0.05);
            }
            QTableWidget::item:selected {
                background-color: rgba(255, 255, 255, 0.1);
                color: #FFFFFF;
            }
        ''')
        return table

    # ---------- 切换选项卡 ----------
    def _on_tab_changed(self, key: str):
        self.stacked.setCurrentIndex(0 if key == "calib" else 1)
        self.btn_export_yaml.setVisible(key == "calib")

    # ---------- 加载数据 ----------
    def load_data(self):
        """ 从 HistoryManager 重新加载数据并刷新两张表格 """
        self.history_mgr = HistoryManager()
        
        # 标定记录
        calib_records = self.history_mgr.get_calibration_records()
        self._calib_ids = [r["id"] for r in calib_records]
        self.calib_table.setRowCount(len(calib_records))
        for i, r in enumerate(calib_records):
            items = [
                QTableWidgetItem(r.get("name", "")),
                QTableWidgetItem(r.get("date", "")),
                QTableWidgetItem(str(r.get("image_count", ""))),
                QTableWidgetItem(f"{r.get('avg_error', 0):.4f}"),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                self.calib_table.setItem(i, col, item)

        # 逆透视记录
        persp_records = self.history_mgr.get_perspective_records()
        self._persp_ids = [r["id"] for r in persp_records]
        self.persp_table.setRowCount(len(persp_records))
        for i, r in enumerate(persp_records):
            summary = (f"obj({r.get('objdx','')}×{r.get('objdy','')}) "
                       f"img({r.get('imgdx','')},{r.get('imgdy','')}) "
                       f"out({r.get('out_w','')}×{r.get('out_h','')})")
            items = [
                QTableWidgetItem(r.get("name", "")),
                QTableWidgetItem(r.get("date", "")),
                QTableWidgetItem(summary),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                self.persp_table.setItem(i, col, item)

    # ---------- 获取选中行 ID ----------
    def _selected_id(self):
        is_calib = (self.stacked.currentIndex() == 0)
        table = self.calib_table if is_calib else self.persp_table
        ids = self._calib_ids if is_calib else self._persp_ids
        selected = table.selectedItems()
        if not selected:
            w = MessageBox("提示", "请先在表格中选择一条记录", self.window())
            w.yesButton.setText("确定")
            w.cancelButton.setText("取消")
            w.exec()
            return None, is_calib
        row = selected[0].row()
        if row < len(ids):
            return ids[row], is_calib
        return None, is_calib

    # ---------- 槽函数 ----------
    def _on_apply_record(self):
        """ 将选中记录的参数应用到当前内存中 """
        record_id, is_calib = self._selected_id()
        if record_id is None:
            return
        mw = self.window()
        if is_calib:
            rec = self.history_mgr.get_calibration_record_by_id(record_id)
            if rec is None:
                return
            K = np.array(rec["K"], dtype=np.float64)
            D = np.array(rec["D"], dtype=np.float64).reshape(-1, 1)
            ci = mw.calibration_interface
            ci._K = K
            ci._D = D
            ci.calib_status_label.setText(
                f"✅ 已应用历史参数\n来源: {rec.get('name', '')}"
            )
            w = MessageBox("应用成功", f"已将「{rec['name']}」的 K/D 参数加载到内存中。", mw)
            w.yesButton.setText("确定")
            w.cancelButton.setText("取消")
            w.exec()
        else:
            rec = self.history_mgr.get_perspective_record_by_id(record_id)
            if rec is None:
                return
            si = mw.settings_interface
            si.input_objdx.setText(str(rec.get("objdx", "")))
            si.input_objdy.setText(str(rec.get("objdy", "")))
            si.input_imgdx.setText(str(rec.get("imgdx", "")))
            si.input_imgdy.setText(str(rec.get("imgdy", "")))
            si.input_result_w.setText(str(rec.get("out_w", "")))
            si.input_result_h.setText(str(rec.get("out_h", "")))
            w = MessageBox("应用成功", f"已将「{rec['name']}」的参数填入设置页面。", mw)
            w.yesButton.setText("确定")
            w.cancelButton.setText("取消")
            w.exec()

    def _on_rename_record(self):
        """ 重命名选中记录 """
        record_id, is_calib = self._selected_id()
        if record_id is None:
            return
        dlg = CustomInputDialog("重命名", "请输入新的记录名称:", self)
        if dlg.exec_() != QDialog.Accepted or not dlg.get_text():
            return
        new_name = dlg.get_text()
        try:
            if is_calib:
                self.history_mgr.rename_record("calibration_records", record_id, new_name)
            else:
                self.history_mgr.rename_record("perspective_records", record_id, new_name)
            self.load_data()
        except ValueError as e:
            w = MessageBox("重命名失败", str(e), self.window())
            w.yesButton.setText("确定")
            w.cancelButton.setText("取消")
            w.exec()

    def _on_export_yaml(self):
        """ 将选中标定记录导出为 YAML """
        record_id, is_calib = self._selected_id()
        if record_id is None or not is_calib:
            return
        rec = self.history_mgr.get_calibration_record_by_id(record_id)
        if rec is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出标定参数", f"{rec.get('name', 'calibration')}.yaml",
            "YAML (*.yaml *.yml);;All (*)",
        )
        if not path:
            return
        K = np.array(rec["K"], dtype=np.float64)
        D = np.array(rec["D"], dtype=np.float64).reshape(-1, 1)
        self.history_mgr.export_calibration_to_yaml(path, K, D)
        w = MessageBox("导出成功", f"YAML 已保存到:\n{path}", self.window())
        w.yesButton.setText("确定")
        w.cancelButton.setText("取消")
        w.exec()

    def _on_delete_record(self):
        """ 删除选中记录 """
        record_id, is_calib = self._selected_id()
        if record_id is None:
            return
        w = MessageBox("确认删除", "确定要删除这条记录吗？", self.window())
        w.yesButton.setText("确定")
        w.cancelButton.setText("取消")
        if w.exec():
            if is_calib:
                self.history_mgr.delete_calibration_record(record_id)
            else:
                self.history_mgr.delete_perspective_record(record_id)
            self.load_data()

    # ---------- 别名 ----------
    def refresh_tables(self):
        self.load_data()


# ============================================================
# 主窗口
# ============================================================
class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vision Transform Master")
        self.setMinimumSize(950, 750)
        self.setWindowIcon(QIcon("assets/logo.ico"))
        
        # 创建子接口页面
        self.home_interface = HomeInterface(self)
        self.calibration_interface = CalibrationInterface(self)
        self.perspective_interface = PerspectiveInterface(self)
        self.history_interface = HistoryInterface(self)
        self.settings_interface = SettingsInterface(self)

        self._init_navigation()

    def _init_navigation(self):
        # 顶部导航
        self.addSubInterface(
            self.home_interface,
            FIF.HOME,
            '首页'
        )
        self.addSubInterface(
            self.calibration_interface,
            FIF.CAMERA,
            '相机标定'
        )
        self.addSubInterface(
            self.perspective_interface,
            FIF.PHOTO,
            '逆透视操作'
        )
        self.addSubInterface(
            self.history_interface,
            FIF.HISTORY,
            '历史记录'
        )

        # 底部导航
        self.addSubInterface(
            self.settings_interface,
            FIF.SETTING,
            '参数设置',
            NavigationItemPosition.BOTTOM
        )


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    # 全局强制深色主题 (必须在窗口实例化之前)
    setTheme(Theme.DARK)

    # 全局字体设置：微软雅黑，字号 10，字重 Medium
    font = QFont("Microsoft YaHei", 10, QFont.Medium)
    app.setFont(font)

    # 初始化并显示主窗口
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
