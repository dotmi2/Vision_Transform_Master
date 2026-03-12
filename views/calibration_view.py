import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QDialog,
    QStackedWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QIcon
from PyQt5.QtCore import Qt, pyqtSignal

from qfluentwidgets import (
    ScrollArea, SimpleCardWidget, SubtitleLabel, BodyLabel, CaptionLabel,
    PushButton, PrimaryPushButton, TransparentPushButton, MessageBox, TableWidget, SegmentedWidget, TitleLabel
)

from board import BoardManager
from calibration import CalibrationEngine
from history_manager import HistoryManager
from image_process import K as DEFAULT_K, D as DEFAULT_D
from views.components import CustomInputDialog

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
