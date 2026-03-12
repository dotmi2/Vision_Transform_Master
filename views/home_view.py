import os
import shutil
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from qfluentwidgets import (
    TitleLabel, BodyLabel, CaptionLabel, PushButton
)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

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
