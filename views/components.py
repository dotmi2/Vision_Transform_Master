from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt

from qfluentwidgets import LineEdit, PushButton, PrimaryPushButton


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
