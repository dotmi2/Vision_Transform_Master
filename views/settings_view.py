from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QWidget, QDialog
from PyQt5.QtCore import Qt

from qfluentwidgets import (
    ScrollArea, SimpleCardWidget, SubtitleLabel, BodyLabel,
    LineEdit, PushButton, PrimaryPushButton, TransparentPushButton, MessageBox
)

from history_manager import HistoryManager
from views.components import CustomInputDialog


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
