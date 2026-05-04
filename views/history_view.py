import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QDialog,
    QStackedWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt

from qfluentwidgets import (
    ScrollArea, SimpleCardWidget, SubtitleLabel, PushButton, PrimaryPushButton, 
    TransparentPushButton, MessageBox, TableWidget, SegmentedWidget
)

from history_manager import HistoryManager
from views.components import CustomInputDialog

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
            img_count = r.get("image_count", 0)
            avg_err = r.get("avg_error", 0)

            # 如果是 0，说明是外部导入的 YAML，显示更友好的文字
            img_count_str = str(img_count) if img_count > 0 else "外部导入"
            avg_err_str = f"{avg_err:.4f}" if img_count > 0 else "-"

            items = [
                QTableWidgetItem(r.get("name", "")),
                QTableWidgetItem(r.get("date", "")),
                QTableWidgetItem(img_count_str),
                QTableWidgetItem(avg_err_str),
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
            ci._model = rec.get("model", "pinhole")
            if rec.get("image_width") and rec.get("image_height"):
                ci._calib_size = (int(rec["image_width"]), int(rec["image_height"]))
            else:
                ci._calib_size = (640, 480)
            ci.calib_status_label.setText(
                f"✅ 已应用历史参数\n来源: {rec.get('name', '')}\n"
                f"模型: {ci._model}  分辨率: {ci._calib_size[0]}×{ci._calib_size[1]}"
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
        image_size = None
        if rec.get("image_width") and rec.get("image_height"):
            image_size = (int(rec["image_width"]), int(rec["image_height"]))
        self.history_mgr.export_calibration_to_yaml(
            path,
            K,
            D,
            model=rec.get("model", "pinhole"),
            image_size=image_size,
            reprojection_error=rec.get("avg_error"),
        )
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
