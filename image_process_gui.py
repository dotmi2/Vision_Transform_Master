import os
import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt

from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, setTheme, Theme,
    FluentIcon as FIF
)

from views.home_view import HomeInterface
from views.settings_view import SettingsInterface
from views.calibration_view import CalibrationInterface
from views.history_view import HistoryInterface
from views.perspective_view import PerspectiveInterface









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
