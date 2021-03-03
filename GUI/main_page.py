from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

class App(QApplication):
    def __init__(self, args):
        super().__init__(args)
        self.window = Window()

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(1000, 1000, 300, 300)
        self.setWindowTitle("Stock App")

        button = QPushButton('Home', self)
        ##button.setToolTip()
        button.move(10,30)
        button.setMask(QRegion(QPolygon()))
        button.setStyleSheet("color: light blue; background-color: Light Tan")
        button.isFlat()
        button.clicked.connect(self.on_click)

    @pyqtSlot()
    def on_click(self):
        print('Home button click')


app = App(sys.argv)
app.window.show()
sys.exit(app.exec_())
##window()

