import subprocess
import sys
from PyQt5.QtWidgets import QApplication


class QtManager:

    def __init__(self):
        self.app = QApplication([])

    def run_app(self):
        self.app.exec_()


if __name__ == "__main__":

    qtm = QtManager()

    qtm.run_app()