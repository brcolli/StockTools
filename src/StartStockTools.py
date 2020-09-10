import subprocess
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QAction, QTabWidget
from PyQt5.QtGui import QPalette, QColor


##################################
# Class controllers for each tab #
##################################


class UpComingEarningsScannerTab(QWidget):

    def __init__(self):

        super().__init__()

        vbox = QVBoxLayout()

        get_earnings = QPushButton('Get earnings')

        vbox.addWidget(get_earnings)

        self.setLayout(vbox)


class TwitterSentimentAnalysis(QWidget):

    def __init__(self):

        super().__init__()

        vbox = QVBoxLayout()

        get_sentiment = QPushButton('Get Twitter sentiment')

        vbox.addWidget(get_sentiment)

        self.setLayout(vbox)


class ShortInterestParser(QWidget):

    def __init__(self):

        super().__init__()

        vbox = QVBoxLayout()

        get_sentiment = QPushButton('Parse short interest file')

        vbox.addWidget(get_sentiment)

        self.setLayout(vbox)


############################
# Main application manager #
############################


class QtManager:

    def __init__(self):

        self.app = QApplication([])
        self.window = QWidget()

        # Set window look
        self.window.setWindowTitle('Stock Tools')
        self.window.setFixedSize(1500, 1000)
        self.set_dark_theme()

        ###########################
        # Add tabs and style them #
        ###########################

        vbox = QVBoxLayout()

        ts = self.return_stylesheet('../styles/dark_tabs.css')

        tab_widget = QTabWidget()
        tab_widget.setAutoFillBackground(True)
        tab_widget.setStyleSheet(ts)

        tab_widget.addTab(UpComingEarningsScannerTab(), 'Upcoming Earnings Scanner')
        tab_widget.addTab(TwitterSentimentAnalysis(), 'Twitter Sentiment Analysis')
        tab_widget.addTab(ShortInterestParser(), 'Short Interest Parser')

        vbox.addWidget(tab_widget)

        # Add widget layout to window
        self.window.setLayout(vbox)

    def run_app(self):
        self.window.show()
        self.app.exec_()

    def set_dark_theme(self):

        dark_palette = QPalette()

        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.darkRed)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.darkRed)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.darkRed)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)

        self.app.setPalette(dark_palette)

    @staticmethod
    def return_stylesheet(filename):
        with open(filename, 'r') as f:
            return f.read()


if __name__ == "__main__":

    qtm = QtManager()

    qtm.run_app()
