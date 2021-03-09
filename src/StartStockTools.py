import sys
import importlib
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLineEdit, QLabel, QMainWindow


StockToolsUI = importlib.import_module('StockToolsUI').Ui_MainWindow
SIM = importlib.import_module('GetDailyShortInterest').ShortInterestManager
NSO = importlib.import_module('NasdaqShareOrdersScraper').NasdaqShareOrdersManager

##################################
# Class controllers for each tab #
##################################


class UpComingEarningsScannerTab(QWidget):

    def __init__(self, ues):

        super().__init__()
        self.ues = ues

        vbox = QVBoxLayout()

        # @TODO refactor this
        self.min_volume_label = QLabel(self)
        self.min_volume_label.setText('Minimum volume:')
        self.min_volume_edit = QLineEdit(self)

        self.min_market_cap_label = QLabel(self)
        self.min_market_cap_label.setText('Minimum market cap:')
        self.min_market_cap_edit = QLineEdit(self)

        self.min_last_closed_label = QLabel(self)
        self.min_last_closed_label.setText('Minimum last closed:')
        self.min_last_closed_edit = QLineEdit(self)

        get_earnings = QPushButton('Get earnings')
        get_earnings.clicked.connect(self.call_upcoming_earnings_scanner)

        vbox.addWidget(self.min_volume_label)
        vbox.addWidget(self.min_volume_edit)
        vbox.addWidget(self.min_market_cap_label)
        vbox.addWidget(self.min_market_cap_edit)
        vbox.addWidget(self.min_last_closed_label)
        vbox.addWidget(self.min_last_closed_edit)
        vbox.addWidget(get_earnings)

        self.setLayout(vbox)

    def call_upcoming_earnings_scanner(self):
        self.ues.main(int(self.min_volume_edit.text()), int(self.min_market_cap_edit.text()),
                      int(self.min_last_closed_edit.text()))


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


class QtManager(QMainWindow, StockToolsUI):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setupUi(self)
        self.connectSignalsSlots()

    def connectSignalsSlots(self):

        self.getLatestDailyShortInterestButton.clicked.connect(self.triggerGetLatestDailyShortInterest)
        self.GetNasdaqShareOrdersButton.clicked.connect(self.triggerGetNasdaqShareOrders)

    def triggerGetLatestDailyShortInterest(self):
        sim = SIM()
        sim.get_latest_short_interest_data()

    def triggerGetNasdaqShareOrders(self):
        nso = NSO()
        nso.write_nasdaq_trade_orders()


def main():

    app = QApplication(sys.argv)
    qtm = QtManager()

    qtm.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
