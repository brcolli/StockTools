import sys
import importlib
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLineEdit, QLabel, QMainWindow


StockToolsUI = importlib.import_module('StockToolsUI').Ui_MainWindow
SIM = importlib.import_module('GetDailyShortInterest').main
NSO = importlib.import_module('NasdaqShareOrdersScraper').main
UES = importlib.import_module('UpcomingEarningsScanner').main
NSA = importlib.import_module('NewsSentimentAnalysis').main

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

        self.getDailyShortInterestButton.clicked.connect(self.triggerGetDailyShortInterest)
        self.getNasdaqShareOrdersButton.clicked.connect(self.triggerGetNasdaqShareOrders)
        self.upcomingEarningsButton.clicked.connect(self.triggerUpcomingEarningsScanner)
        self.newsSentimentAnalysisButton.clicked.connect(self.triggerNewsSentimentAnalysis)

    def triggerGetDailyShortInterest(self):

        ymd1 = self.dailyShortInterestStartDate.text()
        ymd2 = self.dailyShortInterestEndDate.text()
        should_upload = self.dailyShortInterestUploadToDriveCheck.isChecked()

        SIM(ymd1, ymd2, should_upload)

    def triggerUpcomingEarningsScanner(self):

        ymd1 = self.upcomingEarningsStartDate.text()
        ymd2 = self.upcomingEarningsStartDate.text()

        min_vl = self.upcomingEarningsMinVol.text()
        min_mc = self.upcomingEarningsMinMCap.text()
        min_lc = self.upcomingEarningsMinLastClose.text()

        if min_vl == '':
            min_vl = 1E6
        if min_mc == '':
            min_mc = 3E8
        if min_lc == '':
            min_lc = 10

        UES(ymd1, ymd2, min_vl, min_mc, min_lc)

    def triggerNewsSentimentAnalysis(self):

        phrase = self.newsSentimentPhrase.text()
        filter_in_str = self.newsSentimentFilterIn.text()
        filter_out_str = self.newsSentimentFilterOut.text()
        history_count = self.newsSentimentHistoryCount.text()

        if history_count == '':
            history_count = 1000

        # Parse filters to arrays
        if filter_in_str == '':
            filter_in = []
        else:
            filter_in = filter_in_str.split(',')

        if filter_out_str == '':
            filter_out = []
        else:
            filter_out = filter_out_str.split(',')

        NSA(phrase, filter_in, filter_out, int(history_count))

    @staticmethod
    def triggerGetNasdaqShareOrders():
        NSO()


def main():

    app = QApplication(sys.argv)
    qtm = QtManager()

    qtm.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
