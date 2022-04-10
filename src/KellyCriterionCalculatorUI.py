import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from KellyDesign import Ui_MainWindow


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalsSlots()

        self.portValue.valueChanged.connect(self.p_value)
        self.probWinning.valueChanged.connect(self.p_winning)
        self.odds.valueChanged.connect(self.o_winning)
        self.fracKelly.valueChanged.connect(self.f_kelly)

        self.clearButton.clicked.connect(self.clear_values)

        self.PFValue = 0
        self.PoW = 0
        self.Odds = 0.1
        self.FKelly = 0
        self.AmountBet = 0
        self.PercentageBet = 0
        self.set_initial_values()

    def set_initial_values(self):
        self.PFValue = self.portValue.value()
        self.PoW = self.probWinning.value()
        self.Odds = self.odds.value()
        self.FKelly = self.fracKelly.value()
        self.update_outcomes()

    def clear_values(self):
        self.PFValue = 0
        self.PoW = 0
        self.Odds = 0.1
        self.FKelly = 0
        self.AmountBet = 0
        self.PercentageBet = 0

        self.portValue.clear()
        self.probWinning.clear()
        self.odds.clear()
        self.fracKelly.clear()
        self.percentOut.clear()
        self.amountOut.clear()

    def p_value(self, num):
        self.PFValue = num
        self.update_outcomes()

    def p_winning(self, num):
        self.PoW = num
        self.update_outcomes()

    def o_winning(self, num):
        self.Odds = num
        self.update_outcomes()

    def f_kelly(self, num):
        self.FKelly = num
        self.update_outcomes()

    def update_outcomes(self):
        win = self.PoW / 100
        loss = 1 - (self.PoW/100)

        self.PercentageBet = (win - (loss / self.Odds)) * (self.FKelly / 100)
        self.AmountBet = self.PercentageBet * self.PFValue

        self.percentOut.setValue(self.PercentageBet * 100)
        self.amountOut.setValue(self.AmountBet)


    def callto(self):
        print('K')

    def connectSignalsSlots(self):
        self.amountOut.update()
        # self.action_Exit.triggered.connect(self.close)
        # self.action_Find_Replace.triggered.connect(self.findAndReplace)
        # self.action_About.triggered.connect(self.about)


app = QApplication(sys.argv)
win = Window()
win.show()
sys.exit(app.exec())
