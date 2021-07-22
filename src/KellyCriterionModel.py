import random
import pandas as pd


class KellyModel:
    """
    Used to simulate trading with the kelly model, either single simulation of n trades or multiple simulations of n
    trades.
    """
    def __init__(self, beg_capital=10000, edge=0.5, edge_range=0, glr=1.15, fractional_kelly=0.25, l_ceiling=0.2,
                 w_ceiling=0.5, num_runs=100, tax_rate=0.1, snp=0.1478, expected_return=0.0008):
        """
        :param beg_capital: Starting capital (USD)
        :type beg_capital: float
        :param edge: Average chance of winning a trade in (0, 1)
        :type edge: float
        :param  edge_range: The range for the edge for any specific trade:
                            (Current Edge = Edge + random.uniform(-range, +range)
        :type edge_range: float
        :param glr: Gain loss ratio in (0, 1)
        :type glr: float
        :param fractional_kelly: The fraction of the Kelly model to use in (0, 1)
        :type fractional_kelly: float
        :param l_ceiling: Loss ceiling percentage wise for any given trade in (0, 1)
        :type l_ceiling: float
        :param w_ceiling: Winn ceiling percentage wise for any give trade in (0, 1)
        :type w_ceiling: float
        :param num_runs: Number of trades to do perform in a single simulation
        :type num_runs: int
        :param tax_rate: The tax rate (used to calculate alpha) percentage wise in (0, 1)
        :type tax_rate: float
        :param snp: SNP 500 returns (used to calculate alpha) percentage wise in (0, 1)
        :type snp: float
        :param expected_return: Expected return per trade (used to calculate Sortino Ratio) percentage wise in (0, 1)
        :type expected_return: float
        """

        self.beg_capital, self.edge, self.edge_range, self.glr, self.fractional_kelly, self.l_ceiling, self.w_ceiling, \
        self.num_runs = \
            beg_capital, edge, edge_range, glr, fractional_kelly, l_ceiling, w_ceiling, num_runs

        self.tax_rate = tax_rate
        self.snp = snp
        self.expected_return = expected_return

        self.curr_capital = self.beg_capital
        self.num_plays = 0
        self.num_bets = 0
        self.num_wins = 0
        self.num_losses = 0
        self.largest_gain = 0
        self.largest_loss = 0
        self.largest_loss_streak = 0
        self.largest_win_streak = 0
        self.curr_streak = 0

        self.average_gain = 0
        self.average_loss = 0
        self.excess_return = 0
        self.negative_excess_return = 0
        self.square_negative_excess_return = 0

        self.win_v_loss = 0
        self.gross_return = 0
        self.net_return = 0
        self.alpha = 0
        self.win_rate = 0
        self.multiple_on_invested_capital = 0
        self.expected_gl = 0
        self.actual_gl = 0
        self.downside_risk = 0
        self.average_excess_return = 0
        self.sortino = 0

        self.ending_capital = []

        self.keys = ['Ending Capital List', 'Final Ending Capital', 'Wins vs Losses', 'Largest Winning Streak',
                     'Largest Losing Streak',
                     'Gross Return', 'Tax Rate', 'Net Return', 'YTD S&P 500 Return', 'Alpha', 'W', 'L',
                     'Total Trades', 'Win Rate', 'Multiple on Invested Capital', 'Largest Gain', 'Largest Loss',
                     'Average Gain', 'Average Loss', 'Expected Gain/(Loss) Per Trade', 'Actual Gain/Loss Ratio',
                     'Downside Risk', 'Average Excess Return', 'Sortino Ratio']

    def single_run(self):
        """
        Simulates a single trade and records the results
        """

        # Edge calculation for this specific trade using edge range
        cur_edge = self.edge + random.uniform(-1*self.edge_range, self.edge_range)

        # Kelly model bet size
        bet = (cur_edge - ((1 - cur_edge) / self.glr)) * self.fractional_kelly

        trade_return = 0
        self.num_plays += 1
        if bet > 0:
            self.num_bets += 1

            to_trade = bet * self.curr_capital

            # Win
            if random.random() <= cur_edge:
                self.num_wins += 1
                if self.curr_streak < 0:
                    self.curr_streak = 1
                else:
                    self.curr_streak += 1

                if self.curr_streak > self.largest_win_streak:
                    self.largest_win_streak = self.curr_streak

                percent_gain_realized = random.uniform(0, self.w_ceiling)
                trade_return = percent_gain_realized * 1 * to_trade
                diff_in_returns = percent_gain_realized - self.expected_return
                self.excess_return += diff_in_returns
                if diff_in_returns < 0:
                    self.negative_excess_return += diff_in_returns
                    self.square_negative_excess_return += ((-1 * diff_in_returns) ** 2)

                self.average_gain += trade_return
                if trade_return > self.largest_gain:
                    self.largest_gain = trade_return

            # Loss
            else:
                self.num_losses += 1
                if self.curr_streak > 0:
                    self.curr_streak = -1
                else:
                    self.curr_streak -= 1

                if self.curr_streak < self.largest_loss_streak:
                    self.largest_loss_streak = self.curr_streak

                percent_loss_realized = -1 * random.uniform(0, self.l_ceiling)
                trade_return = percent_loss_realized * to_trade
                diff_in_returns = percent_loss_realized - self.expected_return
                self.excess_return += diff_in_returns
                if diff_in_returns < 0:
                    self.negative_excess_return += diff_in_returns
                    self.square_negative_excess_return += ((-1 * diff_in_returns) ** 2)

                self.average_loss -= trade_return
                if (-1 * trade_return) > self.largest_loss:
                    self.largest_loss = (-1 * trade_return)

            self.curr_capital += trade_return
            self.ending_capital.append(self.curr_capital)

    def all_runs(self):
        """
        Simulates a trade (num_runs) times with each trade compounding on the previous
        """
        for i in range(self.num_runs):
            self.single_run()

    def calc_ending_scores(self):
        """
        Calculates ending variables after all the trades are completed for a single simulation
        """
        self.win_v_loss = self.num_wins - self.num_losses
        self.gross_return = (self.curr_capital - self.beg_capital) / self.beg_capital
        self.net_return = self.gross_return * (1 - self.tax_rate)
        self.alpha = self.net_return - self.snp
        self.win_rate = self.num_wins / self.num_bets
        self.multiple_on_invested_capital = self.curr_capital / self.beg_capital
        self.average_gain = (self.average_gain / self.num_wins)
        self.average_loss = (self.average_loss / self.num_losses)
        self.excess_return = self.excess_return / self.num_bets
        self.negative_excess_return = self.negative_excess_return / self.num_bets
        self.square_negative_excess_return = self.square_negative_excess_return / self.num_bets
        self.expected_gl = (self.average_gain * self.edge) - (self.average_loss * (1 - self.edge))
        self.actual_gl = self.average_gain / self.average_loss
        self.downside_risk = (self.square_negative_excess_return ** 0.5)
        self.average_excess_return = self.excess_return
        self.sortino = self.average_excess_return / self.downside_risk

    def to_dict(self):
        """
        Exports values of a single simulation to a dictionary
        """
        values = [self.ending_capital, self.curr_capital, self.win_v_loss, self.largest_win_streak,
                  self.largest_loss_streak,
                  self.gross_return, self.tax_rate, self.net_return, self.snp, self.alpha, self.num_wins,
                  self.num_losses, self.num_bets, self.win_rate, self.multiple_on_invested_capital, self.largest_gain,
                  self.largest_loss, self.average_gain, self.average_loss, self.expected_gl, self.actual_gl,
                  self.downside_risk, self.average_excess_return, self.sortino]

        return dict(zip(self.keys, values))

    def single_sim(self):
        """
        Wrapper for running a single simulation

        :return: Pandas dataframe with results
        :rtype: pd.DataFrame
        """
        self.all_runs()
        self.calc_ending_scores()
        return self.to_dict()

    def multi_sim(self, num):
        """
        Wrapper for running multiple simulations (ie running multiple sets of 100 trades to see what outcomes are
        possible)

        :param num: Number of simulation to run (ie how many times to simulate 100 trades in a row)\
        :type num: int

        :return: Pandas dataframe with results
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame()

        for i in range(num):
            self.__init__()
            df = df.append(self.single_sim(), ignore_index=True)
            df = df[self.keys]

        return df

    @staticmethod
    def save_df(df: pd.DataFrame, name: str):
        """
        Save a dataframe to a csv which can then be opened in Excel for analysis
        """
        df.to_csv(name, index=False)
