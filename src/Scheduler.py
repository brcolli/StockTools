import schedule
import time
import datetime
import enum
import GetDailyShortInterest
import NasdaqShareOrdersScraper
import UpcomingEarningsScanner
import utilities


sim = GetDailyShortInterest.ShortInterestManager
nso = NasdaqShareOrdersScraper.NasdaqShareOrdersManager
UES = UpcomingEarningsScanner.main
Utils = utilities.Utils


# For representing days of the week
class Days(enum.Enum):
    Sun = 1
    Mon = 2
    Tue = 3
    Wed = 4
    Thu = 5
    Fri = 6
    Sat = 7


class ScheduleManager:

    @staticmethod
    def call_daily_short_interest():

        # If not a trading day, don't run
        if Utils.get_last_trading_day().date() != datetime.datetime.today().date():
            return

        sim_obj = sim()
        res = sim_obj.get_latest_short_interest_data()

        sub_dir = 'Daily Short Data/' + '/'.join(res[0].split('/')[2:-1])  # Just get subdirectory path
        Utils.upload_files_to_gdrive(res, sub_dir)

    @staticmethod
    def call_nasdaq_share_orders():

        # If not a trading day, don't run
        if Utils.get_last_trading_day().date() != datetime.datetime.today().date():
            return

        nso.write_nasdaq_trade_orders()

    @staticmethod
    def call_upcoming_earnings_scanner():
        UES()

    @staticmethod
    def loop_schedule_task_weekly(task, day, dtime_lower='00:00'):

        if day == Days.Sun:
            schedule.every().sunday.at(dtime_lower).do(task)
        elif day == Days.Mon:
            schedule.every().monday.at(dtime_lower).do(task)
        elif day == Days.Tue:
            schedule.every().tuesday.at(dtime_lower).do(task)
        elif day == Days.Wed:
            schedule.every().wednesday.at(dtime_lower).do(task)
        elif day == Days.Thu:
            schedule.every().thursday.at(dtime_lower).do(task)
        elif day == Days.Fri:
            schedule.every().friday.at(dtime_lower).do(task)
        elif day == Days.Sat:
            schedule.every().saturday.at(dtime_lower).do(task)

    @staticmethod
    def loop_schedule_task_days(task, num_days=1, dtime_lower='00:00'):

        schedule.every(num_days).day.at(dtime_lower).do(task)

    @staticmethod
    def run_scheduled_tasks(dtime_lower, dtime_upper):

        d_lower = datetime.datetime.strptime(dtime_lower, '%H:%M').time()
        d_upper = datetime.datetime.strptime(dtime_upper, '%H:%M').time()

        next_run_date = datetime.datetime.today()

        while True:

            schedule.run_pending()

            n = datetime.datetime.now()
            curr_time = n.time()

            if d_lower < curr_time < d_upper and n.date() == next_run_date.date():
                schedule.run_all()

            next_run_date = schedule.next_run()

            time.sleep(1800)  # Only check every 30 minutes
            print('Checked to run at: {}.'.format(datetime.datetime.now().time()))


def main():
    sch = ScheduleManager()


if __name__ == "__main__":
    main()
