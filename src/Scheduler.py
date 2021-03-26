import schedule
import time
import importlib
import datetime
import enum


sim = importlib.import_module('GetDailyShortInterest').ShortInterestManager
nso = importlib.import_module('NasdaqShareOrdersScraper').NasdaqShareOrdersManager
UES = importlib.import_module('UpcomingEarningsScanner').main
Utils = importlib.import_module('utilities').Utils


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

        sim_obj = sim()
        res = sim_obj.get_latest_short_interest_data()
        for r in res:
            sub_dir = '/'.join(r.split('/')[2:-1])  # Just get subdirectory path
            Utils.upload_file_to_gdrive(r, 'Daily Short Data')

    @staticmethod
    def call_nasdaq_share_orders():
        nso_obj = nso()
        nso_obj.write_nasdaq_trade_orders()

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


def main():

    sch = ScheduleManager()


if __name__ == "__main__":
    main()
