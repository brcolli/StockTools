import schedule
import time
import importlib
import datetime


sim = importlib.import_module('GetDailyShortInterest').ShortInterestManager
Utils = importlib.import_module('utilities').Utils


class ScheduleManager:

    @staticmethod
    def call_daily_short_interest():

        res = sim.get_latest_short_interest_data()
        Utils.upload_file_to_gdrive(res, 'Daily Short Data')

    @staticmethod
    def loop_schedule_task_days(task, num_days=1, dtime_lower='00:00', dtime_upper='23:59'):

        schedule.every(num_days).day.at(dtime_lower).do(task)

        d_lower = datetime.datetime.strptime(dtime_lower, '%H:%M').time()
        d_upper = datetime.datetime.strptime(dtime_upper, '%H:%M').time()

        while True:

            schedule.run_pending()
            time.sleep(1800)  # Only check every 30 minutes

            n = datetime.datetime.now()
            curr_time = n.time()

            next_run_date = schedule.next_run()

            if d_lower < curr_time < d_upper and n < next_run_date and n.day == next_run_date.day:
                schedule.run_all()


def main():
    sch = ScheduleManager()


if __name__ == "__main__":
    main()
