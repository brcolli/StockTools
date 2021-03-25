import importlib


sm = importlib.import_module('Scheduler').ScheduleManager
ds = importlib.import_module('Scheduler').Days


def main():

    sm.loop_schedule_task_days(sm.call_daily_short_interest, 1, '16:00', '23:58')
    sm.loop_schedule_task_days(sm.call_nasdaq_share_orders, 1, '16:00', '23:58')
    sm.loop_schedule_task_weekly(sm.call_upcoming_earnings_scanner, ds.Sun, '16:00', '23:58')


if __name__ == '__main__':
    main()
