import importlib


sm = importlib.import_module('Scheduler').ScheduleManager


def main():

    sm.loop_schedule_task_days(sm.call_daily_short_interest, 1, '16:00', '23:58')
    sm.loop_schedule_task_days(sm.call_nasdaq_share_orders, 1, '16:00', '23:58')
    sm.loop_schedule_task_days(sm.call_upcoming_earnings_scanner, 7, '16:00', '23:58')


if __name__ == '__main__':
    main()
