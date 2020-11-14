import importlib


sm = importlib.import_module('Scheduler').ScheduleManager


def main():

    sm.loop_schedule_task_days(sm.call_daily_short_interest, 1, '16:00', '23:58')


if __name__ == '__main__':
    main()