import Scheduler


sm = Scheduler.ScheduleManager
ds = Scheduler.Days


def main():

    start_time = '16:00'
    end_time = '23:58'

    sm.loop_schedule_task_days(sm.call_daily_short_interest, 1, start_time)
    sm.loop_schedule_task_days(sm.call_nasdaq_share_orders, 1, start_time)
    sm.loop_schedule_task_weekly(sm.call_upcoming_earnings_scanner, ds.Sun, start_time)

    sm.run_scheduled_tasks(start_time, end_time)


if __name__ == '__main__':
    main()
