import dateparser
from datetime import datetime, timedelta
import re

def parse_natural_date_range(query: str):
    query = query.lower()

    today = datetime.today()
    start_date, end_date = None, None

    # Last month
    if "last month" in query:
        first_day_this_month = today.replace(day=1)
        end_date = first_day_this_month - timedelta(days=1)
        start_date = end_date.replace(day=1)

    # This month
    elif "this month" in query:
        start_date = today.replace(day=1)
        end_date = today

    # Last X days
    match = re.search(r'last (\d+) days?', query)
    if match:
        days = int(match.group(1))
        start_date = today - timedelta(days=days)
        end_date = today

    # Specific month e.g. "April"
    months = {
        'january': 1, 'february': 2, 'march': 3,
        'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9,
        'october': 10, 'november': 11, 'december': 12
    }
    for month_name, month_num in months.items():
        if month_name in query:
            year = today.year
            if f"{month_name} {year-1}" in query:
                year = year - 1
            start_date = datetime(year, month_num, 1)
            if month_num == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)
            break

    # Last week
    if "last week" in query:
        start_date = today - timedelta(days=today.weekday() + 7)
        end_date = start_date + timedelta(days=6)

    # Yesterday
    if "yesterday" in query:
        start_date = today - timedelta(days=1)
        end_date = start_date

    if start_date and end_date:
        return start_date.date(), end_date.date()

    return None, None