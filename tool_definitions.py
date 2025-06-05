from datetime import datetime
import pytz

def get_current_time(timezone: str = "UTC") -> str:
    tz = pytz.timezone(timezone)
    current_time = datetime.now(tz)
    return current_time.strftime("%-I:%M %p")
