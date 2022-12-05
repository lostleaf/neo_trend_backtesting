from datetime import timedelta


def to_timedelta(interval: str) -> timedelta:
    if interval[-1] == 'm' or interval[-1] == 'T':
        return timedelta(minutes=int(interval[:-1]))
    if interval[-1] == 'h' or interval[-1] == 'H':
        return timedelta(hours=int(interval[:-1]))
    raise ValueError(f'Unknown time interval {interval}')