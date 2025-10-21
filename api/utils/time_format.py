def ms_to_hms_pad(ms: int) -> str:
    """
    Convert a duration from milliseconds to a human-readable string in HH:MM:SS format.

    Parameters:
        ms (int): The duration in milliseconds.

    Returns:
        str: A string representing the duration in hours, minutes, and seconds,
             zero-padded to two digits each component (e.g. "00:05:27").

    Example:
        >>> ms_to_hms_pad(27000)
        "00:00:27"
        >>> ms_to_hms_pad(3661000)
        "01:01:01"
    """
    total_seconds = ms // 1000
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
