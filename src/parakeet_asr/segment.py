import csv
import datetime
from pathlib import Path
from typing import List


class AlignedSegment:
    """A class to hold a text segment, start time, and end time."""

    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Segment(text='{self.text[:20]}...', start={self.start:.2f}, end={self.end:.2f})"

    def __str__(self):
        return self.__repr__()


def generate_srt_from_segments(segments: List[AlignedSegment]) -> str:
    """Generates SRT content from a list of AlignedSegment objects."""
    srt_content = []
    for i, seg in enumerate(segments):
        start_time = format_srt_time(seg.start)
        end_time = format_srt_time(seg.end)
        srt_content.append(str(i + 1))
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(seg.text)
        srt_content.append("")
    return "\n".join(srt_content)


def format_srt_time(seconds: float) -> str:
    """Converts seconds to SRT time format HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    delta = datetime.timedelta(seconds=seconds)
    total_int_seconds = int(delta.total_seconds())
    hours = total_int_seconds // 3600
    minutes = (total_int_seconds % 3600) // 60
    seconds_part = total_int_seconds % 60
    milliseconds = delta.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"
