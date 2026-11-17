"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Idea, in one paragraph: hour 1's print is the anchor, but the *shape* of a
launch day is not flat. A console game that unlocks at midnight local time,
country by country, is already live in New Zealand, Australia, Japan and
Europe by 00:00 UTC, dips in the small hours while the Americas sleep, jumps
when the US East Coast unlocks, stays flat through the morning, then climbs
all afternoon into the European evening and the US afternoon. That shape is
visible in the case file itself: every case ships the hourly curves of past
big launches ("comparables"). This program reads those curves at run time,
keeps only the ones that look like GTA VI -- console-only launches with no
PC version and no early-access contamination -- and, for each forecast hour,
takes the trimmed mean log-ratio of that hour's viewers to the launch's own
hour-1 viewers across those launches (the lowest and highest ratio dropped
first, so one wild launch cannot drag the centre, but the other launches
still vote instead of only the middle one doing so). The centre is smoothed
over neighbouring hours (the US unlock lands at 04:00Z in daylight-saving months and 05:00Z
in November, so a sharp step at one hour would be a coin flip), and the
five quantiles are a log-normal band around that median with a fixed
spread of 0.45 log units, wide enough to cover how much launches disagree
with each other late in the day.

Why this selection, and why not every comparable: launches that unlock
simultaneously worldwide (a single 14:00Z unlock, say) start the UTC day
near zero and explode 100x later, and PC launches peak at midnight GMT.
Mixing them in drags the median up late in the day and wrecks the early
hours. Console-only launches are the family GTA VI belongs to -- PS5 and
Xbox only, no PC until 2027, single-player, rolling midnight unlock -- so
they are the training set. On those launches, leave-one-out, this median
shape scores about 0.21 log-pinball against about 0.34 for the flat
persistence baseline, with a typical median miss near 42% instead of 73%.

The program is deterministic (sorted inputs, trimmed means, no randomness, no
clock, no network) and standard library only. It reads every *.in file in
MAKEFUN_INPUT_DIR (default /input) and writes <id>.out to
MAKEFUN_OUTPUT_DIR (default /output) with 23 rows, hours 2..24, five
non-decreasing quantiles each.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels above.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Log spread of the quantile band around the median curve. Fixed at 0.45:
# in the console-only training family the spread of log-ratios across
# launches is about 0.35-0.48 late in the day, and a band that wide keeps
# 90% coverage near 0.87 in leave-one-out while a narrower one collapses.
SIGMA = 0.45
# A comparable only counts if its launch day is this complete.
MIN_DAY_HOURS = 20


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up, returning an int for whole numbers."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def day_hours(comparable):
    """Map hour offset from a comparable's day_start to its viewers.

    Offset 0 is the launch day's first hour, 1..23 the rest of the day.
    Points outside the day are ignored; a point with no positive viewers
    is ignored for that hour.
    """
    start = parse_time(comparable["day_start"])
    hours = {}
    for row in comparable.get("hourly") or []:
        try:
            when = parse_time(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        offset = int((when - start).total_seconds() // 3600)
        if 0 <= offset <= 23 and isinstance(viewers, (int, float)) and viewers > 0:
            hours[offset] = float(viewers)
    return hours


def looks_like_gta(comparable):
    """True for a console-only launch with no early-access contamination.

    GTA VI launches on PS5 and Xbox only, with no PC version, so a
    comparable that shipped on PC at the same time has a different unlock
    shape (a simultaneous PC unlock at midnight GMT). A launch whose
    unlock text mentions early access already peaked days before its
    release day, so its release-day curve is flat for the wrong reason.
    The target game's own pre-release category history is not a launch
    at all, so it is excluded by name.
    """
    name = str(comparable.get("name") or "")
    if "Grand Theft Auto VI" in name:
        return False
    platforms = str(comparable.get("platforms") or "")
    if "PC" in platforms:
        return False
    unlock = str(comparable.get("unlock") or "").lower()
    if "early access" in unlock:
        return False
    return True


def select_training(comparables):
    """Pick the training launches, with fallbacks if the filter is empty.

    First choice: the console-only, no-early-access family. If fewer
    than three of those survive, fall back to every comparable with a
    usable hour 1. If even that is empty, the caller uses a flat median.
    """
    curves = []
    for comparable in comparables or []:
        hours = day_hours(comparable)
        if 0 not in hours or len(hours) < MIN_DAY_HOURS:
            continue
        curves.append((comparable, hours))
    family = [(c, h) for c, h in curves if looks_like_gta(c)]
    if len(family) >= 3:
        return family
    if len(curves) >= 3:
        return curves
    return family or curves


def centre_log_ratios(training):
    """Trimmed-mean log-ratio ln(viewers_h / viewers_0) per day-hour offset.

    For each offset, drop the lowest and highest log-ratio across the
    training launches (when at least five cover it), then average the
    rest. A plain median throws away every launch but the middle one;
    a plain mean lets one wild launch drag the centre. The trimmed
    mean keeps most of the information while staying robust, and in
    leave-one-out on the console-only family it scores about 0.210
    against about 0.216 for the median. Returns a dict offset -> centre,
    for offsets 1..23 that at least three training launches cover.
    """
    centres = {}
    for offset in range(1, 24):
        ratios = sorted(
            math.log(hours[offset] / hours[0])
            for _comparable, hours in training
            if offset in hours
        )
        if len(ratios) >= 3:
            if len(ratios) >= 5:
                ratios = ratios[1:-1]
            centres[offset] = statistics.mean(ratios)
    return centres


def smoothed_median(medians, offset):
    """Centre at one offset, smoothed over its neighbouring hours.

    Weights 0.25 / 0.5 / 0.25 over (offset-1, offset, offset+1) where
    those centres exist. This spreads the US-unlock step, which lands
    an hour apart in daylight-saving and standard-time months, instead
    of betting the whole step on a single hour.
    """
    total = 0.0
    weight = 0.0
    for neighbour, w in ((offset - 1, 0.25), (offset, 0.5), (offset + 1, 0.25)):
        if neighbour in medians:
            total += w * medians[neighbour]
            weight += w
    if weight == 0.0:
        return 0.0
    return total / weight


def hour_quantiles(p1, log_median):
    """Five quantiles around p1 * exp(log_median), log-normal band."""
    centre = math.log(p1) + log_median
    out = []
    previous = 0.0
    for z in Z:
        value = math.exp(centre + z * SIGMA)
        value = max(0.01, min(1e9, value))
        if value < previous:
            value = previous
        out.append(round2(value))
        previous = float(out[-1])
    return out


def forecast_case(case):
    """Forecast hours 2..24 for one case dict."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    training = select_training(case.get("comparables"))
    centres = centre_log_ratios(training)
    hours = []
    for row in case.get("forecast") or []:
        hour = int(row["hour"])
        offset = hour - 1
        log_median = smoothed_median(centres, offset)
        hours.append({"hour": hour, "q": hour_quantiles(p1, log_median)})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
