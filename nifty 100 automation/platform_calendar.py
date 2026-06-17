from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MarketSession:
    date: date
    open_time: time
    close_time: time
    name: str = "regular"


class NSEExchangeCalendar:
    """
    File-backed NSE calendar with a conservative fallback.

    The calendar can be refreshed externally by writing `nse_calendar.json`
    next to the tracker. Keeping this file-backed avoids a brittle runtime
    dependency on one NSE endpoint while still allowing automatic updates from
    scheduled jobs or deployment config.
    """

    FALLBACK_HOLIDAYS: set[date] = {
        date(2024, 1, 22), date(2024, 1, 26), date(2024, 3, 25), date(2024, 3, 29),
        date(2024, 4, 11), date(2024, 4, 14), date(2024, 4, 17), date(2024, 4, 21),
        date(2024, 5, 23), date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15),
        date(2024, 10, 2), date(2024, 10, 12), date(2024, 10, 15), date(2024, 11, 1),
        date(2024, 11, 15), date(2024, 11, 20), date(2024, 12, 25),
        date(2025, 1, 26), date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
        date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1),
        date(2025, 6, 7), date(2025, 8, 15), date(2025, 8, 27), date(2025, 10, 2),
        date(2025, 10, 21), date(2025, 10, 22), date(2025, 11, 5),
        date(2025, 11, 12), date(2025, 12, 25),
        date(2026, 1, 26), date(2026, 3, 23), date(2026, 4, 3), date(2026, 4, 10),
        date(2026, 4, 14), date(2026, 8, 15), date(2026, 10, 2), date(2026, 11, 12),
        date(2026, 12, 25),
    }

    def __init__(self, calendar_path: Path):
        self.calendar_path = calendar_path
        self.holidays: set[date] = set(self.FALLBACK_HOLIDAYS)
        self.special_sessions: dict[date, MarketSession] = {}
        self.source = "fallback"
        self.load()

    def load(self) -> None:
        if not self.calendar_path.exists():
            return
        try:
            payload = json.loads(self.calendar_path.read_text(encoding="utf-8"))
            holidays = {
                date.fromisoformat(str(item["date"] if isinstance(item, dict) else item))
                for item in payload.get("holidays", [])
            }
            sessions: dict[date, MarketSession] = {}
            for item in payload.get("special_sessions", []):
                d = date.fromisoformat(item["date"])
                sessions[d] = MarketSession(
                    date=d,
                    open_time=time.fromisoformat(item.get("open", "09:15")),
                    close_time=time.fromisoformat(item.get("close", "15:30")),
                    name=item.get("name", "special"),
                )
            if holidays:
                self.holidays = holidays
            self.special_sessions = sessions
            self.source = payload.get("source", "file")
        except Exception:
            self.source = "fallback"

    def is_trading_day(self, check_date: Optional[date] = None) -> tuple[bool, str]:
        d = check_date or date.today()
        if d in self.special_sessions:
            session = self.special_sessions[d]
            return True, f"{session.name} session {session.open_time}-{session.close_time}"
        if d.weekday() >= 5:
            return False, f"{d.strftime('%A')} - NSE closed on weekends"
        if d in self.holidays:
            return False, f"{d.isoformat()} is an NSE public holiday"
        return True, f"calendar source: {self.source}"

    def is_open_now(self, now: Optional[datetime] = None) -> tuple[bool, str]:
        now = now or datetime.now()
        open_day, reason = self.is_trading_day(now.date())
        if not open_day:
            return False, reason
        session = self.special_sessions.get(
            now.date(),
            MarketSession(now.date(), time(9, 15), time(15, 30)),
        )
        if session.open_time <= now.time() <= session.close_time:
            return True, f"NSE open ({session.name})"
        return False, f"NSE closed outside {session.open_time}-{session.close_time}"
