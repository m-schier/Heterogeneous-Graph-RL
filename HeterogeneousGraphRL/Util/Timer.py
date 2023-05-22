import warnings
from time import time


_LAST_ACTIVE_INSTANCE = None


class Timer:
    def __init__(self):
        global _LAST_ACTIVE_INSTANCE

        self.durations: dict = {}
        _LAST_ACTIVE_INSTANCE = self

    def time(self, name):
        return _NamedTimerHandle(self, name)

    @staticmethod
    def get_current() -> 'Timer':
        if _LAST_ACTIVE_INSTANCE is None:
            warnings.warn("Timer.get_current() called with no active timer, creating an instance")
            Timer()
        return _LAST_ACTIVE_INSTANCE


class _NamedTimerHandle:
    def __init__(self, timer: Timer, name: str):
        self.timer = timer
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = self.timer.durations.get(self.name, .0)
        self.timer.durations[self.name] = t + time() - self.start
