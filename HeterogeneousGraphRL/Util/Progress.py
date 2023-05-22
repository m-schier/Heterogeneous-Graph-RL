class NoopItWrapper:
    def __init__(self, *args, iterable=None, **kwargs):
        if iterable is not None:
            self.inner = iterable
        elif len(args) > 0:
            self.inner = args[0]
        else:
            self.inner = None

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        if self.inner is None:
            raise ValueError("No iterable set")
        return self.inner


def maybe_tqdm(*args, **kwargs):
    import sys
    import os
    from tqdm import tqdm

    if os.isatty(sys.stdin.fileno()):
        return tqdm(*args, **kwargs)
    else:
        return NoopItWrapper(*args, **kwargs)
