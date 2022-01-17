import time

timing = dict()


def timeit(name: str, starting: bool = False):
    if starting:
        if name not in timing:
            timing[name] = [time.monotonic(), 0.0]
        else:
            timing[name][0] = time.monotonic()
    else:
        timing[name][1] += time.monotonic() - timing[name][0]
