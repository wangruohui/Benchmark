import numpy as np


def stat(times: list):
    assert times, "No times provided"

    first = times[0]
    if len(times) == 1:
        return first, 0, 0, 0

    rest = times[1:]
    n = len(rest)

    X = np.c_[np.ones(n), np.arange(n)]
    y = np.array(rest)

    coef, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)

    if residuals.size == 0:
        r = 0
    else:
        r = np.sqrt(residuals[0] / n)

    return first, coef[0], coef[1], r


def test():
    times = [9]
    print(stat(times))
    times = [9, 1, 1, 1, 1]
    print(stat(times))
    times = [9, 1, 1.1, 1.2]
    print(stat(times))
    times = [9, 1, 1.1, 1.2, 1.3, 1.5]
    print(stat(times))


if __name__ == "__main__":
    test()
