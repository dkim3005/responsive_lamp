import math


def head_wiggle(elapsed: float) -> list[float]:
    return [0.0, 0.0, 0.0, math.sin(2.0 * math.pi * elapsed) * 8.0, 0.0, 0.0]


def base_scan(elapsed: float) -> list[float]:
    return [math.sin(math.pi * elapsed) * 15.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def add_joints(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]

