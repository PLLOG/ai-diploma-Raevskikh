def mean(values: list[float]) -> float:
    #Среднее арифметическое. Требует непустой список.
    if len(values) == 0:
        raise ValueError("mean: empty list")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    #Медиана. Требует непустой список.
    if len(values) == 0:
        raise ValueError("median: empty list")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    else:
        return float(s[mid])


def variance_sample(values: list[float]) -> float:
    #Выборочная дисперсия (деление на n-1).
    n = len(values)
    if n < 2:
        raise ValueError("variance_sample: need at least 2 values")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)


def std_sample(values: list[float]) -> float:
    #Выборочное стандартное отклонение.
    return variance_sample(values) ** 0.5

def trimmed_mean(values: list[float], k: int = 1) -> float:
    #Усечённое среднее: убрать k минимальных и k максимальных.
    n = len(values)
    if n == 0:
        raise ValueError("trimmed_mean: empty list")
    if 2 * k >= n:
        raise ValueError("trimmed_mean: k too large")
    s = sorted(values)
    core = s[k:n - k]
    return mean(core)

def with_outlier(values: list[float], outlier: float) -> list[float]:
    #Вернуть новую выборку, добавив выброс (не меняем исходный список).
    return list(values) + [outlier]

def describe(values: list[float]) -> dict:
    #Короткое описание выборки (как мини-отчёт).
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "std": std_sample(values) if len(values) >= 2 else None,
    }