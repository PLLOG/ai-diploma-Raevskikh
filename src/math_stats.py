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

def prob_event(count_A: int, n: int) -> float:
    """P(A) = count(A)/n"""
    if n <= 0:
        raise ValueError("prob_event: n must be > 0")
    if count_A < 0 or count_A > n:
        raise ValueError("prob_event: invalid count")
    return count_A / n

def prob_conditional(count_A_and_B: int, count_B: int) -> float:
    """P(A|B) = count(A∩B)/count(B)"""
    if count_B <= 0:
        raise ValueError("prob_conditional: count_B must be > 0")
    if count_A_and_B < 0 or count_A_and_B > count_B:
        raise ValueError("prob_conditional: invalid intersection count")
    return count_A_and_B / count_B

def is_independent_by_counts(p_a: float, p_a_given_b: float, tol: float = 0.05) -> bool:
    """Проверка независимости по приближению |P(A|B)-P(A)| <= tol"""
    return abs(p_a_given_b - p_a) <= tol

def bayes_posterior(prior: float, likelihood: float, evidence: float) -> float:
    for name, p in [("prior", prior), ("likelihood", likelihood), ("evidence", evidence)]:
        if p < 0 or p > 1:
            raise ValueError(f"bayes_posterior: {name} must be in [0,1]")
    if evidence == 0:
        raise ValueError("bayes_posterior: evidence must be > 0")
    return (likelihood * prior) / evidence

def prob_from_counts(count: int, n: int) -> float:
    if n <= 0:
        raise ValueError("prob_from_counts: n must be > 0")
    if count < 0 or count > n:
        raise ValueError("prob_from_counts: invalid count")
    return count / n

def ci_mean_normal_approx(values, z: float = 1.96):
    """Приближённый CI для среднего: mean ± z*SEM."""
    m = mean(values)
    se = sem(values)
    return (m - z * se, m + z * se)

def bootstrap_ci_mean(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    means = bootstrap_means(values, n_boot=n_boot, seed=seed)
    low = float(np.quantile(means, alpha/2))
    high = float(np.quantile(means, 1 - alpha/2))
    return low, high