def contingency_2x2(recs: list[dict], a_key: str, b_key: str) -> list[list[int]]:
    """2x2 таблица частот для бинарных признаков a_key и b_key (значения 0/1)."""
    table = [[0, 0], [0, 0]]
    for r in recs:
        a = int(r[a_key])
        b = int(r[b_key])
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("contingency_2x2: values must be 0/1")
        table[a][b] += 1
    return table