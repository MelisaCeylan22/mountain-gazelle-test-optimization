from __future__ import annotations

import numpy as np

from fitness import calculate_fitness


def v_shaped_transfer(x):
    """V(x) = |tanh(x)|: ikili bit değişim olasılığı."""
    return np.abs(np.tanh(x))


def _repair_and_prune(
    bits, coverage, durations, critical, required_count, parents_of,
    children_of, depths,
):
    """Ön koşul ekle, kapsanmayan gereksinimi tamamla, gereksiz testleri at."""
    selected = np.asarray(bits, dtype=bool).copy()
    n, f = coverage.shape
    critical_mask = np.zeros(f, dtype=bool)
    critical_mask[critical] = True

    # Veri üreticimiz ön koşul testini daima önce konumlandırır (DAG).
    for idx in range(n - 1, -1, -1):
        if selected[idx] and len(parents_of[idx]):
            selected[parents_of[idx]] = True

    hits = np.sum(coverage[selected], axis=0).astype(int)

    def add_with_parents(idx):
        stack = [int(idx)]
        while stack:
            current = stack.pop()
            if selected[current]:
                continue
            selected[current] = True
            hits[:] += coverage[current]
            stack.extend(int(p) for p in parents_of[current])

    # Kritik gereksinimler ve ardından asgari genel kapsam doldurulur.
    for _ in range(n + 1):
        uncovered = hits == 0
        critical_missing = uncovered & critical_mask
        if not critical_missing.any() and np.count_nonzero(~uncovered) >= required_count:
            break

        weights = uncovered.astype(float)
        if critical_missing.any():
            weights[critical_mask] = 10.0 * critical_missing[critical_mask]
        gains = coverage @ weights
        gains[selected] = 0.0
        scores = gains / durations
        idx = int(np.argmax(scores))
        if scores[idx] <= 0:
            raise ValueError("Veri kumesinde gerekli kapsami saglayacak test yok")
        add_with_parents(idx)
    else:
        raise ValueError("Kapsam onarimi tamamlanamadi")

    # Önce bağımlı testleri, aynı derinlikte pahalı testleri çıkar.
    # Kısıtları her silmeden sonra tekrar denetleriz.
    order = sorted(np.flatnonzero(selected), key=lambda i: (depths[i], durations[i]), reverse=True)
    covered_count = int(np.count_nonzero(hits))
    for idx in order:
        if not selected[idx] or any(selected[child] for child in children_of[idx]):
            continue
        lost = (hits == 1) & (coverage[idx] != 0)
        if np.any(lost & critical_mask):
            continue
        if covered_count - int(np.count_nonzero(lost)) < required_count:
            continue
        selected[idx] = False
        hits -= coverage[idx]
        covered_count -= int(np.count_nonzero(lost))

    return selected.astype(np.int8)


def binary_mountain_gazelle_optimizer(
    coverage_matrix,
    durations,
    dependency_matrix=None,
    pop_size=20,
    max_iter=50,
    w_time=0.4,
    w_redundancy=0.2,
    random_seed=42,
    critical_feature_indices=None,
    min_coverage=1.0,
    w_count=0.4,
):
    """En iyi ikili seçimi, fitness'ı ve iterasyon eğrisini döndürür."""
    coverage = (np.asarray(coverage_matrix) != 0).astype(np.int32)
    duration = np.asarray(durations, dtype=float)
    if coverage.ndim != 2:
        raise ValueError("coverage_matrix iki boyutlu olmali")
    n, f = coverage.shape
    if n < 1 or f < 1 or duration.shape != (n,) or np.any(duration <= 0):
        raise ValueError("Test / gereksinim boyutlari veya sureler gecersiz")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage (0,1] araliginda olmali")
    if pop_size < 2 or max_iter < 1:
        raise ValueError("pop_size >= 2 ve max_iter >= 1 olmali")

    critical = (
        np.arange(f, dtype=int)
        if critical_feature_indices is None
        else np.asarray(critical_feature_indices, dtype=int)
    )
    if np.any(critical < 0) or np.any(critical >= f):
        raise ValueError("Kritik gereksinim indeksi gecersiz")
    required_count = int(np.ceil(min_coverage * f - 1e-12))
    if np.count_nonzero(np.any(coverage, axis=0)) < required_count:
        raise ValueError("Veri kumesinde minimum kapsam olanaksiz")
    if not np.all(np.any(coverage[:, critical], axis=0)):
        raise ValueError("En az bir kritik gereksinimi kapsayan test yok")

    if dependency_matrix is None:
        dep = np.zeros((n, n), dtype=bool)
    else:
        dep = np.asarray(dependency_matrix) != 0
        if dep.shape != (n, n):
            raise ValueError("dependency_matrix NxN olmali")
    # Şu anki senaryolarda ebeveyn indeksleri her zaman çocuktan küçüktür.
    if np.any(np.tril(dep)):
        raise ValueError("Bagimlilik matrisi sirali ve dongusuz olmali (parent < child)")

    parent_ids, child_ids = np.nonzero(dep)
    parents_of = [np.flatnonzero(dep[:, i]) for i in range(n)]
    children_of = [np.flatnonzero(dep[i]) for i in range(n)]
    depths = np.zeros(n, dtype=int)
    for i in range(n):
        if len(parents_of[i]):
            depths[i] = 1 + int(np.max(depths[parents_of[i]]))

    rng = np.random.default_rng(random_seed)
    continuous_pop = rng.uniform(-2.0, 2.0, (pop_size, n))
    binary_pop = np.empty((pop_size, n), dtype=np.int8)
    scores = np.empty(pop_size, dtype=float)
    for i in range(pop_size):
        initial = (rng.random(n) < 0.5).astype(np.int8)
        binary_pop[i] = _repair_and_prune(
            initial, coverage, duration, critical, required_count,
            parents_of, children_of, depths,
        )
        scores[i] = calculate_fitness(
            binary_pop[i], coverage, duration,
            w_time=w_time, w_redundancy=w_redundancy, w_count=w_count,
            min_coverage=min_coverage, critical_feature_indices=critical,
            dependency_edges=(parent_ids, child_ids),
        )

    best_idx = int(np.argmin(scores))
    best_solution = binary_pop[best_idx].copy()
    best_continuous = continuous_pop[best_idx].copy()
    best_fitness = float(scores[best_idx])
    curve = np.empty(max_iter, dtype=float)

    for iteration in range(max_iter):
        a = -1.0 - iteration / max_iter
        new_cont = []
        new_bin = []
        new_scores = []
        mean_pop = np.mean(continuous_pop, axis=0)

        for i in range(pop_size):
            current = continuous_pop[i]
            random_index = int(rng.integers(pop_size))
            random_position = continuous_pop[random_index]
            bh = random_position + mean_pop * rng.random()
            f_vector = rng.standard_normal(n) * np.exp(2.0 - 2.0 * iteration / max_iter)
            coeffs = [(a + 1.0) + rng.random() * a * rng.standard_normal(n) for _ in range(3)]
            c1, c2, c3 = coeffs
            r1, r2, r3, r4 = rng.integers(1, 3, size=4)

            tsm = best_continuous - np.abs((r1 * bh - r2 * current) * f_vector) * c1
            mh = (bh + c2) + (r3 * best_continuous - r4 * random_position) * c3
            d = (np.abs(current) + np.abs(best_continuous)) * (2.0 * rng.random() - 1.0)
            bmh = (current - d) + (r1 * best_continuous - r2 * bh) * c1
            msf = rng.uniform(-2.0, 2.0, n)

            for candidate in (tsm, mh, bmh, msf):
                candidate = np.clip(candidate, -4.0, 4.0)
                # V-transfer değeri "yeni bitin 1 olma olasılığı" değildir.
                # Sık kullanılan V-şekilli binary uyarlamada bit çevrilir.
                flips = rng.random(n) < 0.4 * v_shaped_transfer(candidate - current)
                new_bits = np.logical_xor(binary_pop[i].astype(bool), flips)
                repaired = _repair_and_prune(
                    new_bits, coverage, duration, critical, required_count,
                    parents_of, children_of, depths,
                )
                score = calculate_fitness(
                    repaired, coverage, duration,
                    w_time=w_time, w_redundancy=w_redundancy, w_count=w_count,
                    min_coverage=min_coverage, critical_feature_indices=critical,
                    dependency_edges=(parent_ids, child_ids),
                )
                new_cont.append(candidate)
                new_bin.append(repaired)
                new_scores.append(score)

        all_cont = np.vstack((continuous_pop, np.asarray(new_cont)))
        all_bin = np.vstack((binary_pop, np.asarray(new_bin)))
        all_scores = np.concatenate((scores, np.asarray(new_scores)))
        survivors = np.argsort(all_scores, kind="stable")[:pop_size]
        continuous_pop = all_cont[survivors].copy()
        binary_pop = all_bin[survivors].copy()
        scores = all_scores[survivors].copy()

        if scores[0] < best_fitness:
            best_fitness = float(scores[0])
            best_solution = binary_pop[0].copy()
            best_continuous = continuous_pop[0].copy()
        curve[iteration] = best_fitness

    return best_solution, best_fitness, curve