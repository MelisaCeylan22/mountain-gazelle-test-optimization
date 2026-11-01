from __future__ import annotations

import numpy as np


def _selected_mask(solution: np.ndarray, num_tests: int) -> np.ndarray:
    selected = np.asarray(solution)
    if selected.shape != (num_tests,):
        raise ValueError(f"solution boyutu ({num_tests},) olmali")
    if not np.all((selected == 0) | (selected == 1)):
        raise ValueError("solution yalnizca 0 ve 1 icermeli")
    return selected.astype(bool, copy=False)


def _dependency_edges(dependency_matrix, num_tests: int):
    if dependency_matrix is None:
        return np.array([], dtype=int), np.array([], dtype=int)
    deps = np.asarray(dependency_matrix)
    if deps.shape != (num_tests, num_tests):
        raise ValueError("dependency_matrix NxN boyutunda olmali")
    return np.nonzero(deps)


def get_suite_metrics(
    solution,
    coverage_matrix,
    durations,
    dependency_matrix=None,
    critical_feature_indices=None,
    dependency_edges=None,
):
    """Seçimin temel sonuçlarını ve kısıt ihlallerini tek yerde hesaplar."""
    coverage = np.asarray(coverage_matrix)
    times = np.asarray(durations, dtype=float)
    if coverage.ndim != 2:
        raise ValueError("coverage_matrix iki boyutlu olmali")
    num_tests, num_features = coverage.shape
    if num_tests == 0 or num_features == 0:
        raise ValueError("En az bir test ve bir gereksinim olmali")
    if times.shape != (num_tests,) or np.any(times <= 0):
        raise ValueError("Her test icin pozitif sure verilmeli")

    selected = _selected_mask(solution, num_tests)
    # int8 matrisle np.dot kullanmak 128+ testte tamsayi tasmasina yol acabilir.
    hits = np.sum(coverage[selected] != 0, axis=0)
    covered = hits > 0

    if critical_feature_indices is None:
        critical = np.arange(num_features)
    else:
        critical = np.asarray(critical_feature_indices, dtype=int)
        if critical.ndim != 1 or np.any(critical < 0) or np.any(critical >= num_features):
            raise ValueError("critical_feature_indices gecerli sutun indeksleri olmali")

    if dependency_edges is None:
        parents, children = _dependency_edges(dependency_matrix, num_tests)
    else:
        parents, children = dependency_edges
        parents = np.asarray(parents, dtype=int)
        children = np.asarray(children, dtype=int)
        if parents.shape != children.shape:
            raise ValueError("dependency_edges dizilerinin boyutu esit olmali")

    broken = int(np.count_nonzero(selected[children] & ~selected[parents]))
    total_hits = int(np.sum(hits))
    redundant_hits = int(np.sum(np.maximum(hits - 1, 0)))

    return {
        "selected_count": int(np.count_nonzero(selected)),
        "total_tests": num_tests,
        "covered_count": int(np.count_nonzero(covered)),
        "total_features": num_features,
        "coverage_ratio": float(np.mean(covered)),
        "critical_missing": int(np.count_nonzero(~covered[critical])),
        "critical_total": int(critical.size),
        "broken_dependencies": broken,
        "selected_duration": float(np.sum(times[selected])),
        "total_duration": float(np.sum(times)),
        "time_ratio": float(np.sum(times[selected]) / np.sum(times)),
        "count_ratio": float(np.mean(selected)),
        "redundant_hits": redundant_hits,
        # Önceki sürümde tüm tam-kapsamlı çözümler için oran 1 olabiliyordu.
        "redundancy_ratio": redundant_hits / max(1, total_hits),
    }


def calculate_fitness(
    solution,
    coverage_matrix,
    durations,
    dependency_matrix=None,
    w_time=0.4,
    w_redundancy=0.2,
    critical_feature_indices=None,
    min_coverage=1.0,
    w_count=0.4,
    dependency_edges=None,
):
    """Küçültülecek fitness; düşük değer daha iyi.

    Eski 100% kapsamlı çağrılar için ``min_coverage=1`` varsayılandır.
    Yeni telekom deneyinde ``min_coverage=.90`` ve metadata'daki kritik
    gereksinim indeksleri mutlaka birlikte verilmelidir.

    Kısıt ihlal eden çözüm >1000, uygun çözüm <=1 aralığındadır.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage (0,1] araliginda olmali")
    weights = np.array([w_time, w_count, w_redundancy], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Agirliklar negatif olmayan sayilar olmali")

    m = get_suite_metrics(
        solution, coverage_matrix, durations, dependency_matrix,
        critical_feature_indices, dependency_edges,
    )
    required_count = int(np.ceil(min_coverage * m["total_features"] - 1e-12))
    missing_coverage = max(0, required_count - m["covered_count"])
    violations = m["critical_missing"] + missing_coverage + m["broken_dependencies"]

    ratios = np.array([m["time_ratio"], m["count_ratio"], m["redundancy_ratio"]])
    objective = float(np.dot(weights, ratios) / weights.sum())
    if m["selected_count"] == 0:
        violations += 1
    if violations:
        return 1000.0 + 100.0 * violations + objective
    return objective


def prioritize_tests_and_calculate_apfd(
    solution,
    coverage_matrix,
    durations,
    dependency_matrix=None,
    fault_detection_matrix=None,
    critical_feature_indices=None,
):
    """Seçili testleri ön koşulları bozmadan sıralar.

    Hata matrisi verilmezse APFD ``None`` döner. Matris verilirse her sütun
    ayrı bir hatayı, her satır o hatayı tespit eden testi temsil eder.
    Sıra oluşturulurken bu hata matrisi *kullanılmaz*: değerlendirmeye sızıntı
    olmaması için yalnızca gereksinim önceliği ve süreye bakılır.
    """
    coverage = np.asarray(coverage_matrix)
    times = np.asarray(durations, dtype=float)
    num_tests, num_features = coverage.shape
    selected = _selected_mask(solution, num_tests)
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        return np.array([], dtype=int), None

    parent_lists = [np.array([], dtype=int) for _ in range(num_tests)]
    if dependency_matrix is not None:
        parents, children = _dependency_edges(dependency_matrix, num_tests)
        for idx in indices:
            parent_lists[idx] = parents[children == idx]
            if np.any(~selected[parent_lists[idx]]):
                raise ValueError("Siralama icin once bagimlilik ihlallerini giderin")

    importance = np.ones(num_features, dtype=float)
    if critical_feature_indices is not None:
        importance[np.asarray(critical_feature_indices, dtype=int)] = 3.0

    ordered = []
    used = set()
    uncovered = np.ones(num_features, dtype=bool)

    while len(ordered) < len(indices):
        ready = [
            int(i) for i in indices
            if int(i) not in used and all(int(p) in used for p in parent_lists[i])
        ]
        if not ready:
            raise ValueError("Bagimlilik grafiğinde döngü var")

        def score(idx):
            new = (coverage[idx] != 0) & uncovered
            return float(np.sum(importance[new]) / max(times[idx], 1e-9))

        next_idx = max(ready, key=lambda i: (score(i), -times[i], -i))
        ordered.append(next_idx)
        used.add(next_idx)
        uncovered &= coverage[next_idx] == 0

    prioritized = np.asarray(ordered, dtype=int)
    if fault_detection_matrix is None:
        return prioritized, None

    faults = np.asarray(fault_detection_matrix)
    if faults.ndim != 2 or faults.shape[0] != num_tests or faults.shape[1] == 0:
        raise ValueError("fault_detection_matrix: test sayisi x hata sayisi olmali")

    detections = faults[prioritized] != 0
    if not np.all(np.any(detections, axis=0)):
        # Klasik APFD bütün referans hataların bulunduğu durumu varsayar.
        return prioritized, None
    first_positions = np.argmax(detections, axis=0) + 1
    n, m = detections.shape
    apfd = 1.0 - np.sum(first_positions) / (n * m) + 1.0 / (2 * n)
    return prioritized, float(100 * apfd)