from __future__ import annotations

import numpy as np

from bmgo import _repair_and_prune
from fitness import calculate_fitness


def _prepare_problem(coverage_matrix, durations, dependency_matrix,
                     critical_feature_indices, min_coverage):
    coverage = (np.asarray(coverage_matrix) != 0).astype(np.int32)
    times = np.asarray(durations, dtype=float)
    if coverage.ndim != 2:
        raise ValueError("coverage_matrix iki boyutlu olmali")
    n, f = coverage.shape
    if n == 0 or f == 0 or times.shape != (n,) or np.any(times <= 0):
        raise ValueError("Gecersiz test sayisi, gereksinim sayisi veya sureler")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage (0, 1] araliginda olmali")

    critical = (
        np.arange(f, dtype=int) if critical_feature_indices is None
        else np.asarray(critical_feature_indices, dtype=int)
    )
    if critical.ndim != 1 or np.any(critical < 0) or np.any(critical >= f):
        raise ValueError("Kritik gereksinim indeksleri gecersiz")

    dep = np.zeros((n, n), dtype=bool) if dependency_matrix is None else (
        np.asarray(dependency_matrix) != 0
    )
    if dep.shape != (n, n) or np.any(np.tril(dep)):
        raise ValueError("dependency_matrix sirali / dongusuz NxN olmali")

    parents, children = np.nonzero(dep)
    parents_of = [np.flatnonzero(dep[:, i]) for i in range(n)]
    children_of = [np.flatnonzero(dep[i]) for i in range(n)]
    depth = np.zeros(n, dtype=int)
    for i in range(n):
        if len(parents_of[i]):
            depth[i] = 1 + int(np.max(depth[parents_of[i]]))

    return (
        coverage, times, critical, int(np.ceil(min_coverage * f - 1e-12)),
        (parents, children), parents_of, children_of, depth,
    )


def greedy_regression_selector(
    coverage_matrix, durations, dependency_matrix=None,
    critical_feature_indices=None, min_coverage=1.0,
    w_time=0.4, w_count=0.4, w_redundancy=0.2,
):
    """Sıfır seçimden başlayan deterministik Greedy + onarım / budama."""
    (coverage, times, critical, required, edges, parents_of, children_of, depth) = (
        _prepare_problem(coverage_matrix, durations, dependency_matrix,
                         critical_feature_indices, min_coverage)
    )
    solution = _repair_and_prune(
        np.zeros(len(times), dtype=np.int8), coverage, times, critical,
        required, parents_of, children_of, depth,
    )
    score = calculate_fitness(
        solution, coverage, times,
        critical_feature_indices=critical, min_coverage=min_coverage,
        w_time=w_time, w_count=w_count, w_redundancy=w_redundancy,
        dependency_edges=edges,
    )
    return solution, score


def genetic_algorithm_optimizer(
    coverage_matrix, durations, dependency_matrix=None,
    pop_size=12, max_generations=80, random_seed=42,
    critical_feature_indices=None, min_coverage=1.0,
    w_time=0.4, w_count=0.4, w_redundancy=0.2,
    crossover_rate=0.8, mutation_rate=None,
):
    """Seçim, çaprazlama, mutasyon ve elitizm kullanan ikili GA.

    Her yeni aday BMGO ile *aynı* `_repair_and_prune` işleminden geçer.
    Dolayısıyla fark, GA ile BMGO'nun arama hareketlerinden kaynaklanır.
    """
    if pop_size < 2 or max_generations < 1:
        raise ValueError("pop_size >= 2, max_generations >= 1 olmali")
    if not 0 <= crossover_rate <= 1:
        raise ValueError("crossover_rate [0,1] araliginda olmali")

    (coverage, times, critical, required, edges, parents_of, children_of, depth) = (
        _prepare_problem(coverage_matrix, durations, dependency_matrix,
                         critical_feature_indices, min_coverage)
    )
    n = len(times)
    if mutation_rate is None:
        mutation_rate = min(0.10, max(1.5 / n, 0.01))
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate [0,1] araliginda olmali")

    rng = np.random.default_rng(random_seed)

    def repair(bits):
        return _repair_and_prune(bits, coverage, times, critical, required,
                                 parents_of, children_of, depth)

    def fitness(bits):
        return calculate_fitness(
            bits, coverage, times,
            critical_feature_indices=critical, min_coverage=min_coverage,
            w_time=w_time, w_count=w_count, w_redundancy=w_redundancy,
            dependency_edges=edges,
        )

    population = np.asarray(
        [repair(rng.integers(0, 2, size=n, dtype=np.int8))
         for _ in range(pop_size)], dtype=np.int8
    )
    scores = np.asarray([fitness(bits) for bits in population])
    best_idx = int(np.argmin(scores))
    best_solution = population[best_idx].copy()
    best_score = float(scores[best_idx])
    convergence = np.empty(max_generations, dtype=float)

    def tournament():
        candidates = rng.integers(0, pop_size, size=min(3, pop_size))
        return population[candidates[np.argmin(scores[candidates])]]

    for generation in range(max_generations):
        next_population = []
        # Bir elit korunur, kalan P-1 çocuk + elit tekrar değerlendirilir.
        elite = population[int(np.argmin(scores))].copy()
        next_population.append(elite)

        for _ in range(pop_size - 1):
            parent_a, parent_b = tournament(), tournament()
            if rng.random() < crossover_rate:
                choice = rng.random(n) < 0.5
                child = np.where(choice, parent_a, parent_b).astype(np.int8)
            else:
                child = parent_a.copy()
            mutation = rng.random(n) < mutation_rate
            child[mutation] ^= 1
            next_population.append(repair(child))

        population = np.asarray(next_population, dtype=np.int8)
        scores = np.asarray([fitness(bits) for bits in population])
        best_idx = int(np.argmin(scores))
        if scores[best_idx] < best_score:
            best_score = float(scores[best_idx])
            best_solution = population[best_idx].copy()
        convergence[generation] = best_score

    return best_solution, best_score, convergence