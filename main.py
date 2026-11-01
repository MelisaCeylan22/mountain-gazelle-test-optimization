from __future__ import annotations

import csv
import os
import time

import matplotlib

matplotlib.use("Agg")  # Sunucu / VS Code ortaminda engelleyici pencere acma.
import matplotlib.pyplot as plt
import numpy as np

from bmgo import binary_mountain_gazelle_optimizer
from baselines import genetic_algorithm_optimizer, greedy_regression_selector
from data_generator import generate_telecom_data
from fitness import get_suite_metrics, prioritize_tests_and_calculate_apfd


SIZES = (50, 100, 250, 500, 1000)
OPTIMIZER_SEEDS = (42, 43, 44)
DATA_SEED = 42  # Adil karsilastirma icin her boyutta sabit veri kaynagi.
POP_SIZE = 12
MAX_ITER = 20
MIN_COVERAGE = 0.90
OUTPUT_DIR = "sonuclar"
GA_GENERATIONS = MAX_ITER * 4  # BMGO ile esit fitness degerlendirme sayisi.


def early_critical_coverage(order, coverage, critical, fraction=0.25):
    """Ilk %25 testte gorulen kritik kurallar. APFD *degildir*."""
    if len(order) == 0 or len(critical) == 0:
        return 0.0
    first_count = max(1, int(np.ceil(len(order) * fraction)))
    hits = np.any(coverage[order[:first_count]][:, critical] != 0, axis=0)
    return float(100.0 * np.mean(hits))


def write_case_table(path, cases):
    """Senaryolari Excel'de okunabilecek UTF-8 CSV olarak kaydeder."""
    if not cases:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=cases[0].keys(), delimiter=";")
        writer.writeheader()
        for case in cases:
            writer.writerow({
                key: " | ".join(map(str, value)) if isinstance(value, (list, tuple)) else value
                for key, value in case.items()
            })


def make_comparison_row(algorithm, n, seed, solution, fitness_value, elapsed,
                        coverage, durations, dependencies, critical, evaluations):
    """Bütün yöntemleri aynı metriklerle değerlendirir ve denetler."""
    m = get_suite_metrics(solution, coverage, durations, dependencies, critical)
    if (m["critical_missing"] != 0 or m["broken_dependencies"] != 0
            or m["coverage_ratio"] + 1e-9 < MIN_COVERAGE):
        raise RuntimeError(f"{algorithm} gecersiz cozum: {n} test, seed={seed}")
    return {
        "algoritma": algorithm,
        "test_sayisi": n,
        "seed": seed,
        "gereksinim_sayisi": m["total_features"],
        "secili_test": m["selected_count"],
        "sure_tasarrufu_yuzde": round(100 * (1 - m["time_ratio"]), 2),
        "genel_kapsam_yuzde": round(100 * m["coverage_ratio"], 2),
        "kritik_kapsam_yuzde": round(100 * (1 - m["critical_missing"] / m["critical_total"]), 2),
        "kopuk_on_kosul": m["broken_dependencies"],
        "fitness": round(float(fitness_value), 6),
        "algoritma_suresi_sn": round(elapsed, 4),
        "fitness_degerlendirmesi": evaluations,
    }


def run_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    comparison_rows = []
    representative_curves = {}
    curves_1000 = {}

    print("\nBMGO - TELEKOM REGRESYON TEST SECIMI VE ONCELIKLENDIRME")
    print("Kural: kritik %100, genel en az %90; on kosullar korunur.")
    print("NOT: Veriler kurgusaldir; hata tespit matrisi yoksa APFD hesaplanmaz.\n")
    print(f"BMGO ve GA es aday butcesi: {POP_SIZE * (1 + 4 * MAX_ITER)} fitness hesaplamasi.\n")

    for n in SIZES:
        coverage, durations, names, features, deps, metadata = generate_telecom_data(
            num_tests=n, seed=DATA_SEED, include_metadata=True
        )
        critical = metadata["critical_feature_indices"]
        if n == 50:
            write_case_table(
                os.path.join(OUTPUT_DIR, "telekom_50_ornek_senaryo.csv"),
                metadata["test_cases"],
            )

        for run_seed in OPTIMIZER_SEEDS:
            start = time.perf_counter()
            solution, fitness_value, convergence = binary_mountain_gazelle_optimizer(
                coverage_matrix=coverage,
                durations=durations,
                dependency_matrix=deps,
                critical_feature_indices=critical,
                min_coverage=MIN_COVERAGE,
                pop_size=POP_SIZE,
                max_iter=MAX_ITER,
                w_time=0.4,
                w_count=0.4,
                w_redundancy=0.2,
                random_seed=run_seed,
            )
            elapsed = time.perf_counter() - start
            metrics = get_suite_metrics(solution, coverage, durations, deps, critical)

            if (metrics["critical_missing"] != 0
                    or metrics["coverage_ratio"] < MIN_COVERAGE - 1e-9
                    or metrics["broken_dependencies"] != 0):
                raise RuntimeError(f"Gecersiz BMGO cozumu: {n} test, seed={run_seed}")

            order, apfd = prioritize_tests_and_calculate_apfd(
                solution, coverage, durations,
                dependency_matrix=deps,
                critical_feature_indices=critical,
            )
            early_coverage = early_critical_coverage(order, coverage, critical)

            result = {
                "test_sayisi": n,
                "seed": run_seed,
                "gereksinim_sayisi": len(features),
                "kritik_gereksinim_sayisi": len(critical),
                "yeni_is_kurali_sayisi": metadata["additional_rule_count"],
                "baglamsal_kapsam_sayisi": metadata["context_rule_count"],
                "secili_test": metrics["selected_count"],
                "elenen_test": n - metrics["selected_count"],
                "toplam_sure_sn": round(metrics["total_duration"], 2),
                "optimize_sure_sn": round(metrics["selected_duration"], 2),
                "sure_tasarrufu_yuzde": round(100 * (1 - metrics["time_ratio"]), 2),
                "genel_kapsam_yuzde": round(100 * metrics["coverage_ratio"], 2),
                "kritik_kapsam_yuzde": round(
                    100 * (1 - metrics["critical_missing"] / metrics["critical_total"]), 2
                ),
                "kopuk_on_kosul": metrics["broken_dependencies"],
                "gereksiz_kapsama_tekrari": metrics["redundant_hits"],
                "ilk_ceyrek_kritik_kural_kapsami_yuzde": round(early_coverage, 2),
                "APFD": "VERI_YOK" if apfd is None else round(apfd, 2),
                "fitness": round(fitness_value, 6),
                "algoritma_suresi_sn": round(elapsed, 4),
            }
            rows.append(result)
            comparison_rows.append(make_comparison_row(
                "BMGO", n, run_seed, solution, fitness_value, elapsed,
                coverage, durations, deps, critical,
                POP_SIZE * (1 + 4 * MAX_ITER),
            ))

            print(
                f"{n:4} test | {len(features):3} kapsam hedefi | seed={run_seed} "
                f"| secilen={metrics['selected_count']:3} "
                f"| kapsam=%{100 * metrics['coverage_ratio']:.1f} "
                f"| kritik=%100 | tasarruf=%{100 * (1 - metrics['time_ratio']):.1f} "
                f"| ihlal={metrics['broken_dependencies']} | {elapsed:.2f} sn"
            )

            if run_seed == OPTIMIZER_SEEDS[0]:
                representative_curves[n] = convergence
                if n == 1000:
                    curves_1000["BMGO"] = convergence
                write_case_table(
                    os.path.join(OUTPUT_DIR, f"telekom_{n}_secili_testler.csv"),
                    [
                        {"oncelik_sirasi": rank, **metadata["test_cases"][int(idx)]}
                        for rank, idx in enumerate(order, 1)
                    ],
                )
                print("  Ilk 5 oncelikli test:")
                for test_idx in order[:5]:
                    print(f"    T_{test_idx + 1:04d} - {names[test_idx]}")

        # Greedy deterministiktir; bir kez calistirmak yeterli.
        start = time.perf_counter()
        greedy_sol, greedy_fit = greedy_regression_selector(
            coverage, durations, deps, critical_feature_indices=critical,
            min_coverage=MIN_COVERAGE,
        )
        greedy_elapsed = time.perf_counter() - start
        comparison_rows.append(make_comparison_row(
            "Greedy", n, "tek", greedy_sol, greedy_fit, greedy_elapsed,
            coverage, durations, deps, critical, "tek-gecis",
        ))

        # GA da BMGO ile esit (P + 4 * iter * P) aday degerlendirir.
        for run_seed in OPTIMIZER_SEEDS:
            start = time.perf_counter()
            ga_sol, ga_fit, ga_curve = genetic_algorithm_optimizer(
                coverage_matrix=coverage,
                durations=durations,
                dependency_matrix=deps,
                pop_size=POP_SIZE,
                max_generations=GA_GENERATIONS,
                random_seed=run_seed,
                critical_feature_indices=critical,
                min_coverage=MIN_COVERAGE,
            )
            ga_elapsed = time.perf_counter() - start
            ga_result = make_comparison_row(
                "GA", n, run_seed, ga_sol, ga_fit, ga_elapsed,
                coverage, durations, deps, critical,
                POP_SIZE * (1 + GA_GENERATIONS),
            )
            comparison_rows.append(ga_result)
            if n == 1000 and run_seed == OPTIMIZER_SEEDS[0]:
                curves_1000["GA"] = ga_curve
            print(
                f"  GA  | {n:4} test | seed={run_seed} "
                f"| secilen={ga_result['secili_test']:3} "
                f"| tasarruf=%{ga_result['sure_tasarrufu_yuzde']:.1f} "
                f"| fitness={ga_fit:.4f} | {ga_elapsed:.2f} sn"
            )
        print(f"  Greedy | {n:4} test | fitness={greedy_fit:.4f} | {greedy_elapsed:.2f} sn\n")

    csv_path = os.path.join(OUTPUT_DIR, "telekom_bmgo_denemeler.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    for n, curve in representative_curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, label=f"{n} test")
    ax.set_title("BMGO Yakinsama Grafigi - Telekom Testleri")
    ax.set_xlabel("Iterasyon")
    ax.set_ylabel("Fitness (dusuk daha iyi)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bmgo_yakinsama.png"), dpi=200)
    plt.close(fig)

    grouped = {
        n: [r["sure_tasarrufu_yuzde"] for r in rows if r["test_sayisi"] == n]
        for n in SIZES
    }
    means = [np.mean(grouped[n]) for n in SIZES]
    stds = [np.std(grouped[n], ddof=1) for n in SIZES]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(SIZES, means, yerr=stds, marker="o", capsize=4)
    ax.set_title("Test Paketi Boyutuna Gore Sure Tasarrufu (3 Kosum)")
    ax.set_xlabel("Baslangictaki test sayisi")
    ax.set_ylabel("Ortalama sure tasarrufu (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bmgo_sure_tasarrufu.png"), dpi=200)
    plt.close(fig)

    print(f"\nCSV: {csv_path}")
    print(f"Senaryo tablolari: {OUTPUT_DIR}/telekom_50_ornek_senaryo.csv ve secili testler")
    print(f"Grafikler: {OUTPUT_DIR}/bmgo_yakinsama.png, {OUTPUT_DIR}/bmgo_sure_tasarrufu.png")

    comparison_path = os.path.join(OUTPUT_DIR, "algoritma_karsilastirma.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(comparison_rows)

    algorithm_names = ("Greedy", "GA", "BMGO")
    x = np.arange(len(SIZES))
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, name in enumerate(algorithm_names):
        means = [
            np.mean([r["sure_tasarrufu_yuzde"] for r in comparison_rows
                     if r["algoritma"] == name and r["test_sayisi"] == n])
            for n in SIZES
        ]
        ax.bar(x + (idx - 1) * 0.25, means, width=0.25, label=name)
    ax.set_xticks(x, [str(n) for n in SIZES])
    ax.set_xlabel("Test sayisi")
    ax.set_ylabel("Ortalama sure tasarrufu (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Greedy - GA - BMGO Karsilastirmasi")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "algoritma_karsilastirma.png"), dpi=200)
    plt.close(fig)

    # Her veri boyutunda ayni fitness tanimi vardir, ancak boyutlar arasi
    # mutlak fitness degerlerini basari siralamasi gibi okumamak gerekir.
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, name in enumerate(algorithm_names):
        means = [
            np.mean([r["fitness"] for r in comparison_rows
                     if r["algoritma"] == name and r["test_sayisi"] == n])
            for n in SIZES
        ]
        ax.bar(x + (idx - 1) * 0.25, means, width=0.25, label=name)
    ax.set_xticks(x, [str(n) for n in SIZES])
    ax.set_xlabel("Test sayisi")
    ax.set_ylabel("Ortalama fitness (dusuk daha iyi)")
    ax.set_title("Ayni Veri Boyutunda Fitness Karsilastirmasi")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "algoritma_fitness_karsilastirma.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in ("BMGO", "GA"):
        curve = curves_1000[name]
        evaluations = POP_SIZE + np.arange(1, len(curve) + 1) * (
            POP_SIZE * 4 if name == "BMGO" else POP_SIZE
        )
        ax.plot(evaluations, curve, label=name)
    ax.set_xlabel("Degerlendirilen aday sayisi")
    ax.set_ylabel("En iyi fitness (dusuk daha iyi)")
    ax.set_title("1000 Testte Es Aday Butcesiyle Yakinsama")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bmgo_ga_es_butce_yakinsama.png"), dpi=200)
    plt.close(fig)

    print(f"\nKarsilastirma CSV: {comparison_path}")
    print("Karsilastirma grafikleri: algoritma_karsilastirma.png, "
          "algoritma_fitness_karsilastirma.png, bmgo_ga_es_butce_yakinsama.png")


if __name__ == "__main__":
    run_experiment()
