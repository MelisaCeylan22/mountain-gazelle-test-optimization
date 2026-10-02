import os
import time
import matplotlib.pyplot as plt
import numpy as np

from bmgo import binary_mountain_gazelle_optimizer
from data_generator import generate_synthetic_data, get_sample_ecommerce_data
from fitness import prioritize_tests_and_calculate_apfd


def print_suite_metrics(
    title,
    solution,
    coverage_matrix,
    durations,
    test_names,
    apfd_score,
    ordered_suite,
    dependency_matrix,
):
    """Terminal için detaylı, okunaklı ve tüm metrikleri içeren raporu basar."""
    selected_indices = np.where(solution == 1)[0]
    total_tests = len(durations)
    selected_count = len(selected_indices)

    initial_duration = int(np.sum(durations))
    optimized_duration = int(np.sum(durations[selected_indices]))
    saved_time = initial_duration - optimized_duration
    time_reduction_pct = (saved_time / initial_duration) * 100

    # Kapsam hesaplama
    feature_counts = np.dot(solution, coverage_matrix)
    total_features = coverage_matrix.shape[1]
    covered_features = int(np.sum(feature_counts > 0))
    coverage_pct = (covered_features / total_features) * 100

    # Tekrar hesaplama
    redundancy_count = int(np.sum(np.maximum(0, feature_counts - 1)))

    # Kopuk Bağımlılık (Broken Dependency) Kontrolü
    broken_dep_count = 0
    if dependency_matrix is not None:
        for i in selected_indices:
            required_parents = np.where(dependency_matrix[:, i] == 1)[0]
            for p in required_parents:
                if solution[p] == 0:
                    broken_dep_count += 1

    print(f"\n{'='*65}")
    print(f" {title}")
    print(f"{'='*65}")
    print(f"Toplam Test Sayısı          : {total_tests}")
    print(f"Seçilen Test Sayısı         : {selected_count} (Elendi: {total_tests - selected_count})")
    print(f"Başlangıç Koşum Süresi      : {initial_duration} sn")
    print(f"Optimize Koşum Süresi       : {optimized_duration} sn")
    print(f"Süre Tasarrufu Oranı        : %{time_reduction_pct:.2f}")
    print(f"Gereksinim/Hata Kapsamı     : %{coverage_pct:.2f} ({covered_features}/{total_features})")
    print(f"Gereksiz Tekrar Sayısı      : {redundancy_count}")
    print(f"Kopuk Bağımlılık Sayısı     : {broken_dep_count} (İhlal Edilen Ön Koşul)")
    print(f"Fail-Fast Erken Yakalama    : %{apfd_score:.2f} (APFD Skoru)")

    if total_tests <= 10:
        print("\nÖnceliklendirilmiş Çalıştırma Sırası (CI/CD Pipeline Planı):")
        for order, idx in enumerate(ordered_suite, 1):
            print(f"  {order}. Sıra -> {test_names[idx]} ({durations[idx]} sn)")
    print(f"{'='*65}\n")


def save_table_as_image(results_data, output_filepath):
    """Tüm yeni metrikleri içeren şık bir PNG tablo görseli oluşturur."""
    fig, ax = plt.subplots(figsize=(15, 3.5))
    ax.axis("tight")
    ax.axis("off")

    columns = [
        "Deney Adı",
        "Toplam",
        "Seçilen",
        "Elenen",
        "İlk Süre",
        "Opt. Süre",
        "Tasarruf",
        "Kapsam",
        "Tekrar",
        "Kopuk Ön Koşul",
        "APFD",
        "Çözüm Süresi",
    ]

    table = ax.table(
        cellText=results_data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 2.3)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1B263B")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F8F9FA" if row % 2 == 0 else "#FFFFFF")

    plt.savefig(output_filepath, bbox_inches="tight", dpi=300)
    plt.close()


def run_experiment():
    print("Yapay Zekâ Destekli Test Orkestrasyonu Başlatılıyor (BMGO)...\n")
    output_dir = "sonuclar"
    os.makedirs(output_dir, exist_ok=True)
    table_rows = []

    # ==========================================
    # DENEY 1: 10 Testlik E-Ticaret Örneği
    # ==========================================
    cov_10, dur_10, names_10, _, dep_10 = get_sample_ecommerce_data()

    start_time = time.time()
    best_sol_10, best_fit_10, curve_10 = binary_mountain_gazelle_optimizer(
        coverage_matrix=cov_10,
        durations=dur_10,
        dependency_matrix=dep_10,
        pop_size=25,
        max_iter=40,
        random_seed=42,
    )
    time_10 = time.time() - start_time

    order_10, apfd_10 = prioritize_tests_and_calculate_apfd(best_sol_10, cov_10, dur_10)

    print_suite_metrics(
        "DENEY 1: 10 Testlik E-Ticaret Regresyon Paketi",
        best_sol_10,
        cov_10,
        dur_10,
        names_10,
        apfd_10,
        order_10,
        dep_10,
    )
    print(f"Deney 1 Çözüm Süresi: {time_10:.3f} saniye")

    # Tablo satırı hesabı (Deney 1)
    sel_10 = int(np.sum(best_sol_10))
    time_save_10 = ((np.sum(dur_10) - np.sum(dur_10[best_sol_10 == 1])) / np.sum(dur_10)) * 100
    feat_10 = np.dot(best_sol_10, cov_10)
    red_10 = int(np.sum(np.maximum(0, feat_10 - 1)))
    broken_10 = sum(
        1 for i in np.where(best_sol_10 == 1)[0]
        for p in np.where(dep_10[:, i] == 1)[0] if best_sol_10[p] == 0
    )

    table_rows.append([
        "10 Testlik Paket",
        len(dur_10),
        sel_10,
        len(dur_10) - sel_10,
        f"{int(np.sum(dur_10))}s",
        f"{int(np.sum(dur_10[best_sol_10 == 1]))}s",
        f"%{time_save_10:.1f}",
        f"%100.0 ({np.sum(feat_10 > 0)}/5)",
        red_10,
        f"{broken_10} adet",
        f"%{apfd_10:.1f}",
        f"{time_10:.2f}s",
    ])

    # ==========================================
    # DENEY 2: 500 Testlik Büyük Ölçekli Paket
    # ==========================================
    print("\n500 Testlik Sentetik Paket Üretiliyor ve Optimize Ediliyor...")
    cov_500, dur_500, names_500, _, dep_500 = generate_synthetic_data(
        num_tests=500, num_features=30, seed=42
    )

    start_time = time.time()
    best_sol_500, best_fit_500, curve_500 = binary_mountain_gazelle_optimizer(
        coverage_matrix=cov_500,
        durations=dur_500,
        pop_size=40,
        max_iter=80,
        w_time=0.4,
        w_redundancy=0.6,
        random_seed=42,
    )
    time_500 = time.time() - start_time

    order_500, apfd_500 = prioritize_tests_and_calculate_apfd(best_sol_500, cov_500, dur_500)

    print_suite_metrics(
        "DENEY 2: 500 Testlik Endüstriyel Ölçekli Paket",
        best_sol_500,
        cov_500,
        dur_500,
        names_500,
        apfd_500,
        order_500,
        dep_500,
    )
    print(f"Deney 2 Çözüm Süresi: {time_500:.3f} saniye")

    # Tablo satırı hesabı (Deney 2)
    sel_500 = int(np.sum(best_sol_500))
    time_save_500 = ((np.sum(dur_500) - np.sum(dur_500[best_sol_500 == 1])) / np.sum(dur_500)) * 100
    feat_500 = np.dot(best_sol_500, cov_500)
    red_500 = int(np.sum(np.maximum(0, feat_500 - 1)))
    broken_500 = sum(
        1 for i in np.where(best_sol_500 == 1)[0]
        for p in np.where(dep_500[:, i] == 1)[0] if best_sol_500[p] == 0
    )

    table_rows.append([
        "500 Testlik Paket",
        len(dur_500),
        sel_500,
        len(dur_500) - sel_500,
        f"{int(np.sum(dur_500))}s",
        f"{int(np.sum(dur_500[best_sol_500 == 1]))}s",
        f"%{time_save_500:.1f}",
        f"%100.0 ({np.sum(feat_500 > 0)}/30)",
        red_500,
        f"{broken_500} adet",
        f"%{apfd_500:.1f}",
        f"{time_500:.2f}s",
    ])

    # ==========================================
    # 3. YENİ TABLOYU PNG OLARAK KAYDETME
    # ==========================================
    table_path = os.path.join(output_dir, "sonuclar_tablosu.png")
    save_table_as_image(table_rows, table_path)
    print(f"Genişletilmiş Tablo PNG kaydedildi: -> {table_path}")

    # ==========================================
    # 4. YAKINSAMA GRAFİKLERİNİ KAYDETME
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(curve_10, color="#1D3557", linewidth=2, marker="o", markersize=4)
    ax1.set_title("10 Testlik Paket Yakınsama Eğrisi")
    ax1.set_xlabel("İterasyon")
    ax1.set_ylabel("Fitness Değeri")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(curve_500, color="#E63946", linewidth=2)
    ax2.set_title("500 Testlik Paket Yakınsama Eğrisi")
    ax2.set_xlabel("İterasyon")
    ax2.set_ylabel("Fitness Değeri")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "convergence_results.png")
    plt.savefig(chart_path, dpi=300)
    print(f"Yakınsama Grafiği kaydedildi: -> {chart_path}")
    plt.show()


if __name__ == "__main__":
    run_experiment()