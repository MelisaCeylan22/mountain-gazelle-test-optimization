import numpy as np


def calculate_fitness(
    solution,
    coverage_matrix,
    durations,
    dependency_matrix=None,
    w_time=0.4,
    w_redundancy=0.6,
):
    """Kapsam, süre, tekrar ve ön koşul bağımlılıklarını denetleyen amaç fonksiyonu."""
    num_tests, num_features = coverage_matrix.shape
    total_duration = np.sum(durations)

    if np.sum(solution) == 0:
        return 1e6

    # 1. SÜRE METRİĞİ
    selected_duration = np.sum(durations * solution)
    time_ratio = selected_duration / total_duration

    # 2. KAPSAM VE EKSİK CEZASI
    feature_test_counts = np.dot(solution, coverage_matrix)
    missing_features_count = np.sum(feature_test_counts == 0)

    if missing_features_count > 0:
        return 1000.0 * missing_features_count + time_ratio

    # 3. BAĞIMLILIK / ÖN KOŞUL KONTROLÜ (Broken Dependency Penalty)
    broken_dependencies = 0
    if dependency_matrix is not None:
        # Eğer T_i seçilmişse (solution[i]==1) ama ön koşulu T_j seçilmemişse (solution[j]==0)
        for i in range(num_tests):
            if solution[i] == 1:
                required_parents = np.where(dependency_matrix[:, i] == 1)[0]
                for p in required_parents:
                    if solution[p] == 0:
                        broken_dependencies += 1

    if broken_dependencies > 0:
        return 2000.0 * broken_dependencies + time_ratio

    # 4. TEKRAR (REDUNDANCY) METRİĞİ
    redundancy_counts = np.maximum(0, feature_test_counts - 1)
    total_redundancy = np.sum(redundancy_counts)
    total_feature_instances = np.sum(feature_test_counts)
    max_possible_redundancy = max(1, total_feature_instances - num_features)
    redundancy_ratio = (
        total_redundancy / max_possible_redundancy
        if max_possible_redundancy > 0
        else 0.0
    )

    # 5. BİRLEŞTİRİLMİŞ FITNESS
    return (w_time * time_ratio) + (w_redundancy * redundancy_ratio)


def prioritize_tests_and_calculate_apfd(solution, coverage_matrix, durations):
    """Seçilen testleri Fail-Fast prensibiyle önceliklendirir ve APFD skorunu hesaplar."""
    selected_indices = np.where(solution == 1)[0]
    if len(selected_indices) == 0:
        return [], 0.0

    # Her testin verimliliği: Kapsadığı Özellik Sayısı / Süresi
    efficiency = []
    for idx in selected_indices:
        feat_count = np.sum(coverage_matrix[idx])
        dur = max(1, durations[idx])
        efficiency.append(feat_count / dur)

    # Verimliliği en yüksek olan test en başa gelecek şekilde sırala (Büyükten küçüğe)
    sort_order = np.argsort(efficiency)[::-1]
    prioritized_suite = selected_indices[sort_order]

    # APFD (Average Percentage of Faults Detected) Hesaplama
    num_features = coverage_matrix.shape[1]
    n = len(prioritized_suite)
    first_detected_order = {}

    for order_idx, test_idx in enumerate(prioritized_suite):
        covered = np.where(coverage_matrix[test_idx] == 1)[0]
        for f in covered:
            if f not in first_detected_order:
                first_detected_order[f] = order_idx + 1  # 1-tabanlı sıra

    sum_ranks = sum(first_detected_order.values())
    apfd = 1.0 - (sum_ranks / (n * num_features)) + (1.0 / (2.0 * n))
    apfd_pct = max(0.0, min(100.0, apfd * 100))

    return prioritized_suite, apfd_pct