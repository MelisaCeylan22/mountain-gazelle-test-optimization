import numpy as np
from fitness import calculate_fitness


def v_shaped_transfer(x):
    """Sürekli konumu 0-1 aralığında olasılığa dönüştüren

    V-şekilli transfer fonksiyonu: T(x) = |tanh(x)|
    """
    return np.abs(np.tanh(x))


def binary_mountain_gazelle_optimizer(
    coverage_matrix,
    durations,
    dependency_matrix=None,
    pop_size=20,
    max_iter=50,
    w_time=0.4,
    w_redundancy=0.6,
    random_seed=42,
):
    """Binary Mountain Gazelle Optimizer (BMGO) - Dağ Ceylanı Algoritması"""
    if random_seed is not None:
        np.random.seed(random_seed)

    dim = len(durations)  # Boyut: Test senaryosu sayısı

    # 1. BAŞLANGIÇ POPÜLASYONU
    continuous_pop = np.random.uniform(-2.0, 2.0, (pop_size, dim))
    binary_pop = np.zeros((pop_size, dim), dtype=int)

    for i in range(pop_size):
        prob = v_shaped_transfer(continuous_pop[i])
        binary_pop[i] = (np.random.rand(dim) < prob).astype(int)
        if np.sum(binary_pop[i]) == 0:
            binary_pop[i, np.random.randint(0, dim)] = 1

    # Başlangıç Fitness Hesaplama
    fitness_values = np.zeros(pop_size)
    for i in range(pop_size):
        fitness_values[i] = calculate_fitness(
            solution=binary_pop[i],
            coverage_matrix=coverage_matrix,
            durations=durations,
            dependency_matrix=dependency_matrix,
            w_time=w_time,
            w_redundancy=w_redundancy,
        )

    best_idx = np.argmin(fitness_values)
    best_fitness = fitness_values[best_idx]
    best_solution = binary_pop[best_idx].copy()
    best_continuous = continuous_pop[best_idx].copy()

    convergence_curve = np.zeros(max_iter)

    # 2. OPTİMİZASYON DÖNGÜSÜ
    for iteration in range(max_iter):
        a = -1.0 + iteration * ((-1.0) / max_iter)

        new_continuous_list = []

        for i in range(pop_size):
            X_cur = continuous_pop[i]

            F = np.random.randn(dim) * np.exp(
                2.0 - iteration * (2.0 / max_iter)
            )

            cof_1 = (a + 1.0) + np.random.rand() * a * np.random.randn(dim)
            cof_2 = (a + 1.0) + np.random.rand() * a * np.random.randn(dim)
            cof_3 = (a + 1.0) + np.random.rand() * a * np.random.randn(dim)

            rand_idx = np.random.randint(0, pop_size)
            X_rand = continuous_pop[rand_idx]
            BH = X_rand + np.mean(continuous_pop, axis=0) * np.random.rand()

            ri_1 = np.random.choice([1, 2])
            ri_2 = np.random.choice([1, 2])
            ri_3 = np.random.choice([1, 2])
            ri_4 = np.random.choice([1, 2])

            # Mekanizma 1: TSM (Bölgeci Erkekler - Sömürü)
            TSM = best_continuous - np.abs((ri_1 * BH - ri_2 * X_cur) * F) * cof_1

            # Mekanizma 2: MH (Anaç Sürüsü)
            MH = (BH + cof_2) + (ri_3 * best_continuous - ri_4 * X_rand) * cof_3

            # Mekanizma 3: BMH (Bekâr Erkek Sürüsü - Keşif)
            D = (np.abs(X_cur) + np.abs(best_continuous)) * (
                2.0 * np.random.rand() - 1.0
            )
            BMH = (X_cur - D) + (ri_1 * best_continuous - ri_2 * BH) * cof_1

            # Mekanizma 4: MSF (Göç / Yırtıcıdan Kaçış - Ani Sıçrama)
            ub, lb = 2.0, -2.0
            MSF = (ub - lb) * np.random.rand(dim) + lb

            new_continuous_list.extend([TSM, MH, BMH, MSF])

        # 3. DÖNÜŞÜM VE FITNESS HESAPLAMA
        all_candidates_cont = np.vstack(
            [continuous_pop, np.array(new_continuous_list)]
        )
        all_candidates_cont = np.clip(all_candidates_cont, -4.0, 4.0)

        total_candidates = len(all_candidates_cont)
        all_candidates_bin = np.zeros((total_candidates, dim), dtype=int)
        all_fitness = np.zeros(total_candidates)

        for k in range(total_candidates):
            prob = v_shaped_transfer(all_candidates_cont[k])
            all_candidates_bin[k] = (np.random.rand(dim) < prob).astype(int)

            if np.sum(all_candidates_bin[k]) == 0:
                all_candidates_bin[k, np.random.randint(0, dim)] = 1

            all_fitness[k] = calculate_fitness(
                solution=all_candidates_bin[k],
                coverage_matrix=coverage_matrix,
                durations=durations,
                dependency_matrix=dependency_matrix,
                w_time=w_time,
                w_redundancy=w_redundancy,
            )

        # 4. HAYATTA KALMA SEÇİLİMİ
        sorted_indices = np.argsort(all_fitness)
        survivors = sorted_indices[:pop_size]

        continuous_pop = all_candidates_cont[survivors]
        binary_pop = all_candidates_bin[survivors]
        fitness_values = all_fitness[survivors]

        if fitness_values[0] < best_fitness:
            best_fitness = fitness_values[0]
            best_solution = binary_pop[0].copy()
            best_continuous = continuous_pop[0].copy()

        convergence_curve[iteration] = best_fitness

    return best_solution, best_fitness, convergence_curve