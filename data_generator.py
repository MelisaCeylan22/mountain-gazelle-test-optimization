import numpy as np


def get_sample_ecommerce_data():
    """10 testlik e-ticaret senaryosu, süreler, özellikler ve bağımlılıklar."""
    features = [
        "F1: Arama",
        "F2: Urun Detay",
        "F3: Sepete Ekleme",
        "F4: Kupon Uygulama",
        "F5: Odeme & Satin Alma",
    ]

    test_names = [
        "T1: Sadece Arama Yapma",
        "T2: Detay Sayfasi Inceleme",
        "T3: Sepete Urun Ekleme",
        "T4: Sepette Kupon Kodu Kullanma",
        "T5: Uctan Uca Satin Alma (Happy Path)",
        "T6: Kuponlu Satin Alma",
        "T7: Sepetten Urun Cikarma & Ekleme",
        "T8: Gecersiz Kupon Deneme",
        "T9: Sadece Odeme Ekrani Kontrolu",
        "T10: Detaydan Sepete Hizli Ekleme",
    ]

    # Test süreleri (saniye)
    durations = np.array([10, 20, 35, 25, 60, 50, 15, 15, 30, 25])

    # Kapsam Matrisi (10 x 5)
    coverage_matrix = np.array(
        [
            [1, 0, 0, 0, 0],  # T1
            [1, 1, 0, 0, 0],  # T2
            [1, 1, 1, 0, 0],  # T3
            [0, 0, 1, 1, 0],  # T4
            [1, 1, 1, 0, 1],  # T5
            [0, 0, 1, 1, 1],  # T6
            [0, 0, 1, 0, 0],  # T7
            [0, 0, 0, 1, 0],  # T8
            [0, 0, 0, 0, 1],  # T9
            [0, 1, 1, 0, 0],  # T10
        ]
    )

    # BAĞIMLILIK MATRİSİ (10 x 10)
    # dependency_matrix[j, i] = 1 ise T_i için T_j ön koşuldur (T_j -> T_i)
    num_tests = len(test_names)
    dependency_matrix = np.zeros((num_tests, num_tests), dtype=int)

    # Gerçekçi ön koşul zincirleri:
    dependency_matrix[0, 1] = 1  # T1 (Arama) koşulmadan T2 (Detay) incelenemez
    dependency_matrix[1, 2] = 1  # T2 (Detay) olmadan T3 (Sepet) yapılamaz
    dependency_matrix[2, 3] = 1  # T3 (Sepet) olmadan T4 (Kupon) uygulanamaz
    dependency_matrix[2, 6] = 1  # T3 (Sepet) olmadan T7 (Sepetten Çıkarma) olamaz
    dependency_matrix[3, 5] = 1  # T4 (Kupon) olmadan T6 (Kuponlu Satın Alma) olamaz

    return coverage_matrix, durations, test_names, features, dependency_matrix


def generate_synthetic_data(num_tests=500, num_features=30, seed=42):
    """500 testlik sentetik veri ve gerçekçi bağımlılık zinciri üretici."""
    np.random.seed(seed)

    durations = np.random.randint(5, 121, size=num_tests)
    coverage_matrix = (np.random.rand(num_tests, num_features) < 0.15).astype(int)

    empty_features = np.where(coverage_matrix.sum(axis=0) == 0)[0]
    for feat in empty_features:
        random_test = np.random.randint(0, num_tests)
        coverage_matrix[random_test, feat] = 1

    test_names = [f"T_{i+1}" for i in range(num_tests)]
    features = [f"Feature_{j+1}" for j in range(num_features)]

    # Sentetik Bağımlılık Matrisi: Her test sadece kendinden önceki testlere bağımlı olabilir (Döngü oluşmaz)
    dependency_matrix = np.zeros((num_tests, num_tests), dtype=int)
    for i in range(1, num_tests):
        if np.random.rand() < 0.10:  # %10 olasılıkla bir ön koşul testi olsun
            parent = np.random.randint(0, i)
            dependency_matrix[parent, i] = 1

    return coverage_matrix, durations, test_names, features, dependency_matrix