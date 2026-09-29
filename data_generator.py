import numpy as np


def get_sample_ecommerce_data():
    """Konuştuğumuz 10 testlik örnek e-ticaret regresyon paketi."""
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

    # Her testin koşulma süresi (saniye cinsinden)
    # Toplam süre: 285 saniye
    durations = np.array([10, 20, 35, 25, 60, 50, 15, 15, 30, 25])

    # Kapsam Matrisi (Coverage Matrix)
    # Satırlar: 10 Test Senaryosu
    # Sütunlar: 5 Temel Özellik [F1, F2, F3, F4, F5]
    # 1: Test bu özelliği içeriyor / test ediyor, 0: İçermiyor
    coverage_matrix = np.array(
        [
            [1, 0, 0, 0, 0],  # T1: F1
            [1, 1, 0, 0, 0],  # T2: F1, F2
            [1, 1, 1, 0, 0],  # T3: F1, F2, F3
            [0, 0, 1, 1, 0],  # T4: F3, F4
            [1, 1, 1, 0, 1],  # T5: F1, F2, F3, F5
            [0, 0, 1, 1, 1],  # T6: F3, F4, F5
            [0, 0, 1, 0, 0],  # T7: F3
            [0, 0, 0, 1, 0],  # T8: F4
            [0, 0, 0, 0, 1],  # T9: F5
            [0, 1, 1, 0, 0],  # T10: F2, F3
        ]
    )

    return coverage_matrix, durations, test_names, features


def generate_synthetic_data(num_tests=500, num_features=30, seed=42):
    """İleride ölçeklenebilirlik (scalability) için kullanılacak 500-1000 testlik sentetik veri üretici."""
    np.random.seed(seed)

    # 5 sn ile 120 sn arasında gerçekçi test süreleri
    durations = np.random.randint(5, 121, size=num_tests)

    # Her testin rastgele özellikleri kapsaması (ortalama %15-20 doluluk)
    coverage_matrix = (
        np.random.rand(num_tests, num_features) < 0.15
    ).astype(int)

    # Hiçbir özelliğin boş kalmadığından emin olmak için kontrol
    empty_features = np.where(coverage_matrix.sum(axis=0) == 0)[0]
    for feat in empty_features:
        random_test = np.random.randint(0, num_tests)
        coverage_matrix[random_test, feat] = 1

    test_names = [f"T_{i+1}" for i in range(num_tests)]
    features = [f"Feature_{j+1}" for j in range(num_features)]

    return coverage_matrix, durations, test_names, features


if __name__ == "__main__":
    cov, dur, tests, feats = get_sample_ecommerce_data()
    print("--- 10 Testlik Örnek Veri Seti Hazır ---")
    print(f"Toplam Test Sayısı: {len(tests)}")
    print(f"Toplam Özellik Sayısı: {len(feats)}")
    print(f"Tüm Testlerin Toplam Koşum Süresi: {dur.sum()} saniye")
    print("\nKapsam Matrisi Boyutu (Test x Özellik):", cov.shape)