"""
Ridge ve Lasso Regresyon Modellerinin GridSearchCV ile Hiperparametreleri Taraması
Konu: Üç farklı hiperparametreyi tarayarak en iyi modeli bulma ve karşılaştırma
"""

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

print("="*90)
print("RIDGE VE LASSO REGRESYON MODELLERİ - GRIDSEARCHCV HİPERPARAMETRE TARAMASI")
print("="*90)

# ============================================================================
# 1. VERİ YÜKLEME VE HAZIRLAMA
# ============================================================================
print("\n" + "─"*90)
print("ADIM 1: VERİ YÜKLEME VE HAZIRLAMA")
print("─"*90)

# California Housing Dataset yükleme
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"\n✓ Veri Seti: California Housing")
print(f"  • Örnek Sayısı: {X.shape[0]:,}")
print(f"  • Özellik Sayısı: {X.shape[1]}")
print(f"  • Özellikler: {', '.join(feature_names)}")
print(f"  • Hedef: Konut Fiyatı (1000 USD cinsinden)")
print(f"  • Fiyat Aralığı: ${y.min():.2f}K - ${y.max():.2f}K")

# Veri standardizasyonu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim-Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n✓ Eğitim-Test Ayrımı (80-20):")
print(f"  • Eğitim Seti: {X_train.shape[0]:,} örnek")
print(f"  • Test Seti: {X_test.shape[0]:,} örnek")

# ============================================================================
# 2. RIDGE REGRESSION - HİPERPARAMETRE TARAMASI
# ============================================================================
print("\n" + "="*90)
print("ADIM 2: RIDGE REGRESSION - HİPERPARAMETRE TARAMASI")
print("="*90)

ridge_params = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],     # L2 düzenleme gücü
    'solver': ['auto', 'svd', 'cholesky'],      # Çözüm algoritması
    'fit_intercept': [True, False]              # Kesişim terimi
}

print("\n✓ Taranacak Hiperparametreler:")
print(f"\n  1. ALPHA (L2 Düzenleme Gücü):")
print(f"     Değerler: {ridge_params['alpha']}")
print(f"     Açıklama: Küçük değerler → daha az düzenleme, Büyük değerler → daha fazla düzenleme")

print(f"\n  2. SOLVER (Çözüm Algoritması):")
print(f"     Değerler: {ridge_params['solver']}")
print(f"     • auto: Otomatik seçim (giriş tipi tabanlı)")
print(f"     • svd: Tekil Değer Ayrışması (küçük veri setleri için iyi)")
print(f"     • cholesky: Cholesky ayrışması (hızlı, yüksek boyutlu veri için)")

print(f"\n  3. FIT_INTERCEPT (Kesişim Terimi):")
print(f"     Değerler: {ridge_params['fit_intercept']}")
print(f"     • True: Model kesişim terimi içerir")
print(f"     • False: Kesişim terimi olmadan eğilir")

total_ridge_combinations = len(ridge_params['alpha']) * len(ridge_params['solver']) * len(ridge_params['fit_intercept'])
print(f"\n✓ Toplam Kombinasyon Sayısı: {total_ridge_combinations}")
print(f"  (5-Fold CV ile: {total_ridge_combinations * 5} = {total_ridge_combinations * 5} model eğitilecek)")

print("\n⏳ Ridge Regression Eğitiliyor...")
ridge_grid = GridSearchCV(
    Ridge(), 
    ridge_params, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1,
    verbose=0
)
ridge_grid.fit(X_train, y_train)
print("✓ Ridge Regression Eğitimi Tamamlandı")

# Ridge tahminleri
ridge_pred_train = ridge_grid.predict(X_train)
ridge_pred_test = ridge_grid.predict(X_test)

# Ridge metrikleri
ridge_train_r2 = r2_score(y_train, ridge_pred_train)
ridge_test_r2 = r2_score(y_test, ridge_pred_test)
ridge_test_mse = mean_squared_error(y_test, ridge_pred_test)
ridge_test_rmse = np.sqrt(ridge_test_mse)
ridge_test_mae = mean_absolute_error(y_test, ridge_pred_test)

# ============================================================================
# 3. LASSO REGRESSION - HİPERPARAMETRE TARAMASI
# ============================================================================
print("\n" + "="*90)
print("ADIM 3: LASSO REGRESSION - HİPERPARAMETRE TARAMASI")
print("="*90)

lasso_params = {
    'alpha': [0.001, 0.01, 0.05, 0.1, 0.5],     # L1 düzenleme gücü (Lasso için daha düşük)
    'max_iter': [5000, 10000, 50000],           # Maksimum iterasyon
    'fit_intercept': [True, False]              # Kesişim terimi
}

print("\n✓ Taranacak Hiperparametreler:")
print(f"\n  1. ALPHA (L1 Düzenleme Gücü):")
print(f"     Değerler: {lasso_params['alpha']}")
print(f"     Açıklama: Bazı katsayıları tam sıfıra ayarlayarak feature selection yapar")

print(f"\n  2. MAX_ITER (Maksimum Iterasyon):")
print(f"     Değerler: {lasso_params['max_iter']}")
print(f"     Açıklama: Koordinat inişi algoritması için iterasyon sınırı")

print(f"\n  3. FIT_INTERCEPT (Kesişim Terimi):")
print(f"     Değerler: {lasso_params['fit_intercept']}")

total_lasso_combinations = len(lasso_params['alpha']) * len(lasso_params['max_iter']) * len(lasso_params['fit_intercept'])
print(f"\n✓ Toplam Kombinasyon Sayısı: {total_lasso_combinations}")
print(f"  (5-Fold CV ile: {total_lasso_combinations * 5} = {total_lasso_combinations * 5} model eğitilecek)")

print("\n⏳ Lasso Regression Eğitiliyor...")
lasso_grid = GridSearchCV(
    Lasso(random_state=42), 
    lasso_params, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1,
    verbose=0
)
lasso_grid.fit(X_train, y_train)
print("✓ Lasso Regression Eğitimi Tamamlandı")

# Lasso tahminleri
lasso_pred_train = lasso_grid.predict(X_train)
lasso_pred_test = lasso_grid.predict(X_test)

# Lasso metrikleri
lasso_train_r2 = r2_score(y_train, lasso_pred_train)
lasso_test_r2 = r2_score(y_test, lasso_pred_test)
lasso_test_mse = mean_squared_error(y_test, lasso_pred_test)
lasso_test_rmse = np.sqrt(lasso_test_mse)
lasso_test_mae = mean_absolute_error(y_test, lasso_pred_test)

# ============================================================================
# 4. RIDGE SONUÇLARI
# ============================================================================
print("\n" + "="*90)
print("ADIM 4: RIDGE REGRESSION - DETAYLI SONUÇLAR")
print("="*90)

print("\n✓ EN İYİ HİPERPARAMETRELER:")
for param, value in ridge_grid.best_params_.items():
    print(f"  • {param:18s}: {value}")

print(f"\n✓ EN İYİ HYPERPARAMETER KOMBİNASYONU İÇİN:")
print(f"  • Cross-Validation R² (CV): {ridge_grid.best_score_:.6f}")
print(f"  • En iyi CV sırası: {ridge_grid.best_index_ + 1}/{len(ridge_grid.cv_results_['mean_test_score'])}")

print(f"\n✓ TEST SETİ PERFORMANSI:")
print(f"  • Test R² Skoru:           {ridge_test_r2:.6f}")
print(f"  • Test MSE:                {ridge_test_mse:.6f}")
print(f"  • Test RMSE:               {ridge_test_rmse:.6f}")
print(f"  • Test MAE:                {ridge_test_mae:.6f}")
print(f"  • Eğitim R² Skoru:         {ridge_train_r2:.6f}")
print(f"  • Overfitting göstergesi:  {ridge_train_r2 - ridge_test_r2:.6f} (düşük = iyi)")

# Ridge katsayıları analizi
ridge_coef = ridge_grid.best_estimator_.coef_
ridge_intercept = ridge_grid.best_estimator_.intercept_

print(f"\n✓ KATSAYI ANALİZİ (Ridge):")
print(f"  • Kesişim Terimi: {ridge_intercept:.6f}")
print(f"  • Katsayıların Ortalama Mutlak Değeri: {np.mean(np.abs(ridge_coef)):.6f}")
print(f"  • Katsayıların Standart Sapması: {np.std(ridge_coef):.6f}")
print(f"  • En Büyük Katsayı (Mutlak): {np.max(np.abs(ridge_coef)):.6f}")

# ============================================================================
# 5. LASSO SONUÇLARI
# ============================================================================
print("\n" + "="*90)
print("ADIM 5: LASSO REGRESSION - DETAYLI SONUÇLAR")
print("="*90)

print("\n✓ EN İYİ HİPERPARAMETRELER:")
for param, value in lasso_grid.best_params_.items():
    print(f"  • {param:18s}: {value}")

print(f"\n✓ EN İYİ HYPERPARAMETER KOMBİNASYONU İÇİN:")
print(f"  • Cross-Validation R² (CV): {lasso_grid.best_score_:.6f}")
print(f"  • En iyi CV sırası: {lasso_grid.best_index_ + 1}/{len(lasso_grid.cv_results_['mean_test_score'])}")

print(f"\n✓ TEST SETİ PERFORMANSI:")
print(f"  • Test R² Skoru:           {lasso_test_r2:.6f}")
print(f"  • Test MSE:                {lasso_test_mse:.6f}")
print(f"  • Test RMSE:               {lasso_test_rmse:.6f}")
print(f"  • Test MAE:                {lasso_test_mae:.6f}")
print(f"  • Eğitim R² Skoru:         {lasso_train_r2:.6f}")
print(f"  • Overfitting göstergesi:  {lasso_train_r2 - lasso_test_r2:.6f} (düşük = iyi)")

# Lasso katsayıları analizi
lasso_coef = lasso_grid.best_estimator_.coef_
lasso_intercept = lasso_grid.best_estimator_.intercept_
zero_coef_count = np.sum(lasso_coef == 0)
non_zero_coef_count = np.sum(lasso_coef != 0)

print(f"\n✓ KATSAYI ANALİZİ (Lasso):")
print(f"  • Kesişim Terimi: {lasso_intercept:.6f}")
print(f"  • Sıfır Katsayısı Sayısı: {zero_coef_count}/{len(lasso_coef)} özellik")
print(f"  • Sıfırlanmayan Katsayı Sayısı: {non_zero_coef_count}/{len(lasso_coef)}")
print(f"  • Feature Selection Oranı: %{(zero_coef_count/len(lasso_coef)*100):.1f}")

if non_zero_coef_count > 0:
    print(f"  • Seçilen Özelliklerin Ortalama Mutlak Değeri: {np.mean(np.abs(lasso_coef[lasso_coef != 0])):.6f}")
    print(f"  • Seçilen Özelliklerin Standart Sapması: {np.std(lasso_coef[lasso_coef != 0]):.6f}")

print(f"\n✓ SEÇILEN ÖZELLİKLER (Sıfır olmayan katsayılar):")
selected_features = np.where(lasso_coef != 0)[0]
for idx in selected_features:
    print(f"  • {feature_names[idx]:30s}: {lasso_coef[idx]:10.6f}")

# ============================================================================
# 6. MODELLERİN KARŞILAŞTIRILMASI
# ============================================================================
print("\n" + "="*90)
print("ADIM 6: MODELLERİN KARŞILAŞTIRILMASI")
print("="*90)

comparison_data = {
    'Metrik': [
        'CV R² Skoru',
        'Test R² Skoru',
        'Test MSE',
        'Test RMSE',
        'Test MAE',
        'Eğitim R²',
        'Overfitting Farkı'
    ],
    'Ridge': [
        f"{ridge_grid.best_score_:.6f}",
        f"{ridge_test_r2:.6f}",
        f"{ridge_test_mse:.6f}",
        f"{ridge_test_rmse:.6f}",
        f"{ridge_test_mae:.6f}",
        f"{ridge_train_r2:.6f}",
        f"{ridge_train_r2 - ridge_test_r2:.6f}"
    ],
    'Lasso': [
        f"{lasso_grid.best_score_:.6f}",
        f"{lasso_test_r2:.6f}",
        f"{lasso_test_mse:.6f}",
        f"{lasso_test_rmse:.6f}",
        f"{lasso_test_mae:.6f}",
        f"{lasso_train_r2:.6f}",
        f"{lasso_train_r2 - lasso_test_r2:.6f}"
    ],
    'Fark (Ridge-Lasso)': [
        f"{float(ridge_grid.best_score_) - float(lasso_grid.best_score_):+.6f}",
        f"{ridge_test_r2 - lasso_test_r2:+.6f}",
        f"{ridge_test_mse - lasso_test_mse:+.6f}",
        f"{ridge_test_rmse - lasso_test_rmse:+.6f}",
        f"{ridge_test_mae - lasso_test_mae:+.6f}",
        f"{ridge_train_r2 - lasso_train_r2:+.6f}",
        f"{(ridge_train_r2 - ridge_test_r2) - (lasso_train_r2 - lasso_test_r2):+.6f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# 7. EN İYİ MODEL SEÇİMİ VE YORUMLAMA
# ============================================================================
print("\n" + "="*90)
print("ADIM 7: EN İYİ MODEL SEÇİMİ VE YORUMLAMA")
print("="*90)

# R² temelinde karşılaştırma
r2_diff = ridge_test_r2 - lasso_test_r2

if ridge_test_r2 > lasso_test_r2:
    best_model = "RIDGE"
    better_score = ridge_test_r2
    worse_score = lasso_test_r2
else:
    best_model = "LASSO"
    better_score = lasso_test_r2
    worse_score = ridge_test_r2

print(f"\n🏆 EN İYİ MODEL: {best_model} REGRESSION")
print(f"   Test R² Skoru: {better_score:.6f}")
print(f"   Diğer Modelin R² Skoru: {worse_score:.6f}")
print(f"   Fark: {abs(r2_diff):.6f}")
print(f"   Başarı Yüzdesi: %{(better_score*100):.2f}")

# ============================================================================
# 8. DETAYLI YORUMLAMA
# ============================================================================
print("\n" + "="*90)
print("ADIM 8: DETAYLI YORUMLAMA VE İÇGÖRÜLER")
print("="*90)

print("\n" + "─"*90)
print("1. RIDGE REGRESSION YORUMU:")
print("─"*90)

ridge_best_alpha = ridge_grid.best_params_['alpha']
print(f"\n✓ En İyi Alpha Değeri: {ridge_best_alpha}")
if ridge_best_alpha <= 0.1:
    print(f"  → ZAYıF DÜZENLEME: Model veri setine daha yakından uyar")
    print(f"  → Daha DÜŞÜK bias, daha YÜKSEK variance")
elif ridge_best_alpha >= 10:
    print(f"  → GÜÇLÜ DÜZENLEME: Model veri setine daha az uyar")
    print(f"  → Daha YÜKSEK bias, daha DÜŞÜK variance")
else:
    print(f"  → ORTA DÜZENLEME: Bias-variance trade-off dengesi")

print(f"\n✓ Solver Seçimi: {ridge_grid.best_params_['solver']}")
print(f"  → Bu veri seti büyüklüğü ve özellik sayısı için uygun seçim")

print(f"\n✓ Kesişim Terimi: {'Kullanılıyor' if ridge_grid.best_params_['fit_intercept'] else 'Kullanılmıyor'}")

print(f"\n✓ Model Özellikleri:")
print(f"  • L2 Düzenleme: Tüm katsayıları küçültür (sıfırlama yapmaz)")
print(f"  • Tüm {len(ridge_coef)} özelliği modelde tutar")
print(f"  • Daha kararlı ve robust tahminler üretir")
print(f"  • Çoklu doğrusallık (multicollinearity) problemine karşı dayanıklı")

print(f"\n✓ Performans Özeti:")
print(f"  • CV R² Skoru: {ridge_grid.best_score_:.6f}")
print(f"  • Test R² Skoru: {ridge_test_r2:.6f}")
if ridge_test_r2 > 0.9:
    print(f"  • Sonuç: ÇOK İYİ performans (%{ridge_test_r2*100:.1f} açıklanabilir varyans)")
elif ridge_test_r2 > 0.7:
    print(f"  • Sonuç: İYİ performans (%{ridge_test_r2*100:.1f} açıklanabilir varyans)")
else:
    print(f"  • Sonuç: UYGUN performans (%{ridge_test_r2*100:.1f} açıklanabilir varyans)")

print("\n" + "─"*90)
print("2. LASSO REGRESSION YORUMU:")
print("─"*90)

lasso_best_alpha = lasso_grid.best_params_['alpha']
print(f"\n✓ En İyi Alpha Değeri: {lasso_best_alpha}")
if lasso_best_alpha <= 0.01:
    print(f"  → ZAYıF DÜZENLEME: Daha fazla özellik seçilir")
elif lasso_best_alpha >= 0.1:
    print(f"  → GÜÇLÜ DÜZENLEME: Daha az özellik seçilir (sparser model)")
else:
    print(f"  → ORTA DÜZENLEME: Dengeli feature selection")

print(f"\n✓ Max Iterasyon: {lasso_grid.best_params_['max_iter']}")
print(f"  → Koordinat inişi algoritması bu iterasyon sayısında yakınsadı")

print(f"\n✓ Kesişim Terimi: {'Kullanılıyor' if lasso_grid.best_params_['fit_intercept'] else 'Kullanılmıyor'}")

print(f"\n✓ Model Özellikleri:")
print(f"  • L1 Düzenleme: Bazı katsayıları tamamen sıfıra ayarlar")
print(f"  • Feature Selection: Önemli özellikleri otomatik seçer")
print(f"  • Seçilen Özellik Sayısı: {non_zero_coef_count}/{len(lasso_coef)}")
if zero_coef_count > 0:
    print(f"  • Boyut Azaltma: %{(zero_coef_count/len(lasso_coef)*100):.1f} özellik eliminasyonu")
print(f"  • Daha sade ve yorumlanabilir modeller oluşturur")
print(f"  • Özellik seçimi yapmak istenen problemler için ideal")

print(f"\n✓ Performans Özeti:")
print(f"  • CV R² Skoru: {lasso_grid.best_score_:.6f}")
print(f"  • Test R² Skoru: {lasso_test_r2:.6f}")
if lasso_test_r2 > 0.9:
    print(f"  • Sonuç: ÇOK İYİ performans (%{lasso_test_r2*100:.1f} açıklanabilir varyans)")
elif lasso_test_r2 > 0.7:
    print(f"  • Sonuç: İYİ performans (%{lasso_test_r2*100:.1f} açıklanabilir varyans)")
else:
    print(f"  • Sonuç: UYGUN performans (%{lasso_test_r2*100:.1f} açıklanabilir varyans)")

print("\n" + "─"*90)
print("3. RIDGE vs LASSO KARŞILAŞTIRMASI:")
print("─"*90)

print(f"\n✓ Performans Farkı (Test R²):")
if abs(r2_diff) < 0.01:
    print(f"  • Modeller benzer performans gösteriyor ({abs(r2_diff):.6f} fark)")
else:
    print(f"  • Ridge: {ridge_test_r2:.6f}, Lasso: {lasso_test_r2:.6f}")
    print(f"  • Fark: {abs(r2_diff):.6f}")

print(f"\n✓ Overfitting Karşılaştırması:")
ridge_overfit = ridge_train_r2 - ridge_test_r2
lasso_overfit = lasso_train_r2 - lasso_test_r2

if ridge_overfit < lasso_overfit:
    print(f"  • Ridge daha az overfitting gösteriyor")
    print(f"    Ridge overfitting: {ridge_overfit:.6f}")
    print(f"    Lasso overfitting: {lasso_overfit:.6f}")
else:
    print(f"  • Lasso daha az overfitting gösteriyor")
    print(f"    Ridge overfitting: {ridge_overfit:.6f}")
    print(f"    Lasso overfitting: {lasso_overfit:.6f}")

print(f"\n✓ Modellerin Davranışı:")
print(f"  Ridge:")
print(f"    - Tüm özelliği tutuyor ({len(ridge_coef)} özellik)")
print(f"    - Katsayılar küçük ama sıfır değil")
print(f"    - Korelasyonlu özellikler arasında yük paylaşılıyor")
print(f"\n  Lasso:")
print(f"    - Önemli özellikleri seçiyor ({non_zero_coef_count}/{len(lasso_coef)})")
print(f"    - Gereksiz özellikleri tamamen eliminasyonu")
print(f"    - Daha interpretable (yorumlanabilir) model")

# ============================================================================
# 9. SONUÇ VE ÖNERİ
# ============================================================================
print("\n" + "="*90)
print("ADIM 9: SONUÇ VE ÖNERİ")
print("="*90)

print(f"\n🎯 GENEL SONUÇ:\n")

if best_model == "RIDGE":
    print(f"✅ SEÇİLEN MODEL: RIDGE REGRESSION")
    print(f"\n   Nedenleri:")
    print(f"   1. Daha Yüksek Test R² Skoru: {ridge_test_r2:.6f} ({abs(r2_diff)*100:.2f}% daha iyi)")
    print(f"   2. Daha Stabil Performans: Tüm özellikleri kullanarak daha robust")
    print(f"   3. Daha Düşük Overfitting: {ridge_overfit:.6f} (Lasso: {lasso_overfit:.6f})")
    print(f"   4. En İyi Hiperparametreler:")
    for param, value in ridge_grid.best_params_.items():
        print(f"      • {param}: {value}")
    print(f"\n   Kullanım Alanları:")
    print(f"   • Korelasyonlu özelliklerin olduğu durumlarda")
    print(f"   • Tüm özelliklerin modelde tutulması gereken durumlarda")
    print(f"   • Maksimum doğruluk istenen uygulamalarda")

else:
    print(f"✅ SEÇİLEN MODEL: LASSO REGRESSION")
    print(f"\n   Nedenleri:")
    print(f"   1. Daha Yüksek Test R² Skoru: {lasso_test_r2:.6f} ({abs(r2_diff)*100:.2f}% daha iyi)")
    print(f"   2. Otomatik Feature Selection: {non_zero_coef_count}/{len(lasso_coef)} özellik seçedi")
    print(f"   3. Daha Sade Model: %{(zero_coef_count/len(lasso_coef)*100):.1f} boyut azaltma")
    print(f"   4. Daha Düşük Overfitting: {lasso_overfit:.6f} (Ridge: {ridge_overfit:.6f})")
    print(f"   5. En İyi Hiperparametreler:")
    for param, value in lasso_grid.best_params_.items():
        print(f"      • {param}: {value}")
    print(f"\n   Kullanım Alanları:")
    print(f"   • Özelliklerin sayısı çok fazla olduğunda")
    print(f"   • Model interpretability (yorumlanabilirlik) önemli olduğunda")
    print(f"   • Gereksiz özellikleri otomatik olarak elemiş istediğinizde")

print("\n" + "="*90)
print("Analiz Tamamlandı!")
print("="*90)
