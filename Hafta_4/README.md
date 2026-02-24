# Veri Madenciliği ve İş Zekası - 4. Hafta Notları

## 1. Temel İstatistik ve Öğrenme Kavramları
* **Varyans:** Veri noktalarının aritmetik ortalamadan uzaklığı ve dağılım ölçüsü.
* **Underfitting (Eksik Öğrenme):** Modelin eğitim verisindeki örüntüyü yeterince öğrenememesi durumu.
* **Overfitting (Aşırı Öğrenme):** Modelin eğitim verisini ezberlemesi ve genelleme yeteneğini kaybederek yeni verilerde başarısız olması durumu. 

[Image of underfitting optimal overfitting diagram]

* **Kapasite Dengesi:** Underfitting ve overfitting'i önlemek için modelin karmaşıklığı (YZ kapasitesi) ile eğitim veri hacmi birbirine uygun olmalıdır.

## 2. Makine Öğrenmesi Türleri ve Değerlendirme
* **Supervised Learning (Gözetimli Öğrenme):** Modelin etiketlenmiş girdi ve çıktı verileri üzerinden eğitilmesi.
* **Unsupervised Learning (Gözetimsiz Öğrenme):** Etiketlenmemiş veriler kullanılarak verinin içindeki gizli yapıların veya kümelerin keşfedilmesi.
* **Confusion Matrix (Karmaşıklık Matrisi):** Sınıflandırma algoritmalarının performansını (Doğru/Yanlış Pozitif ve Negatif oranları) ölçmek için kullanılan hata matrisi.

## 3. Eğitim Süreci ve Algoritmalar
* **Eğitim (Training):** Veri uzayında noktaların konumlarının hesaplanması ve sınıf/küme sınırlarının belirlenmesi süreci.
* **K-Nearest Neighbors (K-En Yakın Komşu / KNN):** Bir noktanın sınıfını, veri uzayındaki en yakın 'K' adet komşusunun ağırlığına göre belirleyen algoritma. 
* **Merkez Nokta (Centroid / Central Point):** Kümeleme algoritmalarında (Örn. K-Means) bir kümenin merkezini temsil eden koordinat.
* **Başlatma (Initialization):** Kümeleme işlemlerinde belirsizlik durumunda, atanan ilk veri noktası merkez (central point) olarak kabul edilir ve iterasyonlarla yeri güncellenir.

## 4. Hiperparametre Optimizasyonu
* **Hiperparametre Tanımı:** Model eğitilmeden önce manuel olarak ayarlanan ve öğrenme sürecini kontrol eden parametreler. 
* **Değişkenlik:** Hiperparametreler hiçbir zaman sabit değildir; her veri setinin (dataset) yapısına göre değişiklik gösterir.
* **Grid Search (Izgara Arama):** Optimum hiperparametre kombinasyonunu bulmak için belirlenen değer aralıklarını sistematik olarak deneyen analiz yöntemidir (Örn: Scikit-learn `GridSearchCV`).