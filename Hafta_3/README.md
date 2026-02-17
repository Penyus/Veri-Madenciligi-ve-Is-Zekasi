# Iris Veri Seti ve Algoritma Notları

Bu proje, Iris veri seti kullanılarak makine öğrenmesi algoritmalarının temel çalışma prensiplerini ve veri işleme adımlarını içermektedir.

## 1. Veri Ayrıştırma (Data Splitting)

Veri seti eğitim ve test olarak ikiye ayrılırken rastgelelik faktörü sabitlenmelidir.
- **`random_state=0`**: Veri setinden alınan örneklemin (sample) her çalıştırıldığında **aynı rastgelelikte** olmasını sağlar. Bu, sonuçların tutarlı ve tekrarlanabilir olması için gereklidir.

## 2. Algoritma Çalışma Prensibi

Model, bir girdi verisinin ($x$) hangi sınıfa ait olduğunu belirlerken mesafe temelli bir yaklaşım kullanır. Eğer bir veri noktası kümelerin ortasında veya belirsiz bir konumda ise süreç şu şekilde işler:

1.  **Merkez Belirleme:** Öncelikle sınıflara ait kümelerin merkez noktaları (centroid) tespit edilir.
2.  **Mesafe Hesabı:** Yeni gelen $x$ noktası ile kümelerin merkez noktaları arasına sanal bir çizgi çekilir.
3.  **Öklid Mesafesi ($r$):** İki nokta arasındaki mesafe aşağıdaki formül ile hesaplanır:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

## 3. Eğitim (Training)

Makine öğrenmesinde **Training** (Eğitim) süreci; veri noktalarının analiz edilerek, her bir kümeyi en iyi temsil eden **merkez noktasının bulunması** işlemidir.
