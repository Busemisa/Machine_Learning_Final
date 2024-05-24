Genel Bakış ve Amaç
Bu kodlar, farklı makine öğrenmesi algoritmalarını kullanarak bir diyabet veri seti üzerinde sınıflandırma modelleri oluşturmayı, değerlendirmeyi ve karşılaştırmayı amaçlamaktadır. Farklı modellerin performans metriklerini hesaplayarak, hangisinin daha iyi sonuçlar verdiğini belirlemek için kapsamlı bir analiz sunmaktadır. Kullanılan algoritmalar arasında Naive Bayes, K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), Support Vector Machine (SVM), ve RandomForestClassifier bulunmaktadır. Ayrıca, modellerin performansını optimize etmek için hiperparametre ayarları yapılmıştır.

Kodların Genel Yapısı ve İşleyişi
Veri Yükleme ve Hazırlama:

Veri Seti Yükleme: Veri seti diabetes.csv dosyasından yüklenir. Bu veri seti, diyabet teşhisi için kullanılan çeşitli özellikleri içerir.
Özellikler ve Hedef Değişken Ayrımı: Özellikler (X) ve hedef değişken (y) ayrılır.
Model Oluşturma ve Değerlendirme:

Naive Bayes:

Gaussian Naive Bayes modeli oluşturulur ve eğitim verisi ile eğitilir.
Eğitim ve test verisi üzerinde tahminler yapılır.
Performans metrikleri (doğruluk, kesinlik, hatırlama, F1 skoru) hesaplanır.
Karışıklık matrisleri ve ROC eğrisi görselleştirilir.
Sonuçlar CSV dosyalarına kaydedilir.
K-Nearest Neighbors (KNN):

Farklı k değerleri için çapraz doğrulama ile en iyi k değeri belirlenir.
En iyi k değeri ile KNN modeli oluşturulur ve eğitilir.
Eğitim ve test verisi üzerinde tahminler yapılır.
Performans metrikleri hesaplanır ve karışıklık matrisleri görselleştirilir.
Sonuçlar CSV dosyalarına kaydedilir ve ROC eğrisi çizilir.
Multi-Layer Perceptron (MLP) ve Support Vector Machine (SVM):

MLP ve SVM modelleri oluşturulur ve eğitim verisi ile eğitilir.
Eğitim ve test verisi üzerinde tahminler yapılır.
Performans metrikleri hesaplanır ve karışıklık matrisleri görselleştirilir.
Sonuçlar CSV dosyalarına kaydedilir ve ROC eğrisi çizilir.
RandomForestClassifier ile Hiperparametre Optimizasyonu:

Sayısal ve kategorik sütunlar belirlenir.
Ön işleme adımları (eksik değer doldurma, standardizasyon, one-hot encoding) uygulanır.
RandomForestClassifier modeli için Grid Search kullanılarak en iyi hiperparametreler belirlenir.
En iyi model seçilir ve çapraz doğrulama ile değerlendirilir.
Tahminler yapılır ve sınıflandırma raporu oluşturulur.
Sonuçlar CSV dosyasına kaydedilir.
Kullanılan Teknikler ve Yöntemler
Veri Ön İşleme: Eksik değerlerin doldurulması, verilerin ölçeklenmesi ve kategorik verilerin one-hot encoding ile işlenmesi.
Model Eğitimi ve Tahmin: Farklı makine öğrenmesi algoritmaları kullanılarak modellerin eğitilmesi ve tahminlerin yapılması.
Performans Değerlendirme: Doğruluk, kesinlik, hatırlama, F1 skoru gibi performans metriklerinin hesaplanması. Karışıklık matrislerinin ve ROC eğrisinin görselleştirilmesi.
Hiperparametre Optimizasyonu: Grid Search yöntemi ile en iyi hiperparametrelerin belirlenmesi.
Çapraz Doğrulama: Modelin genelleme yeteneğini değerlendirmek için çapraz doğrulama yönteminin kullanılması.
Sonuçların Kaydedilmesi: Performans metriklerinin, karışıklık matrislerinin ve diğer önemli sonuçların CSV dosyalarına kaydedilmesi.
Kodların Amacı ve Katkısı
Bu kodların amacı, diyabet veri seti üzerinde farklı makine öğrenmesi modellerinin performansını karşılaştırmaktır. Kullanıcılar, hangi modelin diyabet teşhisi için daha iyi sonuçlar verdiğini belirlemek için bu kodları kullanabilirler. Ayrıca, hiperparametre optimizasyonu ve çapraz doğrulama gibi ileri düzey tekniklerin nasıl uygulanacağını öğrenmek için de faydalı bir örnek sunar. Bu, hem akademik araştırmalar hem de pratik uygulamalar için değerli bir bilgi sağlar.

Sonuç
Bu kodlar, diyabet teşhisi için farklı makine öğrenmesi modellerini karşılaştırmak ve en iyi performansı elde etmek için kapsamlı bir yaklaşımla hazırlanmıştır. Veri ön işleme, model eğitimi, performans değerlendirme, ve hiperparametre optimizasyonu gibi adımları içerir. Kullanıcılar bu kodları kullanarak, farklı modellerin performansını değerlendirebilir ve en iyi sonucu veren modeli seçebilirler. Bu süreç, makine öğrenmesi modellerinin gerçek dünya uygulamalarında nasıl kullanılacağını anlamak için önemli bir rehber sunar.
