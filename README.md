# VpnAttack

Proje Amacı
Bu projenin amacı, VPN kullanılarak yapılan saldırıları tespit etmektir. Veriler üzerinde yapılan ön işleme ve sınıflandırma adımlarıyla, VPN trafiği ile normal trafik arasındaki farklar analiz edilmiştir. Model, yeni gelen verileri sınıflandırarak VPN üzerinden yapılan saldırıları tespit etmeyi hedefler.

Kullanılan Kütüphaneler
Pandas: Veri işleme ve analiz için kullanıldı.
Scikit-learn (sklearn): Makine öğrenimi modelleri ve algoritmaları için kullanıldı (özellikle En Yakın Komşu algoritması).

Adımlar
Veri Seti: Veri seti, VPN saldırıları ile normal trafik verilerini içermektedir.
Veri Ön İşleme: Eksik veriler, kategorik değişkenler ve normalizasyon işlemleri gerçekleştirilmiştir.
Modelleme: En Yakın Komşu (KNN) algoritması kullanılarak sınıflandırma yapılmıştır.
Model Performansı: Modelin doğruluğu ve performansı, test verisi üzerinde değerlendirilmiştir.


Veri seti "new_train_data.parquet".
