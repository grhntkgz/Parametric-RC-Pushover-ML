# SAP2000 Python RC Model Generator

Bu proje SAP2000 OAPI / COM API kullanarak kontrollü-randomize betonarme 3B çerçeve bina modelleri üretir. Amaç nihai betonarme detay tasarımı değildir; TS 500 ve TBDY 2018'e açıkça aykırı olmayan, makul ön eleme sınırlarından geçen sentetik modeller oluşturmaktır.

## Dosya Yapısı

```text
.
├── config.py
├── design_rules.py
├── sap_api.py
├── model_generator.py
├── main.py
├── README.md
└── generated_models/
    ├── Model_0001_..._PushD030.sdb
    ├── Model_0001_..._metadata.json
    └── models_metadata.csv
```

## Kurulum

Windows ortamında SAP2000 kurulu olmalı ve COM/OAPI erişimi etkin olmalıdır.

```powershell
python -m pip install comtypes
python main.py
```

Kontrol panelini açmak için:

```powershell
python dashboard/server.py
```

Ardından tarayıcıdan şu adrese gidin:

```text
http://127.0.0.1:8765
```

Ana ayarlar `config.py` içindedir:

- `n_iter`
- kat, aks, açıklık ve kat yüksekliği aralıkları
- beton ve çelik sınıfları
- kolon/kiriş kesit adayları
- donatı oranı sınırları
- pushover hedef öteleme oranı ve yük dağılımı
- ön eleme kontrol limitleri
- çıktı klasörü

## Notlar

SAP2000 OAPI imzaları sürüme göre küçük farklılıklar gösterebilir. `sap_api.py` içindeki tüm SAP2000 çağrıları dönüş kodu kontrolü yapar ve başarısız çağrıda anlamlı hata üretir. Her uygun modelde X/Y yönlerinde pushover lateral yük paternleri ve nonlinear static pushover case'leri tanımlanır. `run_analysis_after_save=True` ise model kaydedildikten sonra analiz çalıştırılır ve dosya tekrar kaydedilir.

Pushover hedef deplasmanı `pushover_target_drift_ratio * toplam_yukseklik` olarak üretilir. Bu oran `config.py` içinde aralık ve adım olarak tutulur; böylece ileride kolon/kiriş boyutu, donatı oranı ve yükleme parametreleriyle birlikte optimizasyon değişkeni yapılabilir. Plastik mafsal atama ve pushover sonuç okuma bu aşamada bilinçli olarak eklenmemiştir, fakat mimaride bu adımlar için yer bırakılmıştır.
