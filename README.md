# malmem-xai-detection

## Bellek Dökümleri Üzerinden Makine Öğrenmesi ile Gizlenmiş Zararlı Yazılım Tespiti (Explainable AI - XAI)

Bu proje, bellek dökümleri (memory dumps) üzerinden gelişmiş ve gizlenmiş (obfuscated) zararlı yazılımların makine öğrenmesi ve yapay zeka yöntemleriyle tespit edilmesini ve karar mekanizmalarının SHAP (SHapley Additive exPlanations) ile açıklanmasını amaçlamaktadır.

---

## 👥 Proje Ekibi
* **Selim Şahin**
* **Ahmet Çöl**

---

## 📌 Proje Özeti ve Amacı
Geleneksel imza tabanlı antivirüs ve güvenlik yazılımları; paketleyiciler (packers), kod şifreleme ve dinamik bellek enjeksiyonu gibi gizleme (obfuscation) teknikleri kullanan modern tehditleri tespit etmekte yetersiz kalmaktadır. 

Bu çalışmada, Kanada Siber Güvenlik Enstitüsü (UNB CIC) tarafından yayımlanan **CIC-MalMem-2022** veri seti kullanılmıştır. Veri seti, Volatility bellek adli bilişim aracıyla çıkarılan süreç, iş parçacığı, DLL, handle, sürücü, servis ve bellek enjeksiyonu özniteliklerini içermektedir. Projede **XGBoost** ve **LightGBM** algoritmaları hem **ikili (Binary: Benign vs Malware)** hem de **çok sınıflı (Multiclass: Benign, Ransomware, Spyware, Trojan)** olarak eğitilmiş, ardından **TreeExplainer** tabanlı SHAP analizi ile zararlı yazılımların RAM üzerinde bıraktığı izler tespit edilmiştir.

---

## 📊 Veri Seti Bilgileri: CIC-MalMem-2022
* **Kaynak:** University of New Brunswick (UNB) - Canadian Institute for Cybersecurity
* **Dosya:** `Obfuscated-MalMem2022.csv`
* **Toplam Satır:** 58,596 (Tekrarlayan 534 kayıt temizlendikten sonra: 58,062 benzersiz satır)
* **Öznitelik Sayısı:** 55 adet sayısal bellek dökümü özniteliği
* **Hedef Sınıflar:**
  * **İkili Sınıflandırma (Class):**
    * `Benign`: 29,231 (%50.34)
    * `Malware`: 28,831 (%49.66)
  * **Çok Sınıflı Sınıflandırma (Category):**
    * `Benign`: 29,231 (%50.34)
    * `Spyware`: 9,815 (%16.90)
    * `Ransomware`: 9,529 (%16.41)
    * `Trojan`: 9,487 (%16.34)

### Bellek Analizi Modülleri (Volatility)
* `pslist`: Süreç ve iş parçacığı istatistikleri
* `dlllist`: Yüklenen dinamik bağlantı kitaplıkları (DLL)
* `handles`: Süreçler tarafından açılan handle, port, muteks, dosya ve olay nesneleri
* `ldrmodules`: Süreç bellek blokları ile modül listesi arasındaki uyumsuzluklar (DLL gizleme)
* `malfind`: Gizli veya enjekte edilmiş kod blokları, bellek koruma izinleri (PAGE_EXECUTE_READWRITE)
* `psxview`: Gizlenmiş süreçlerin tespiti (rootkit mekanizmaları)
* `svcscan`: Windows servis kayıtları ve sürücü servisleri
* `callbacks`: Çekirdek seviyesindeki geri çağırma (callback) fonksiyonları

---

## 📁 Proje Dosya Yapısı
```text
.
├── data/
│   ├── Obfuscated-MalMem2022.csv     # Ham CIC-MalMem-2022 veri seti
│   ├── X_train.csv                   # %80 Eğitim özellik matrisi
│   ├── X_test.csv                    # %20 Test özellik matrisi
│   ├── y_train_class.csv             # İkili eğitim hedefleri
│   ├── y_test_class.csv              # İkili test hedefleri
│   ├── y_train_category.csv          # Çok sınıflı eğitim hedefleri
│   ├── y_test_category.csv           # Çok sınıflı test hedefleri
│   └── preprocessing_metadata.json   # Ön işleme meta verileri
├── outputs/
│   ├── model_karsilastirma.csv       # Modellerin metrik karşılaştırma tablosu
│   ├── confusion_matrix.png          # 300 DPI Karmaşıklık matrisleri grafiği
│   ├── best_model.joblib             # En başarılı eğitilmiş model nesnesi
│   ├── best_model_info.json          # En başarılı modelin meta bilgileri
│   ├── shap_summary.png              # 300 DPI SHAP Beeswarm dağılım grafiği
│   ├── shap_bar.png                  # 300 DPI SHAP Global öznitelik önem grafiği
│   └── shap_feature_importance.csv   # Detaylı SHAP önem sıralaması tablosu
├── 1_veri_on_isleme.py               # Eksik/tekrar temizleme, hedef kodlama, stratified ayırma
├── 2_model_egitimi.py                # XGBoost ve LightGBM model eğitimi ve metrik değerlendirme
├── 3_shap_analizi.py                 # TreeExplainer ile bellek dökümü açıklanabilirlik analizi
├── requirements.txt                  # Gerekli Python kütüphaneleri
└── README.md                         # Proje dokümantasyonu
```

---

## ⚙️ Kurulum ve Gereksinimler

Proje **Python 3.9+** (tercihen 3.10 - 3.12) sürümleriyle uyumludur.

### 1. Sanal Ortam Oluşturma (Önerilen)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 2. Bağımlılıkların Kurulması
```bash
pip install -r requirements.txt
```

---

## 🚀 Projeyi Çalıştırma Adımları

Proje modüler olarak 3 ardışık adımdan oluşmaktadır:

### Adım 1: Veri Ön İşleme
Ham veri setini temizler, `Class` ve `Category` etiketlerini kodlar ve %80 eğitim / %20 test olarak stratified ayırarak `data/` altına kaydeder:
```bash
python 1_veri_on_isleme.py
```

### Adım 2: Model Eğitimi ve Karşılaştırma
XGBoost ve LightGBM modellerini eğitir, test seti üzerinde metrikleri hesaplar, terminale karşılaştırma tablosunu yazdırır, `outputs/model_karsilastirma.csv` dosyasını ve 300 DPI `outputs/confusion_matrix.png` grafiğini üretir:
```bash
python 2_model_egitimi.py
```

### Adım 3: SHAP ile Açıklanabilirlik Analizi
En başarılı model üzerinde `shap.TreeExplainer` çalıştırarak zararlı yazılımları açığa çıkaran kritik bellek parametrelerini tespit eder, 300 DPI `outputs/shap_summary.png` ve `outputs/shap_bar.png` grafiklerini oluşturur:
```bash
python 3_shap_analizi.py
```

---

## 📈 Deneysel Sonuçlar ve Model Karşılaştırması

Test kümesi (11,613 örnek) üzerinde elde edilen sonuçlar:

| Görev | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Eğitim Süresi (sn) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **İkili Sınıflandırma (Class)** | **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.40 |
| **İkili Sınıflandırma (Class)** | **LightGBM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.27 |
| **Çok Sınıflı (Category)** | **LightGBM** | **0.8564** | **0.8562** | **0.8564** | **0.8562** | **0.9758** | 0.99 |
| **Çok Sınıflı (Category)** | **XGBoost** | **0.8528** | **0.8529** | **0.8528** | **0.8527** | **0.9744** | 1.18 |

> **Önemli Bulgular:**
> * İkili sınıflandırmada (Benign vs Malware) hem XGBoost hem de LightGBM bellek dökümü özniteliklerinden %100 doğruluk ve kusursuz ROC-AUC skoru elde etmiştir. Bu durum, gizlenmiş zararlı yazılımların diskte kendilerini saklasalar dahi RAM üzerinde belirgin ve ayırt edilebilir anomali izleri bıraktığını kanıtlamaktadır.
> * Çok sınıflı sınıflandırmada (Ransomware, Spyware, Trojan ve Benign ayrımı) modeller %85.6+ doğruluk ve 0.975+ ROC-AUC performansına ulaşmıştır.

---

## 🔍 SHAP Açıklanabilirlik Analizi ve Güvenlik Bulguları

SHAP analizi sonucunda modellerin karar verirken en çok ağırlık verdiği bellek mekanizmaları ve Volatility modülleri tespit edilmiştir:

### En Etkili Bellek Parametreleri
1. **`svcscan.nservices` & `svcscan.process_services`:** Zararlı yazılımların kalıcılık (persistence) sağlamak veya ayrıcalık yükseltmek amacıyla Windows arka plan servislerini manipüle etmesi veya yeni servis kayıtları oluşturması en güçlü tespiti sağlamaktadır.
2. **`handles.avg_handles_per_proc` & `handles.nevent` / `handles.nthread`:** Zararlı yazılımların proses başına açtığı handle ve olay (event/synchronization) sayılarının normal yazılımlara kıyasla istatistiksel sapma göstermesi.
3. **`ldrmodules.not_in_load`:** Gizlenmiş zararlı yazılımların PEB (Process Environment Block) üzerinden modül bağlantılarını kopararak (DLL unlinking) kendilerini adli bilişim araçlarından saklama girişimleri.
4. **`malfind.uniqueInjections` & `malfind.commitCharge`:** Bellek alanına harici kod enjeksiyonu yapılması ve `PAGE_EXECUTE_READWRITE` korumalı hafıza bloklarının varlığı.
5. **`pslist.avg_threads` & `pslist.nppid`:** Şüpheli ebeveyn-çocuk proses ilişkileri (PID/PPID anormallikleri) ve anormal iş parçacığı sayıları.

### Kümülatif Modül Etkisi
* **`svcscan`**: Toplam SHAP = **5.3669** (En belirleyici ana modül)
* **`handles`**: Toplam SHAP = **3.0224**
* **`pslist`**: Toplam SHAP = **0.9125**
* **`ldrmodules`**: Toplam SHAP = **0.4722**
* **`malfind`**: Toplam SHAP = **0.3153**

---

## 📜 Lisans ve Atıf
* Veri Seti: Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB).
* Proje kodu akademik ve araştırma amaçlı geliştirilmiştir.


