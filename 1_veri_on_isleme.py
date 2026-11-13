"""
1_veri_on_isleme.py
===================
Bellek Dökümleri Üzerinden Makine Öğrenmesi ile Gizlenmiş Zararlı Yazılım Tespiti
Veri Ön İşleme ve Hazırlık Betiği

Ekip: Selim Şahin & Ahmet Çöl
Veri Seti: CIC-MalMem-2022 (Obfuscated-MalMem2022.csv)
"""

import os
import json
import tempfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Matplotlib/sistem geçici dizin ayarı
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

def main():
    print("=" * 70)
    print("ADIM 1: VERİ ÖN İŞLEME VE HAZIRLIK (CIC-MalMem-2022)")
    print("=" * 70)

    raw_path = os.path.join("data", "Obfuscated-MalMem2022.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Veri seti bulunamadı: {raw_path}")

    print(f"[*] Ham veri seti yükleniyor: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_rows, initial_cols = df.shape
    print(f"[*] Yüklenen veri boyutu: {initial_rows:,} satır, {initial_cols} sütun")

    # 1. Eksik Değer Kontrolü
    missing_vals = df.isnull().sum().sum()
    print(f"[*] Toplam eksik (NaN/Null) değer sayısı: {missing_vals}")
    if missing_vals > 0:
        df = df.dropna()
        print(f"[!] Eksik değerler temizlendi. Yeni satır sayısı: {len(df):,}")
    else:
        print("[+] Eksik veri bulunmamaktadır.")

    # 2. Tekrarlayan Kayıtların (Duplicates) Temizlenmesi
    duplicate_count = df.duplicated().sum()
    print(f"[*] Tekrarlayan kayıt sayısı: {duplicate_count:,}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[+] Tekrarlayan kayıtlar temizlendi. Kalan satır sayısı: {len(df):,}")
    else:
        print("[+] Tekrarlayan kayıt bulunmamaktadır.")

    # 3. Hedef Değişkenlerin (Class ve Category) Hazırlanması
    print("\n[*] Hedef değişkenler ayrıştırılıyor...")

    # İkili Sınıflandırma: Class (Benign: 0, Malware: 1)
    class_mapping = {"Benign": 0, "Malware": 1}
    df["Class_Encoded"] = df["Class"].map(class_mapping)

    # Çok Sınıflı Sınıflandırma: Category (Benign, Ransomware, Spyware, Trojan)
    def parse_macro_category(val):
        val_str = str(val).strip()
        if "-" in val_str:
            return val_str.split("-")[0]
        return val_str

    df["Category_Macro"] = df["Category"].apply(parse_macro_category)
    category_mapping = {
        "Benign": 0,
        "Ransomware": 1,
        "Spyware": 2,
        "Trojan": 3
    }
    df["Category_Encoded"] = df["Category_Macro"].map(category_mapping)

    print("\n--- Sınıf Dağılımları ---")
    print("1. İkili Sınıf (Class) Dağılımı:")
    for cls_name, count in df["Class"].value_counts().items():
        ratio = (count / len(df)) * 100
        print(f"   - {cls_name:<10}: {count:>6,} örnek (%{ratio:.2f})")

    print("\n2. Çoklu Sınıf (Category) Dağılımı:")
    for cat_name, count in df["Category_Macro"].value_counts().items():
        ratio = (count / len(df)) * 100
        print(f"   - {cat_name:<10}: {count:>6,} örnek (%{ratio:.2f})")

    # 4. Özellik (Features) ve Hedef Matrislerinin Ayrılması
    feature_cols = [c for c in df.columns if c not in [
        "Class", "Category", "Class_Encoded", "Category_Macro", "Category_Encoded"
    ]]
    print(f"\n[*] Bellek dökümü öznitelik (feature) sayısı: {len(feature_cols)}")

    X = df[feature_cols]
    y_class = df["Class_Encoded"]
    y_cat = df["Category_Encoded"]

    # 5. %80 Eğitim / %20 Test Stratified Ayrımı
    # y_cat üzerinden stratify ederek hem ikili hem çoklu sınıfların oranlarını tam koruyoruz
    print("[*] Veri seti %80 Eğitim ve %20 Test olarak stratified bölünüyor...")
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y_cat
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)

    y_train_class = y_class.iloc[train_idx].reset_index(drop=True)
    y_test_class = y_class.iloc[test_idx].reset_index(drop=True)

    y_train_cat = y_cat.iloc[train_idx].reset_index(drop=True)
    y_test_cat = y_cat.iloc[test_idx].reset_index(drop=True)

    print(f"   - Eğitim Seti Boyutu: {X_train.shape[0]:,} örnek ({X_train.shape[1]} öznitelik)")
    print(f"   - Test Seti Boyutu  : {X_test.shape[0]:,} örnek ({X_test.shape[1]} öznitelik)")

    # 6. İşlenmiş Verilerin Kaydedilmesi
    print("\n[*] İşlenmiş veri setleri 'data/' dizinine kaydediliyor...")
    os.makedirs("data", exist_ok=True)

    X_train.to_csv(os.path.join("data", "X_train.csv"), index=False)
    X_test.to_csv(os.path.join("data", "X_test.csv"), index=False)

    y_train_class.to_csv(os.path.join("data", "y_train_class.csv"), index=False, header=True)
    y_test_class.to_csv(os.path.join("data", "y_test_class.csv"), index=False, header=True)

    y_train_cat.to_csv(os.path.join("data", "y_train_category.csv"), index=False, header=True)
    y_test_cat.to_csv(os.path.join("data", "y_test_category.csv"), index=False, header=True)

    metadata = {
        "initial_rows": initial_rows,
        "initial_cols": initial_cols,
        "duplicates_removed": int(duplicate_count),
        "clean_rows": int(len(df)),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "class_mapping": class_mapping,
        "category_mapping": category_mapping,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": 42
    }

    with open(os.path.join("data", "preprocessing_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("[+] Veri ön işleme başarıyla tamamlandı!")
    print("=" * 70)

if __name__ == "__main__":
    main()

