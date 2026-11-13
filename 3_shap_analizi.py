"""
3_shap_analizi.py
=================
Bellek Dökümleri Üzerinden Makine Öğrenmesi ile Gizlenmiş Zararlı Yazılım Tespiti
Model Açıklanabilirliği ve SHAP (TreeExplainer) Analiz Betiği

Ekip: Selim Şahin & Ahmet Çöl
Analiz: TreeExplainer, Beeswarm Summary Grafiği, Bar Grafiği, Bellek Modülü Dağılımı
"""

import os
import sys
import json
import tempfile
import joblib
import pandas as pd
import numpy as np

# Matplotlib/sistem geçici dizin ayarı (shap importundan önce tanımlanmalı)
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import shap

def main():
    print("=" * 70)
    print("ADIM 3: MODEL AÇIKLANABİLİRLİĞİ VE SHAP (TreeExplainer) ANALİZİ")
    print("=" * 70)

    model_path = os.path.join("outputs", "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}. Lütfen önce 2_model_egitimi.py çalıştırın.")

    info_path = os.path.join("outputs", "best_model_info.json")
    model_name = "Best Model"
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            model_name = info.get("best_model_name", "Best Model")

    print(f"[*] Eğitilmiş model yükleniyor: {model_path} ({model_name})")
    model = joblib.load(model_path)

    test_path = os.path.join("data", "X_test.csv")
    print(f"[*] Test verisi yükleniyor: {test_path}")
    X_test = pd.read_csv(test_path)

    # Temsili ve hızlı bir analiz için örneklem seçimi (1,500 örnek)
    sample_size = min(1500, len(X_test))
    print(f"[*] SHAP hesaplaması için {sample_size:,} adet rastgele test örneği seçiliyor...")
    sample_df = X_test.sample(n=sample_size, random_state=42)

    print("[*] TreeExplainer oluşturuluyor ve SHAP değerleri hesaplanıyor...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_df)

    # İkili sınıflandırma çıktı boyut kontrolü
    if isinstance(shap_values, list):
        # 1. sınıf (Malware) SHAP değerleri
        shap_vals_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals_to_plot = shap_values[:, :, 1]
    else:
        shap_vals_to_plot = shap_values

    print("[+] SHAP değerleri başarıyla hesaplandı.")

    # 1. Bellek Özelliklerinin Ortalama |SHAP| Değerleri Sıralaması
    mean_abs_shap = np.abs(shap_vals_to_plot).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        "Öznitelik (Feature)": sample_df.columns,
        "Ortalama_|SHAP|": mean_abs_shap
    }).sort_values(by="Ortalama_|SHAP|", ascending=False).reset_index(drop=True)

    print("\n" + "-" * 70)
    print("EN ETKİLİ İLK 15 BELLEK DÖKÜMÜ PARAMETRESİ (SHAP Importance):")
    print("-" * 70)
    for idx, row in feature_importance_df.head(15).iterrows():
        print(f" {idx+1:>2}. {row['Öznitelik (Feature)']:<35} : {row['Ortalama_|SHAP|']:.4f}")
    print("-" * 70)

    # 2. Volatility Bellek Modüllerine Göre Kümülatif Etki Analizi
    # (svcscan, handles, pslist, ldrmodules, malfind, dlllist, psxview, callbacks, modules vb.)
    def extract_module(feat_name):
        return feat_name.split(".")[0] if "." in feat_name else "other"

    feature_importance_df["Modul"] = feature_importance_df["Öznitelik (Feature)"].apply(extract_module)
    module_summary = feature_importance_df.groupby("Modul")["Ortalama_|SHAP|"].agg(["sum", "count"]).rename(
        columns={"sum": "Toplam_Etki", "count": "Öznitelik_Sayısı"}
    ).sort_values(by="Toplam_Etki", ascending=False)

    print("\n" + "-" * 70)
    print("VOLATILITY BELLEK MODÜLLERİ KÜMÜLATİF ETKİ DAĞILIMI:")
    print("-" * 70)
    for mod_name, row in module_summary.iterrows():
        print(f" - {mod_name:<15}: Toplam SHAP = {row['Toplam_Etki']:>7.4f} ({int(row['Öznitelik_Sayısı']):>2} öznitelik)")
    print("-" * 70 + "\n")

    # 3. Grafikleri Oluşturma ve 300 DPI Kaydetme
    os.makedirs("outputs", exist_ok=True)

    # A) SHAP Summary (Beeswarm) Grafiği
    print("[*] outputs/shap_summary.png oluşturuluyor (300 DPI)...")
    plt.figure(figsize=(13, 9), dpi=300)
    shap.summary_plot(
        shap_vals_to_plot,
        sample_df,
        max_display=20,
        show=False
    )
    plt.title(
        f"CIC-MalMem-2022 Gizlenmiş Zararlı Yazılım Tespiti\nSHAP Summary Plot (Beeswarm) - Model: {model_name}",
        fontsize=13, fontweight="bold", pad=20
    )
    plt.xlabel("SHAP Değeri (Model Çıktısına Etkisi: Pozitif = Zararlı / Malware)", fontsize=11)
    summary_path = os.path.join("outputs", "shap_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Beeswarm grafiği kaydedildi: {summary_path}")

    # B) SHAP Bar Grafiği (Global Özellik Önemi)
    print("[*] outputs/shap_bar.png oluşturuluyor (300 DPI)...")
    plt.figure(figsize=(13, 9), dpi=300)
    shap.summary_plot(
        shap_vals_to_plot,
        sample_df,
        plot_type="bar",
        max_display=20,
        show=False
    )
    plt.title(
        f"CIC-MalMem-2022 En Belirleyici 20 Bellek Dökümü Özniteliği\nOrtalama (|SHAP|) - Model: {model_name}",
        fontsize=13, fontweight="bold", pad=20
    )
    plt.xlabel("Ortalama |SHAP Değeri| (Öznitelik Önemi)", fontsize=11)
    bar_path = os.path.join("outputs", "shap_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Bar grafiği kaydedildi: {bar_path}")

    # 4. SHAP Sonuçlarını CSV olarak da kaydet
    shap_csv_path = os.path.join("outputs", "shap_feature_importance.csv")
    feature_importance_df.to_csv(shap_csv_path, index=False, encoding="utf-8-sig")
    print(f"[+] Detaylı öznitelik önem tablosu kaydedildi: {shap_csv_path}")

    print("\n[+] SHAP analizi başarıyla tamamlandı!")
    print("=" * 70)

if __name__ == "__main__":
    main()

