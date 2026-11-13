"""
2_model_egitimi.py
==================
Bellek Dökümleri Üzerinden Makine Öğrenmesi ile Gizlenmiş Zararlı Yazılım Tespiti
Model Eğitimi, Değerlendirme ve Karşılaştırma Betiği

Ekip: Selim Şahin & Ahmet Çöl
Modeller: XGBoost, LightGBM
Metrikler: Accuracy, Precision, Recall, F1-Score, ROC-AUC
"""

import os
import time
import json
import tempfile
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

# Matplotlib/sistem geçici dizin ayarı
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

def evaluate_binary_model(name, model, X_train, y_train, X_test, y_test):
    """İkili sınıflandırma modeli eğitimi ve metrik hesabı."""
    print(f"[*] {name} (İkili Sınıflandırma) eğitiliyor...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Görev": "İkili Sınıflandırma (Class)",
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Egitim_Suresi_sn": train_time,
        "cm": cm,
        "model_obj": model
    }
    return metrics

def evaluate_multiclass_model(name, model, X_train, y_train, X_test, y_test):
    """Çok sınıflı sınıflandırma modeli eğitimi ve metrik hesabı."""
    print(f"[*] {name} (Çok Sınıflı) eğitiliyor...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Görev": "Çok Sınıflı (Category)",
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Egitim_Suresi_sn": train_time,
        "cm": cm,
        "model_obj": model
    }
    return metrics

def print_results_table(results_df):
    """Terminal için şık karşılaştırma tablosu yazdırır."""
    header = f"{'Görev':<28} | {'Model':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Süre (sn)':<9}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("MODEL PERFORMANS KARŞILAŞTIRMA TABLOSU")
    print(sep)
    print(header)
    print(sep)
    for _, row in results_df.iterrows():
        print(f"{row['Görev']:<28} | {row['Model']:<12} | {row['Accuracy']:<10.4f} | {row['Precision']:<10.4f} | {row['Recall']:<10.4f} | {row['F1-Score']:<10.4f} | {row['ROC-AUC']:<10.4f} | {row['Egitim_Suresi_sn']:<9.2f}")
    print(sep + "\n")

def plot_confusion_matrices(bin_xgb, bin_lgb, multi_xgb, multi_lgb, class_labels, cat_labels, save_path):
    """Tüm modellerin karmaşıklık matrislerini 2x2 300 DPI grafik olarak kaydeder."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)

    # 1. XGBoost İkili
    sns.heatmap(
        bin_xgb["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 0],
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True
    )
    axes[0, 0].set_title(
        f"XGBoost (İkili)\nAcc: {bin_xgb['Accuracy']:.4f} | F1: {bin_xgb['F1-Score']:.4f} | AUC: {bin_xgb['ROC-AUC']:.4f}",
        fontsize=12, fontweight="bold", pad=10
    )
    axes[0, 0].set_xlabel("Tahmin Edilen Sınıf", fontsize=10)
    axes[0, 0].set_ylabel("Gerçek Sınıf", fontsize=10)

    # 2. LightGBM İkili
    sns.heatmap(
        bin_lgb["cm"],
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes[0, 1],
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True
    )
    axes[0, 1].set_title(
        f"LightGBM (İkili)\nAcc: {bin_lgb['Accuracy']:.4f} | F1: {bin_lgb['F1-Score']:.4f} | AUC: {bin_lgb['ROC-AUC']:.4f}",
        fontsize=12, fontweight="bold", pad=10
    )
    axes[0, 1].set_xlabel("Tahmin Edilen Sınıf", fontsize=10)
    axes[0, 1].set_ylabel("Gerçek Sınıf", fontsize=10)

    # 3. XGBoost Çok Sınıflı
    sns.heatmap(
        multi_xgb["cm"],
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=axes[1, 0],
        xticklabels=cat_labels,
        yticklabels=cat_labels,
        cbar=True
    )
    axes[1, 0].set_title(
        f"XGBoost (Çok Sınıflı: Kategori)\nAcc: {multi_xgb['Accuracy']:.4f} | F1 (w): {multi_xgb['F1-Score']:.4f} | AUC: {multi_xgb['ROC-AUC']:.4f}",
        fontsize=12, fontweight="bold", pad=10
    )
    axes[1, 0].set_xlabel("Tahmin Edilen Sınıf", fontsize=10)
    axes[1, 0].set_ylabel("Gerçek Sınıf", fontsize=10)

    # 4. LightGBM Çok Sınıflı
    sns.heatmap(
        multi_lgb["cm"],
        annot=True,
        fmt="d",
        cmap="Purples",
        ax=axes[1, 1],
        xticklabels=cat_labels,
        yticklabels=cat_labels,
        cbar=True
    )
    axes[1, 1].set_title(
        f"LightGBM (Çok Sınıflı: Kategori)\nAcc: {multi_lgb['Accuracy']:.4f} | F1 (w): {multi_lgb['F1-Score']:.4f} | AUC: {multi_lgb['ROC-AUC']:.4f}",
        fontsize=12, fontweight="bold", pad=10
    )
    axes[1, 1].set_xlabel("Tahmin Edilen Sınıf", fontsize=10)
    axes[1, 1].set_ylabel("Gerçek Sınıf", fontsize=10)

    fig.suptitle(
        "CIC-MalMem-2022 Bellek Dökümü Modelleri - Karmaşıklık Matrisleri (Confusion Matrices)",
        fontsize=15, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Karmaşıklık matrisleri kaydedildi: {save_path} (300 DPI)")

def main():
    print("=" * 70)
    print("ADIM 2: MODEL EĞİTİMİ VE PERFORMANS KARŞILAŞTIRMASI")
    print("=" * 70)

    os.makedirs("outputs", exist_ok=True)

    print("[*] Ön işlenmiş veriler 'data/' dizininden okunuyor...")
    X_train = pd.read_csv(os.path.join("data", "X_train.csv"))
    X_test = pd.read_csv(os.path.join("data", "X_test.csv"))

    y_train_class = pd.read_csv(os.path.join("data", "y_train_class.csv")).squeeze()
    y_test_class = pd.read_csv(os.path.join("data", "y_test_class.csv")).squeeze()

    y_train_cat = pd.read_csv(os.path.join("data", "y_train_category.csv")).squeeze()
    y_test_cat = pd.read_csv(os.path.join("data", "y_test_category.csv")).squeeze()

    metadata_path = os.path.join("data", "preprocessing_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    class_labels = ["Benign", "Malware"]
    cat_labels = ["Benign", "Ransomware", "Spyware", "Trojan"]

    print(f"[*] Özellik sayısı: {X_train.shape[1]}")
    print(f"[*] Eğitim örnek sayısı: {X_train.shape[0]:,}, Test örnek sayısı: {X_test.shape[0]:,}")

    # Model Tanımları
    xgb_bin_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1
    )

    lgb_bin_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )

    xgb_multi_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1
    )

    lgb_multi_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )

    # İkili Sınıflandırma Eğitimi
    res_xgb_bin = evaluate_binary_model("XGBoost", xgb_bin_model, X_train, y_train_class, X_test, y_test_class)
    res_lgb_bin = evaluate_binary_model("LightGBM", lgb_bin_model, X_train, y_train_class, X_test, y_test_class)

    # Çok Sınıflı Sınıflandırma Eğitimi
    res_xgb_multi = evaluate_multiclass_model("XGBoost", xgb_multi_model, X_train, y_train_cat, X_test, y_test_cat)
    res_lgb_multi = evaluate_multiclass_model("LightGBM", lgb_multi_model, X_train, y_train_cat, X_test, y_test_cat)

    all_results = [res_xgb_bin, res_lgb_bin, res_xgb_multi, res_lgb_multi]

    # Tablo ve CSV Hazırlama
    summary_data = []
    for r in all_results:
        summary_data.append({
            "Görev": r["Görev"],
            "Model": r["Model"],
            "Accuracy": round(r["Accuracy"], 5),
            "Precision": round(r["Precision"], 5),
            "Recall": round(r["Recall"], 5),
            "F1-Score": round(r["F1-Score"], 5),
            "ROC-AUC": round(r["ROC-AUC"], 5),
            "Egitim_Suresi_sn": round(r["Egitim_Suresi_sn"], 2)
        })

    results_df = pd.DataFrame(summary_data)
    print_results_table(results_df)

    csv_path = os.path.join("outputs", "model_karsilastirma.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[+] Karşılaştırma sonuçları kaydedildi: {csv_path}")

    # Karmaşıklık Matrisleri Grafiği
    cm_path = os.path.join("outputs", "confusion_matrix.png")
    plot_confusion_matrices(
        res_xgb_bin, res_lgb_bin,
        res_xgb_multi, res_lgb_multi,
        class_labels, cat_labels,
        cm_path
    )

    # En başarılı İkili Sınıflandırma Modelini SHAP için kaydetme
    # (F1-Score ve ROC-AUC'a göre karşılaştır)
    if res_xgb_bin["F1-Score"] >= res_lgb_bin["F1-Score"]:
        best_name = "XGBoost"
        best_model = res_xgb_bin["model_obj"]
        best_metrics = res_xgb_bin
    else:
        best_name = "LightGBM"
        best_model = res_lgb_bin["model_obj"]
        best_metrics = res_lgb_bin

    best_model_path = os.path.join("outputs", "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"[+] En başarılı model ({best_name}) kaydedildi: {best_model_path}")

    best_info = {
        "best_model_name": best_name,
        "task": "Binary Classification (Class: Benign vs Malware)",
        "accuracy": best_metrics["Accuracy"],
        "f1_score": best_metrics["F1-Score"],
        "roc_auc": best_metrics["ROC-AUC"],
        "feature_names": list(X_train.columns)
    }
    with open(os.path.join("outputs", "best_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=4, ensure_ascii=False)

    print("[+] Model eğitimi ve değerlendirme başarıyla tamamlandı!")
    print("=" * 70)

if __name__ == "__main__":
    main()

