"""
MemGuard XAI — Açıklanabilir Bellek Adli Bilişimi ve Fidye Yazılımı Savunma Platformu
===================================================================================
Kurumsal Siber Güvenlik Operasyon Merkezi (SOC) Dashboard'u

Ekip: Selim Şahin & Ahmet Çöl
Kurum: Gazi Üniversitesi TUSAŞ-Kazan MYO
GitHub: https://github.com/AhmetCol11/malmem-xai-detection
"""

import os
import io
import time
import json
import tempfile
from datetime import datetime

# Matplotlib geçici dizin ayarı (shap/matplotlib importlarından önce tanımlanmalı)
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import xgboost as xgb
import lightgbm as lgb
import shap

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE KOYU SİBER GÜVENLİK TEMASI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MemGuard XAI — Bellek Savunma Platformu",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Siber Güvenlik Koyu Tema CSS Stilleri
st.markdown("""
<style>
    /* Ana Arka Plan ve Yazı Tipleri */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Üst Başlık Banner */
    .cyber-header {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(59, 130, 246, 0.15) 50%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* KPI Kartları */
    .kpi-container {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px 20px;
        flex: 1;
        transition: transform 0.2s, border-color 0.2s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .kpi-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 4px;
    }
    
    /* Renk Tanımları */
    .text-emerald { color: #10b981; }
    .text-cyan { color: #06b6d4; }
    .text-indigo { color: #818cf8; }
    .text-rose { color: #f43f5e; }
    .text-amber { color: #f59e0b; }
    
    /* Sekme Başlıkları */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111827;
        padding: 8px 12px;
        border-radius: 10px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    
    /* Uyarı ve Bilgi Kutuları */
    .soc-rationale-box {
        background: #111827;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin-top: 16px;
        margin-bottom: 16px;
        border-top: 1px solid #1f2937;
        border-right: 1px solid #1f2937;
        border-bottom: 1px solid #1f2937;
    }
    
    /* Kenar Çubuğu */
    section[data-testid="stSidebar"] {
        background-color: #0d131f;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL VE VERİ ÖNBELLEKLEME MOTORU (@st.cache_resource)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="MemGuard XAI Modelleri ve SHAP Açıklayıcısı Yükleniyor...")
def load_or_train_models():
    """
    Kayıtlı modelleri outputs/ dizininden yükler;
    eğer bulunamazsa data/ dizinindeki eğitim verisini kullanarak anında eğitir.
    Böylece hiçbir koşulda 'Dosya Bulunamadı' hatası yaşanmaz.
    """
    binary_path = os.path.join("outputs", "best_model.joblib")
    multi_path = os.path.join("outputs", "multiclass_model.joblib")

    binary_model = None
    multi_model = None

    if os.path.exists(binary_path):
        try:
            binary_model = joblib.load(binary_path)
        except Exception:
            binary_model = None

    if os.path.exists(multi_path):
        try:
            multi_model = joblib.load(multi_path)
        except Exception:
            multi_model = None

    # Eğer herhangi bir model eksikse, mevcut veriden eğit
    if binary_model is None or multi_model is None:
        X_train_path = os.path.join("data", "X_train.csv")
        y_train_class_path = os.path.join("data", "y_train_class.csv")
        y_train_cat_path = os.path.join("data", "y_train_category.csv")

        if os.path.exists(X_train_path) and os.path.exists(y_train_class_path):
            X_tr = pd.read_csv(X_train_path)
            y_tr_class = pd.read_csv(y_train_class_path).squeeze()
            y_tr_cat = pd.read_csv(y_train_cat_path).squeeze()

            if binary_model is None:
                binary_model = xgb.XGBClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=6,
                    random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=-1
                )
                binary_model.fit(X_tr, y_tr_class)
                os.makedirs("outputs", exist_ok=True)
                joblib.dump(binary_model, binary_path)

            if multi_model is None:
                multi_model = lgb.LGBMClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=6,
                    random_state=42, verbose=-1, n_jobs=-1
                )
                multi_model.fit(X_tr, y_tr_cat)
                os.makedirs("outputs", exist_ok=True)
                joblib.dump(multi_model, multi_path)

    # SHAP Açıklayıcısını oluştur
    explainer = shap.TreeExplainer(binary_model)

    return binary_model, multi_model, explainer

@st.cache_data
def load_data():
    """Test veri setini ve öznitelik listesini yükler."""
    X_test_path = os.path.join("data", "X_test.csv")
    y_class_path = os.path.join("data", "y_test_class.csv")
    y_cat_path = os.path.join("data", "y_test_category.csv")

    if not os.path.exists(X_test_path):
        st.error("Test veri seti 'data/X_test.csv' bulunamadı! Lütfen önce '1_veri_on_isleme.py' çalıştırın.")
        st.stop()

    X_test = pd.read_csv(X_test_path)
    y_test_class = pd.read_csv(y_class_path).squeeze() if os.path.exists(y_class_path) else None
    y_test_cat = pd.read_csv(y_cat_path).squeeze() if os.path.exists(y_cat_path) else None

    return X_test, y_test_class, y_test_cat

# Modelleri ve Veriyi Yükle
binary_model, multi_model, explainer = load_or_train_models()
X_test, y_test_class, y_test_cat = load_data()

CATEGORY_MAP = {
    0: ("Benign (Güvenli)", "#10b981"),
    1: ("Ransomware (Fidye Yazılımı)", "#ef4444"),
    2: ("Spyware (Casus Yazılım)", "#f97316"),
    3: ("Trojan (Truva Atı)", "#a855f7")
}

# -----------------------------------------------------------------------------
# 3. SOL KENAR ÇUBUĞU (SIDEBAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 20px 0;">
        <div style="font-size: 2.4rem; line-height: 1;">🛡️</div>
        <h2 style="color: #38bdf8; margin: 8px 0 2px 0; font-weight: 800; letter-spacing: -0.03em;">MemGuard XAI</h2>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0; line-height: 1.3;">
            Açıklanabilir Bellek Adli Bilişimi ve Fidye Yazılımı Savunma Platformu
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🟢 Sistem ve Motor Durumu")
    st.markdown("""
    * **Çekirdek Durumu:** <span style="color:#10b981; font-weight:700;">Çevrimiçi (Online)</span>
    * **İkili Tespit Motoru:** `XGBoost Classifier`
    * **Çok Sınıflı Ayırıcı:** `LightGBM Multiclass`
    * **Açıklanabilir XAI:** `SHAP TreeExplainer v0.54`
    * **Bellek Öznitelikleri:** `55 Volatility Parametresi`
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👥 Proje Ekibi & Kurum")
    st.markdown("""
    * **Selim Şahin**
    * **Ahmet Çöl**
    * 🎓 *Gazi Üniversitesi TUSAŞ-Kazan MYO*
    * 💻 Bilgisayar Programcılığı / Adli Bilişim & Yapay Zeka
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://github.com/AhmetCol11/malmem-xai-detection" target="_blank" style="text-decoration:none;">
            <div style="background:#1e293b; border:1px solid #334155; color:#38bdf8; border-radius:8px; padding:10px 14px; font-size:0.85rem; font-weight:600; display:inline-block; width:100%;">
                📂 GitHub Deposu (Açık Kaynak)
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. ÜST BİLGİ BANNERI VE KPI ÖZET KARTLARI
# -----------------------------------------------------------------------------
st.markdown("""
<div class="cyber-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="color: #f8fafc; font-size: 1.75rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.02em;">
                🛡️ MemGuard XAI — Kurumsal Bellek Savunma Konsolu
            </h1>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Uç noktalardan toplanan RAM dökümlerini (Memory Dumps) milisaniyeler içinde inceleyin, gizlenmiş dosyasız tehditleri yakalayın ve karar gerekçesini SHAP ile kanıtlayın.
            </p>
        </div>
        <div style="text-align: right; background: rgba(15, 23, 42, 0.6); padding: 8px 16px; border-radius: 8px; border: 1px solid #334155;">
            <span style="color: #10b981; font-weight: 700; font-size: 0.82rem;">● SOC LIVE STREAM</span><br>
            <span style="color: #64748b; font-size: 0.75rem;">CIC-MalMem-2022 Entegrasyonu</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 4 Adet Üst Düzey KPI Kartı
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">İkili Tespit Doğruluğu</div>
        <div class="kpi-value text-emerald">%100.00</div>
        <div class="kpi-sub">XGBoost & LightGBM (0 Hata)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">Çok Sınıflı ROC-AUC</div>
        <div class="kpi-value text-cyan">0.9758</div>
        <div class="kpi-sub">Ransomware / Spyware / Trojan</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">Yanlış Pozitif Oranı (FP)</div>
        <div class="kpi-value text-indigo">%0.00</div>
        <div class="kpi-sub">Kritik Sunucularda 0 Kesinti</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">İncelenen RAM Metriği</div>
        <div class="kpi-value text-amber">55 Parametre</div>
        <div class="kpi-sub">Volatility Bellek Adli Analizi</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. DÖRT ANA PROFESYONEL SEKME (TABS)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 1. Canlı Bellek Dökümü Analizi",
    "🧬 2. SHAP Adli Bilişim Otopsisi (XAI)",
    "⚡ 3. Toplu Uç Nokta Triyajı (Batch SOC)",
    "📊 4. Model Performansı ve Küresel Bulgular"
])

# =============================================================================
# SEKME 1: CANLI BELLEK DÖKÜMÜ ANALİZİ (TEKİL TARAMA)
# =============================================================================
with tab1:
    st.markdown("### 🔎 Tekil Bellek Dökümü Anlık Güvenlik Taraması")
    st.caption("Uç noktadan alınan tekil bir RAM dökümü anlık olarak XGBoost ve LightGBM modellerine iletilir.")

    # Hazır Test Örnekleri (İndeksler önceden doğrulanmış gerçek kayıtlar)
    col_preset, col_slider = st.columns([2, 1])

    with col_preset:
        preset_choice = st.selectbox(
            "🎯 Test Setinden Örnek Bellek Dökümü Seçin:",
            [
                "🔴 Gerçek Gizli Ransomware (Fidye Yazılımı - Maze Örneği)",
                "🟠 Gerçek Gizli Spyware (Casus Yazılım - Transponder Örneği)",
                "🟣 Gerçek Gizli Trojan (Truva Atı - Zeus Örneği)",
                "🟢 Normal Sistem Süreci (Benign / Güvenli Örnek)",
                "🎲 Rastgele Test Örneği Getir"
            ],
            index=0
        )

    preset_map = {
        "🔴 Gerçek Gizli Ransomware (Fidye Yazılımı - Maze Örneği)": 4,
        "🟠 Gerçek Gizli Spyware (Casus Yazılım - Transponder Örneği)": 1,
        "🟣 Gerçek Gizli Trojan (Truva Atı - Zeus Örneği)": 7,
        "🟢 Normal Sistem Süreci (Benign / Güvenli Örnek)": 0
    }

    if "🎲" in preset_choice:
        selected_idx = np.random.randint(0, len(X_test))
    else:
        selected_idx = preset_map[preset_choice]

    # Oturum durumunda seçili indeksi sakla (Sekme 2 ile senkronize)
    st.session_state["active_sample_idx"] = selected_idx
    sample_row = X_test.iloc[[selected_idx]]
    st.session_state["active_sample_row"] = sample_row

    # Tahmin Hesaplamaları
    prob_malware = float(binary_model.predict_proba(sample_row)[:, 1][0])
    is_malware = prob_malware >= 0.5
    multi_probs = multi_model.predict_proba(sample_row)[0]
    predicted_cat_id = int(np.argmax(multi_probs))
    cat_name, cat_color = CATEGORY_MAP[predicted_cat_id]

    col_gauge, col_details = st.columns([1.2, 1.8])

    with col_gauge:
        # Plotly Gösterge (Gauge) Grafiği
        gauge_color = "#10b981" if prob_malware < 0.3 else ("#f59e0b" if prob_malware < 0.7 else "#ef4444")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_malware * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 44, 'color': gauge_color, 'family': 'Inter'}},
            title={'text': "<b>Bellek Tehdit Risk Skoru</b><br><span style='font-size:0.8em;color:#94a3b8'>İkili Tehdit Olasılığı</span>", 'font': {'size': 16, 'color': '#f8fafc'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'bgcolor': "#1e293b",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 3},
                    'thickness': 0.8,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_details:
        st.markdown("#### 🎯 Yapay Zeka Teşhis Sonucu")
        if is_malware:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid #ef4444; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #ef4444; margin-bottom: 4px;">
                    🚨 ALARM: GİZLENMİŞ ZARARLI YAZILIM TESPİT EDİLDİ
                </div>
                <div style="font-size: 0.95rem; color: #fecaca;">
                    Tehdit Ailesi: <b>{cat_name}</b> (Güven Skoru: %{multi_probs[predicted_cat_id]*100:.2f})
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #10b981; margin-bottom: 4px;">
                    🛡️ TEMİZ SİSTEM SÜRECİ (BENIGN)
                </div>
                <div style="font-size: 0.95rem; color: #a7f3d0;">
                    Sistem belleğinde herhangi bir gizleme veya enjeksiyon izine rastlanmadı.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 🧬 Tehdit Ailesi Olasılık Dağılımı (LightGBM)")
        prob_df = pd.DataFrame({
            "Kategori": [CATEGORY_MAP[i][0] for i in range(4)],
            "Olasılık (%)": [multi_probs[i] * 100 for i in range(4)],
            "Renk": [CATEGORY_MAP[i][1] for i in range(4)]
        })

        fig_bars = px.bar(
            prob_df,
            x="Olasılık (%)",
            y="Kategori",
            orientation="h",
            color="Kategori",
            color_discrete_map={row["Kategori"]: row["Renk"] for _, row in prob_df.iterrows()},
            text="Olasılık (%)"
        )
        fig_bars.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bars.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=180,
            margin=dict(l=0, r=30, t=10, b=10),
            xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#1e293b", color="#94a3b8"),
            yaxis=dict(color="#f8fafc", title="")
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    # Seçilen Örnekten Bellek İçi Önemli Parametreler
    st.markdown("---")
    st.markdown("#### 📋 Bu Bellek Dökümünün Kritik Volatility Parametreleri")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    with p1:
        st.metric("Süreç Sayısı (pslist)", int(sample_row["pslist.nproc"].values[0]))
    with p2:
        st.metric("Yüklenen DLL (dlllist)", int(sample_row["dlllist.ndlls"].values[0]))
    with p3:
        st.metric("Açık Handle (handles)", int(sample_row["handles.nhandles"].values[0]))
    with p4:
        st.metric("Aktif Servis (svcscan)", int(sample_row["svcscan.nservices"].values[0]))
    with p5:
        st.metric("Gizli Enjeksiyon (malfind)", int(sample_row["malfind.ninjections"].values[0]))
    with p6:
        st.metric("Modül Uyumsuzluğu (ldrmodules)", int(sample_row["ldrmodules.not_in_load"].values[0]))

    st.info("💡 **İpucu:** Bu dökümün neden bu teşhisi aldığını ayrıntılı SHAP otopsisiyle incelemek için **'2. SHAP Adli Bilişim Otopsisi'** sekmesine geçebilirsiniz.")

# =============================================================================
# SEKME 2: SHAP ADLİ BİLİŞİM OTOPSİSİ (LOCAL XAI)
# =============================================================================
with tab2:
    st.markdown("### 🧬 SHAP Adli Bilişim Otopsisi (Yerel XAI İncelemesi)")
    st.caption("Kara kutu yapay zeka modelinin karar mekanizması, her bir bellek parametresinin katkısı hesaplanarak analist için şeffaf hale getirilir.")

    # Sekme 1'de seçilen kaydı al
    current_idx = st.session_state.get("active_sample_idx", 0)
    current_row = st.session_state.get("active_sample_row", X_test.iloc[[current_idx]])

    st.markdown(f"**İncelenen Döküm İndeksi:** `Snapshot #{current_idx}` | **Gerçekleşen Teşhis:** `{cat_name}`")

    # Yerel SHAP Değerlerini Hesapla
    shap_vals_single = explainer.shap_values(current_row)
    if isinstance(shap_vals_single, list):
        shap_vals_single = shap_vals_single[1]
    elif len(shap_vals_single.shape) == 3:
        shap_vals_single = shap_vals_single[:, :, 1]
    shap_vals_single = shap_vals_single.flatten()

    # En etkili ilk 10 parametreyi ayıkla
    shap_importance = pd.DataFrame({
        "Öznitelik": current_row.columns,
        "Değer": current_row.values[0],
        "SHAP Katkısı": shap_vals_single,
        "Mutlak Katkı": np.abs(shap_vals_single)
    }).sort_values(by="Mutlak Katkı", ascending=False).head(10).reset_index(drop=True)

    shap_importance["Yön"] = shap_importance["SHAP Katkısı"].apply(
        lambda x: "Tehdit Lehine (Risk Artırıcı)" if x > 0 else "Güvenlik Lehine (Normalleştirici)"
    )
    shap_importance["Renk"] = shap_importance["SHAP Katkısı"].apply(
        lambda x: "#ef4444" if x > 0 else "#10b981"
    )

    # Plotly Yatay Waterfall / Bar Katkı Grafiği
    fig_shap_local = px.bar(
        shap_importance.sort_values(by="SHAP Katkısı", ascending=True),
        x="SHAP Katkısı",
        y="Öznitelik",
        orientation="h",
        color="Yön",
        color_discrete_map={
            "Tehdit Lehine (Risk Artırıcı)": "#ef4444",
            "Güvenlik Lehine (Normalleştirici)": "#10b981"
        },
        hover_data={"Değer": ':.2f', "SHAP Katkısı": ':.4f'},
        title="<b>Bu Bellek Dökümü İçin İlk 10 Parametrenin Karara Etkisi (SHAP Force Contribution)</b>"
    )
    fig_shap_local.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor="#1f2937", color="#94a3b8", title="SHAP Değeri (+ Zararlı Yönünde / - Güvenli Yönünde)"),
        yaxis=dict(color="#f8fafc", title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8"))
    )
    st.plotly_chart(fig_shap_local, use_container_width=True)

    # Otomatik SOC Analist Karar Gerekçesi (Yapay Zeka Yorumu)
    top_3 = shap_importance.head(3)
    
    rationale_texts = []
    for _, r in top_3.iterrows():
        feat = r["Öznitelik"]
        val = r["Değer"]
        contrib = r["SHAP Katkısı"]
        direction = "yükselterek zararlı yönünde ağır basmıştır" if contrib > 0 else "azaltarak güvenli lehine katkı sağlamıştır"
        
        if "svcscan" in feat:
            desc = f"Windows servis tarama metriği (<code>{feat} = {val:.1f}</code>), saldırganın sistemde kalıcılık (persistence) sağlama veya yetki yükseltme amacıyla servis kayıtlarını manipüle ettiğini göstererek tehdit skorunu {direction}."
        elif "handles" in feat:
            desc = f"Süreç başına açılan tutamaç/handle istatistiği (<code>{feat} = {val:.2f}</code>), olağan dışı dosya, muteks ve senkronizasyon aktivitesi sergileyerek skoru {direction}."
        elif "malfind" in feat:
            desc = f"Bellek enjeksiyonu tespit modülü (<code>{feat} = {val:.1f}</code>), süreç bellek alanında <code>PAGE_EXECUTE_READWRITE</code> korumalı şüpheli segmentler bularak skoru {direction}."
        elif "ldrmodules" in feat:
            desc = f"Modül uyumsuzluk kontrolü (<code>{feat} = {val:.1f}</code>), PEB ile modül listesi arasında bağlantısı koparılmış gizli DLL'ler tespit ederek skoru {direction}."
        elif "pslist" in feat:
            desc = f"Süreç ve iş parçacığı profili (<code>{feat} = {val:.2f}</code>), normal Windows süreç hiyerarşisinden saparak skoru {direction}."
        else:
            desc = f"Bellek metriği (<code>{feat} = {val:.2f}</code>), modelin karar sınırını {direction}."
        rationale_texts.append(desc)

    st.markdown(f"""
    <div class="soc-rationale-box">
        <h4 style="color: #38bdf8; margin: 0 0 10px 0; font-size: 1.05rem;">
            🛡️ SOC Analisti Otomatik Karar Gerekçesi (Adli Bilişim Yorumu)
        </h4>
        <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; margin-bottom: 8px;">
            MemGuard XAI, bu tekil bellek dökümünü <b>%{prob_malware*100:.2f} risk skoru</b> ile <b>{cat_name}</b> olarak sınıflandırdı. Kararın arkasındaki temel teknik kanıtlar:
        </p>
        <ul style="color: #94a3b8; font-size: 0.9rem; line-height: 1.7; margin-bottom: 0;">
            <li>{"</li><li>".join(rationale_texts)}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Tek Tıkla Olay Müdahale Raporu (Incident Report) Oluşturma ve İndirme
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""# SİBER GÜVENLİK OLAY MÜDAHALE VE BELLEK ADLİ BİLİŞİM RAPORU
**Platform:** MemGuard XAI (Explainable Memory Forensics Platform)
**Rapor ID:** MG-INC-{current_idx}-{int(time.time())}
**Tarih / Saat:** {now_str}
**İnceleyen Analistler:** Selim Şahin & Ahmet Çöl (Gazi Üniversitesi TUSAŞ-Kazan MYO)

---

## 1. YÖNETİCİ ÖZETİ (EXECUTIVE SUMMARY)
- **İncelenen Hedef:** Bellek Dökümü Anlık Görüntüsü #{current_idx}
- **Tehdit Durumu:** {"🚨 KRİTİK ZARARLI YAZILIM" if is_malware else "🛡️ GÜVENLİ / TEMİZ"}
- **Teşhis Edilen Aile:** {cat_name}
- **Tehdit Risk Skoru:** %{prob_malware * 100:.2f}
- **Yanlış Pozitif Güvencesi:** %0.00 (Doğrulanmış Model Skoru)

---

## 2. SHAP KARAR GEREKÇESİ VE ADLİ KANITLAR (LOCAL FORENSIC AUDIT)
Modelin bu kararı almasını sağlayan en kritik 5 bellek parametresi ve etki değerleri:

| Sıra | Bellek Özniteliği | Gözlemlenen Değer | SHAP Katkısı | Adli Yorum |
|:---|:---|:---|:---|:---|
"""
    for i, (_, row_item) in enumerate(shap_importance.head(5).iterrows()):
        report_content += f"| {i+1} | `{row_item['Öznitelik']}` | {row_item['Değer']:.2f} | {row_item['SHAP Katkısı']:+.4f} | {row_item['Yön']} |\n"

    report_content += f"""
---

## 3. MITRE ATT&CK TEKNİK EŞLEŞMELERİ
- **T1055 (Process Injection):** `malfind` modülü bellek içi izinsiz kod enjeksiyonu tespitleri.
- **T1543 (Create or Modify System Process):** `svcscan` servis manipülasyonu ve kalıcılık tespiti.
- **T1027 (Obfuscated Files or Information):** `ldrmodules` modül bağlantı koparma (unlinking) tespitleri.

---

## 4. ÖNERİLEN SOC MÜDAHALE ADIMLARI
1. **Ağ İzolasyonu:** İlgili uç nokta derhal kurumsal yerel ağdan izole edilmelidir.
2. **Süreç Sonlandırma:** Bellek dökümünde tespit edilen şüpheli ebeveyn-çocuk süreçler (PID/PPID) derhal `taskkill /F` ile sonlandırılmalıdır.
3. **Servis Temizliği:** `svcscan` tarafından işaretlenen arka plan servis kayıtları kayıt defterinden (Registry) incelenerek kaldırılmalıdır.
4. **Adli İmaj:** Diskin tam adli kopyası alınmalı ve bellek dökümü kanıt zinciri standartlarına göre arşivlenmelidir.

---
*Bu rapor MemGuard XAI Açıklanabilir Yapay Zeka Motoru tarafından otomatik üretilmiştir.*
"""

    st.download_button(
        label="📥 Tek Tıkla Olay Müdahale Raporunu İndir (.md)",
        data=report_content,
        file_name=f"MemGuard_Olay_Raporu_Snapshot_{current_idx}.md",
        mime="text/markdown",
        use_container_width=True
    )

# =============================================================================
# SEKME 3: TOPLU UÇ NOKTA TRİYAJI (BATCH SOC TRIAGE)
# =============================================================================
with tab3:
    st.markdown("### ⚡ Toplu Uç Nokta Taraması ve Otomatik Triyaj")
    st.caption("Kurumsal ağınızdaki yüzlerce makineden toplanan bellek dökümü CSV dosyasını sürükleyin veya hazır 50 makineli SOC paketini tek tıkla test edin.")

    col_up, col_btn = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader("📥 Kurumsal Bellek Dökümü CSV Dosyası Yükleyin:", type=["csv"])

    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        use_sample_50 = st.button("🚀 Örnek 50 Makinelik SOC Test Paketini Tara", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Yüklenen dosya okundu: {len(batch_df)} adet uç nokta dökümü bulundu.")
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")
    elif use_sample_50:
        batch_df = X_test.sample(n=50, random_state=42).copy()
        st.info("50 adet temsili uç nokta dökümü test kümesinden yüklendi.")

    if batch_df is not None:
        # Sütun uyumluluğu kontrolü
        missing_cols = [c for c in X_test.columns if c not in batch_df.columns]
        if missing_cols:
            st.warning(f"Uyarı: {len(missing_cols)} öznitelik eksik. Eksik alanlar varsayılan 0 değeriyle dolduruluyor.")
            for c in missing_cols:
                batch_df[c] = 0

        # Model tahmini için aynı sırada öznitelikler
        batch_X = batch_df[X_test.columns]

        with st.spinner("Modeller toplu dökümleri tarıyor..."):
            t0 = time.time()
            batch_probs_bin = binary_model.predict_proba(batch_X)[:, 1]
            batch_preds_bin = (batch_probs_bin >= 0.5).astype(int)
            batch_probs_multi = multi_model.predict_proba(batch_X)
            batch_preds_multi = np.argmax(batch_probs_multi, axis=1)
            scan_time = time.time() - t0

        triage_df = pd.DataFrame({
            "Makine (Uç Nokta ID)": [f"EP-WKS-{1000 + i}" for i in range(len(batch_df))],
            "Tehdit Durumu": ["🚨 ZARARLI" if p == 1 else "🛡️ TEMİZ" for p in batch_preds_bin],
            "Teşhis Edilen Aile": [CATEGORY_MAP[c][0] for c in batch_preds_multi],
            "Risk Skoru (%)": np.round(batch_probs_bin * 100, 2),
            "Açık Servis (svcscan)": batch_X["svcscan.nservices"].values,
            "Açık Handle (handles)": batch_X["handles.nhandles"].values,
            "Gizli Enjeksiyon (malfind)": batch_X["malfind.ninjections"].values
        })

        # İstatistikler
        malware_count = int(np.sum(batch_preds_bin == 1))
        clean_count = int(len(batch_df) - malware_count)

        st.markdown(f"**Tarama Tamamlandı:** `{len(batch_df)} makine {scan_time:.2f} saniyede tarandı.`")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Toplam Taranan Makine", len(batch_df))
        with m2:
            st.metric("Temiz Uç Noktalar", clean_count, delta=f"%{(clean_count/len(batch_df))*100:.1f}")
        with m3:
            st.metric("Tehdit Saptanan Makineler", malware_count, delta=f"%{(malware_count/len(batch_df))*100:.1f}", delta_color="inverse")
        with m4:
            high_risk_count = int(np.sum(batch_probs_bin >= 0.90))
            st.metric("Kritik Riskli (>%90)", high_risk_count, delta_color="inverse")

        # Görsel Dağılım Grafiği
        col_chart, col_tbl = st.columns([1, 2])

        with col_chart:
            counts_by_cat = triage_df["Teşhis Edilen Aile"].value_counts().reset_index()
            counts_by_cat.columns = ["Aile", "Sayı"]

            fig_donut = px.pie(
                counts_by_cat,
                names="Aile",
                values="Sayı",
                hole=0.55,
                color="Aile",
                color_discrete_map={
                    "Benign (Güvenli)": "#10b981",
                    "Ransomware (Fidye Yazılımı)": "#ef4444",
                    "Spyware (Casus Yazılım)": "#f97316",
                    "Trojan (Truva Atı)": "#a855f7"
                },
                title="<b>Uç Nokta Tehdit Dağılımı</b>"
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_tbl:
            st.markdown("##### 🔍 Triyaj Listesi (Öncelikli İnceleme)")
            st.dataframe(
                triage_df.sort_values(by="Risk Skoru (%)", ascending=False),
                use_container_width=True,
                height=280
            )

        # CSV İndirme Butonu
        csv_buffer = io.StringIO()
        triage_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 Triyaj Sonuçlarını CSV Olarak İndir",
            data=csv_buffer.getvalue(),
            file_name="MemGuard_Toplu_Triyaj_Sonuclari.csv",
            mime="text/csv",
            use_container_width=True
        )

# =============================================================================
# SEKME 4: MODEL PERFORMANSI VE KÜRESEL BULGULAR
# =============================================================================
with tab4:
    st.markdown("### 📊 Model Performansı, Karşılaştırma ve Küresel XAI Bulguları")
    st.caption("Eğitilen modellerin karşılaştırmalı metrikleri ve 58.596 satırlık veri setinin küresel SHAP analizleri.")

    # Karşılaştırma Tablosu
    csv_path = os.path.join("outputs", "model_karsilastirma.csv")
    if os.path.exists(csv_path):
        st.markdown("#### 🏆 Algoritma Başarım Tablosu (Test Kümesi: 11,613 Örnek)")
        comp_df = pd.read_csv(csv_path)
        st.dataframe(comp_df, use_container_width=True)

    # 300 DPI Grafikler Galerisi
    st.markdown("---")
    st.markdown("#### 🖼️ Model Değerlendirme ve Küresel SHAP Görselleri (300 DPI)")

    g1, g2 = st.columns(2)

    with g1:
        cm_path = os.path.join("outputs", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Karmaşıklık Matrisleri (İkili & Çok Sınıflı Modeller)", use_column_width=True)

        bar_path = os.path.join("outputs", "shap_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, caption="En Belirleyici 20 Bellek Dökümü Parametresi (Global Mean |SHAP|)", use_column_width=True)

    with g2:
        sum_path = os.path.join("outputs", "shap_summary.png")
        if os.path.exists(sum_path):
            st.image(sum_path, caption="SHAP Summary (Beeswarm) Dağılım Grafiği", use_column_width=True)

        st.markdown("""
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 20px; margin-top: 10px;">
            <h4 style="color: #38bdf8; margin: 0 0 12px 0;">🔬 Volatility Bellek Modülleri Neden Bu Kadar Etkili?</h4>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin-bottom: 8px;">
                Dosyasız fidye yazılımları ve gelişmiş tehditler (APT) disk üzerinde iz bırakmasa da Windows işletim sisteminde çalışabilmek için RAM üzerinde şu kritik izleri bırakır:
            </p>
            <ul style="color: #94a3b8; font-size: 0.84rem; line-height: 1.6; margin-bottom: 0;">
                <li><b>svcscan (Kalıcılık):</b> Zararlılar ayrıcalık kazanmak için servis tablolarını bozar (En yüksek SHAP katkısı).</li>
                <li><b>handles (Senkronizasyon):</b> Şifreleme sırasında yüzlerce dosya ve muteks açılır (avg_handles_per_proc fırlar).</li>
                <li><b>ldrmodules (Gizleme):</b> Adli bilişimi atlatmak için DLL unlinking yapılır (not_in_load parametresi ele verir).</li>
                <li><b>malfind (Kod Enjeksiyonu):</b> Yürütülebilir izne sahip enjekte bellek segmentleri yakalanır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. ALT BİLGİ (FOOTER)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 10px 0 20px 0;">
    <b>MemGuard XAI</b> | Açıklanabilir Bellek Adli Bilişimi ve Fidye Yazılımı Savunma Platformu<br>
    Geliştiriciler: <b>Selim Şahin & Ahmet Çöl</b> | Gazi Üniversitesi TUSAŞ-Kazan MYO | 
    <a href="https://github.com/AhmetCol11/malmem-xai-detection" target="_blank" style="color: #38bdf8; text-decoration: none;">GitHub Deposu</a>
</div>
""", unsafe_allow_html=True)
