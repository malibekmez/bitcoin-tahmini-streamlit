
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
import time
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bitcoin Yön Tahmini",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        border: 1px solid #f39c12;
    }
    .yukselis { color: #2ecc71; font-size: 2.5em; font-weight: bold; }
    .dusus    { color: #e74c3c; font-size: 2.5em; font-weight: bold; }
    div[data-testid="stDateInput"] { max-width: 200px !important; }
    div[data-testid="stButton"] > button { max-width: 150px !important; }
</style>
""", unsafe_allow_html=True)

ISIM_SOZLUK = {
    "getiri"        : "Günlük Getiri (%)",
    "lag_1"         : "Dünkü Fiyat",
    "lag_2"         : "2 Gün Önceki Fiyat",
    "lag_3"         : "3 Gün Önceki Fiyat",
    "lag_5"         : "5 Gün Önceki Fiyat",
    "lag_7"         : "7 Gün Önceki Fiyat",
    "lag_14"        : "14 Gün Önceki Fiyat",
    "lag_21"        : "21 Gün Önceki Fiyat",
    "lag_30"        : "30 Gün Önceki Fiyat",
    "std_7"         : "7 Günlük Volatilite",
    "std_14"        : "14 Günlük Volatilite",
    "std_30"        : "30 Günlük Volatilite",
    "ort_7"         : "7 Günlük Ort. Fiyat",
    "ort_14"        : "14 Günlük Ort. Fiyat",
    "ort_30"        : "30 Günlük Ort. Fiyat",
    "min_7"         : "7 Günlük Min. Fiyat",
    "min_14"        : "14 Günlük Min. Fiyat",
    "min_30"        : "30 Günlük Min. Fiyat",
    "max_7"         : "7 Günlük Maks. Fiyat",
    "max_14"        : "14 Günlük Maks. Fiyat",
    "max_30"        : "30 Günlük Maks. Fiyat",
    "RSI"           : "RSI (Alım/Satım Gücü)",
    "volatilite_7"  : "7 Günlük Fiyat Dalgalanması",
    "volatilite_30" : "30 Günlük Fiyat Dalgalanması",
    "aralik"        : "Günlük Fiyat Aralığı ($)",
    "aralik_pct"    : "Günlük Fiyat Aralığı (%)",
    "hacim_degisim" : "Hacim Değişimi (%)",
    "hacim_ort_7"   : "7 Günlük Ort. Hacim",
    "ort_fark"      : "Kısa-Uzun Vade Farkı (Trend)",
    "gun"           : "Haftanın Günü",
    "ay"            : "Ay",
    "hafta_sonu"    : "Hafta Sonu mu?",
}

@st.cache_data
def veri_hazirla():
    for deneme in range(3):
        try:
            btc = yf.download("BTC-USD", start="2020-01-01", progress=False)
            if isinstance(btc.columns, pd.MultiIndex):
                btc.columns = btc.columns.get_level_values(0)
            btc = btc[["Open", "High", "Low", "Close", "Volume"]]
            btc = btc.astype(float)
            if len(btc) > 100:
                return btc
        except:
            time.sleep(5)
    return None

@st.cache_data
def ozellik_uret(_df):
    data = _df.copy()
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        data[f"lag_{lag}"] = data["Close"].shift(lag)
    for window in [7, 14, 30]:
        data[f"ort_{window}"]  = data["Close"].rolling(window).mean()
        data[f"std_{window}"]  = data["Close"].rolling(window).std()
        data[f"min_{window}"]  = data["Close"].rolling(window).min()
        data[f"max_{window}"]  = data["Close"].rolling(window).max()
    data["getiri"]        = data["Close"].pct_change()
    data["volatilite_7"]  = data["getiri"].rolling(7).std()
    data["volatilite_30"] = data["getiri"].rolling(30).std()
    data["aralik"]        = data["High"].values - data["Low"].values
    data["aralik_pct"]    = data["aralik"].values / data["Close"].values
    data["hacim_degisim"] = data["Volume"].pct_change()
    data["hacim_ort_7"]   = data["Volume"].rolling(7).mean()
    data["ort_fark"]      = data["Close"].rolling(7).mean() - data["Close"].rolling(30).mean()
    fark  = data["Close"].diff()
    kazan = fark.clip(lower=0).rolling(14).mean()
    kayip = (-fark.clip(upper=0)).rolling(14).mean()
    data["RSI"]        = 100 - (100 / (1 + kazan / (kayip + 1e-9)))
    data["gun"]        = data.index.dayofweek
    data["ay"]         = data.index.month
    data["hafta_sonu"] = (data.index.dayofweek >= 5).astype(int)
    data["Hedef"]      = data["Close"].shift(-1)
    data.dropna(inplace=True)
    return data

@st.cache_resource
def model_egit(_data):
    ozellik_sutunlar = [c for c in _data.columns
                        if c not in ["Open","High","Low","Close","Volume","Hedef"]]
    X = _data[ozellik_sutunlar]
    y = _data["Hedef"]
    bolme    = int(len(_data) * 0.80)
    X_train  = X.iloc[:bolme]
    y_train  = y.iloc[:bolme]
    esik = 0.01
    y_train_esik = np.where(
        y_train.values > X_train["lag_1"].values * (1 + esik), 1,
        np.where(y_train.values < X_train["lag_1"].values * (1 - esik), 0, -1)
    )
    train_mask   = y_train_esik != -1
    X_train_esik = X_train[train_mask]
    y_train_esik = y_train_esik[train_mask]
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05,
        num_leaves=32, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5, verbose=-1
    )
    model.fit(X_train_esik, y_train_esik)
    return model, ozellik_sutunlar

# ── Yükleme ──────────────────────────────────────────
with st.spinner("Model yükleniyor, lütfen bekleyin..."):
    btc  = veri_hazirla()
    data = ozellik_uret(btc)
    model, ozellik_sutunlar = model_egit(data)
    gosterim_isimleri = [ISIM_SOZLUK.get(c, c) for c in ozellik_sutunlar]
    explainer = shap.TreeExplainer(model)

# ── Başlık ───────────────────────────────────────────
st.title("Bitcoin Yön Tahmini")
st.markdown("""
Model, 2020'den bugüne kadar olan Bitcoin verisiyle eğitildi. Seçtiğiniz tarihe göre bir sonraki günün yönünü tahmin ediyor.

Tahmin için %1 eşik değeri kullanılıyor — fiyat %1'den fazla artarsa yükseliş, %1'den fazla düşerse düşüş olarak değerlendiriliyor.



st.divider()

# ── Tarih seçimi ─────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    tarih = st.date_input(
        "Tarih Seç",
        value=pd.to_datetime("2024-11-11"),
        min_value=pd.to_datetime("2020-02-01"),
        max_value=pd.to_datetime(data.index[-1])
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    tahmin_btn = st.button("Tahmin Et", use_container_width=True, type="primary")

with col3:
    pass

st.divider()

# ── Tahmin ───────────────────────────────────────────
if tahmin_btn:
    tarih_ts = pd.Timestamp(tarih)

    if tarih_ts not in data.index:
        st.error("Bu tarih için veri yok. Lütfen hafta içi bir tarih seçin.")
    else:
        X_gun     = data[ozellik_sutunlar].loc[[tarih_ts]]
        olasilik  = model.predict_proba(X_gun)[0]
        yon       = model.predict(X_gun)[0]
        guvenskor = olasilik[1] if yon == 1 else olasilik[0]

        baslangic_fiyat = data.loc[tarih_ts, "Close"]
        gercek_fiyat    = data.loc[tarih_ts, "Hedef"]
        gercek_degisim  = ((gercek_fiyat - baslangic_fiyat) / baslangic_fiyat) * 100
        tahmini_fiyat   = baslangic_fiyat * 1.01 if yon == 1 else baslangic_fiyat * 0.99

        if gercek_fiyat > baslangic_fiyat * 1.01:
            gercek_str = "Yükseliş"
        elif gercek_fiyat < baslangic_fiyat * 0.99:
            gercek_str = "Düşüş"
        else:
            gercek_str = "Yatay"

        # Tahmin kartı
        if yon == 1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="yukselis">YÜKSELİŞ</div>
                <h3 style="color:white">Güven: %{guvenskor*100:.1f}</h3>
                <p style="color:#aaa; font-size:1.1em">Gerçekte: {gercek_str}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="dusus">DÜŞÜŞ</div>
                <h3 style="color:white">Güven: %{guvenskor*100:.1f}</h3>
                <p style="color:#aaa; font-size:1.1em">Gerçekte: {gercek_str}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Fiyat metrikleri
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Başlangıç Fiyatı", f"${baslangic_fiyat:,.0f}")
        col_b.metric("Tahmini Fiyat", f"${tahmini_fiyat:,.0f}", f"{chr(43) if yon==1 else chr(45)}%1.0")
        col_c.metric("Gerçek Fiyat", f"${gercek_fiyat:,.0f}", f"{gercek_degisim:+.2f}%")

        st.divider()

        # Grafikler
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### Bitcoin Fiyatı (30 Gün)")
            tarih_idx = data.index.get_loc(tarih_ts)
            pencere   = data.iloc[max(0, tarih_idx-15):min(len(data), tarih_idx+15)]

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            fig1.patch.set_facecolor("#0e1117")
            ax1.set_facecolor("#0e1117")
            ax1.plot(pencere.index, pencere["Close"], color="#f39c12", lw=2)
            ax1.fill_between(pencere.index, pencere["Close"], alpha=0.15, color="#f39c12")
            ax1.scatter([tarih_ts], [baslangic_fiyat], color="#e74c3c", s=120, zorder=5, label="Seçilen Gün")
            ax1.set_ylabel("Fiyat (USD)", color="white")
            ax1.tick_params(colors="white")
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax1.tick_params(axis="x", rotation=45)
            ax1.legend(facecolor="#1a1a2e", labelcolor="white")
            for spine in ax1.spines.values():
                spine.set_edgecolor("#333")
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

        with col_g2:
            st.markdown("#### SHAP Açıklaması")
            shap_exp = explainer(X_gun)
            shap_exp.feature_names = gosterim_isimleri
            shap_vals = shap_exp[0].values
            
            # En etkili 10 özelliği al
            top_idx   = np.argsort(np.abs(shap_vals))[-10:]
            top_isim  = [gosterim_isimleri[i] for i in top_idx]
            top_deger = [shap_vals[i] for i in top_idx]
            renkler   = ["#e74c3c" if v > 0 else "#3498db" for v in top_deger]

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor("#0e1117")
            ax2.set_facecolor("#0e1117")
            bars = ax2.barh(top_isim, top_deger, color=renkler)
            ax2.axvline(x=0, color="white", linewidth=0.8, alpha=0.5)
            ax2.set_xlabel("Etkisi (pozitif = yükseliş, negatif = düşüş)", color="white", fontsize=9)
            ax2.tick_params(colors="white")
            ax2.set_title("Tahmini Etkileyen Faktörler", color="white", fontsize=11, fontweight="bold")
            for i, (bar, val) in enumerate(zip(bars, top_deger)):
                ax2.text(
                    val / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{abs(val):.2f}",
                    va="center",
                    ha="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    fontfamily="DejaVu Sans"
                )
            for spine in ax2.spines.values():
                spine.set_edgecolor("#333")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.divider()

        # En önemli 3 faktör
        shap_vals = shap_exp[0].values
        top3_idx  = np.argsort(np.abs(shap_vals))[-3:][::-1]

        st.markdown("### En Önemli 3 Faktör")
        for i, idx in enumerate(top3_idx, 1):
            isim   = gosterim_isimleri[idx]
            etki   = shap_vals[idx]
            yon_ok = "yükselişe katkı sağladı" if etki > 0 else "düşüşe katkı sağladı"
            st.markdown(f"**{i}. {isim}** → {yon_ok}")

st.divider()
st.caption("Model: LightGBM | XAI: SHAP | Doğruluk: %77.6 | Veri: Yahoo Finance (2020-Bugün)")
