import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
import os

# Configuration de la page Streamlit
st.set_page_config(page_title="Projet Gabriel Princet", layout="wide")
st.title("Projet Gestion de Portefeuille Gabriel PRINCET")

# DONNÉES DU PORTEFEUILLE 
portefeuille = {
    "UBI.PA": {"quantite": 170, "prix_achat": 11.944},
    "MC.PA": {"quantite": 3, "prix_achat": 604.147},
    "BNP.PA": {"quantite": 29, "prix_achat": 67.795},
    "NOV.DE": {"quantite": 25, "prix_achat": 61.204},
    "ESE.PA": {"quantite": 135, "prix_achat": 26.361},
    "ETSZ.DE": {"quantite": 144, "prix_achat": 16.813},
    "PAEEM.PA": {"quantite": 182, "prix_achat": 25.98},
}

noms_affichage = {
    "UBI.PA": "Ubisoft",
    "MC.PA": "LVMH",
    "BNP.PA": "BNP Paribas",
    "NOV.DE": "Novo Nordisk",
    "ESE.PA": "ETF S&P 500",
    "ETSZ.DE": "ETF EUROSTOXX 600",
    "PAEEM.PA": "ETF Emerging Market",
}

# --- HISTORIQUE & DIVIDENDES ---
historique_gains = {
    "EN.PA": {
        "nom": "Bouygues", 
        "quantite": 150, 
        "prix_achat": 29.23,
        "prix_vente": 39.15, 
        "pv_realisee": 536.40
    },
}

dividendes_recus = {
    "Bouygues": 90.00,
    "LVMH": 31.50, 
    "Novo Nordisk": 9.00
}

# CALCULS DES ENCAISSEMENTS
total_dividendes = sum(dividendes_recus.values())
pv_realisee_totale = sum(action["pv_realisee"] for action in historique_gains.values())
total_encaisse = total_dividendes + pv_realisee_totale  # Dividendes + Ventes

tickers = list(portefeuille.keys())
benchmark = "IWDA.AS"

# --- RÉCUPÉRATION DES DONNÉES ---
with st.spinner('Connexion aux marchés en cours...'):
    prix_actuels = yf.download(tickers, period="1d", progress=False)["Close"].iloc[-1]
    data_1y = yf.download(tickers, period="1y", progress=False)["Close"].ffill().dropna()
    data_5y = yf.download(tickers, period="5y", progress=False)["Close"].ffill().dropna()
    b_1y = yf.download(benchmark, period="1y", progress=False)["Close"].ffill().dropna()
    b_5y = yf.download(benchmark, period="5y", progress=False)["Close"].ffill().dropna()

    ret_1y = data_1y.pct_change().dropna()
    ret_5y = data_5y.pct_change().dropna()
    rb_1y = b_1y.pct_change().dropna()
    if isinstance(rb_1y, pd.DataFrame): rb_1y = rb_1y.iloc[:, 0]
    rb_5y = b_5y.pct_change().dropna()
    if isinstance(rb_5y, pd.DataFrame): rb_5y = rb_5y.iloc[:, 0]

# --- CALCULS DE BASE ---
rows = []
valeur_totale = 0
plus_value_totale = 0
investi_total = 0

for t, info in portefeuille.items():
    q = info["quantite"]
    prix = prix_actuels[t]
    valeur = q * prix
    investi = q * info["prix_achat"]
    pv = valeur - investi
    perf = pv / investi * 100
    valeur_totale += valeur
    plus_value_totale += pv
    investi_total += investi

    vol_actif_1y = ret_1y[t].std() * np.sqrt(252)
    vol_actif_5y = ret_5y[t].std() * np.sqrt(252)

    rows.append([
        noms_affichage[t], q, round(prix, 2), round(valeur, 2),
        round(pv, 2), round(perf, 2),
        round(vol_actif_1y * 100, 2), round(vol_actif_5y * 100, 2)
    ])

df = pd.DataFrame(rows, columns=[
    "Actif", "Quantité", "Prix actuel (€)", "Valeur (€)",
    "Plus-value (€)", "Performance (%)",
    "Vol 1 an (%)", "Vol 5 ans (%)"
])

df["Pondération (%)"] = df["Valeur (€)"] / valeur_totale * 100
poids = df["Pondération (%)"].values / 100

# --- CALCULS DE RISQUE (ORIGINAUX) ---
Rf = 0.03 
vol_pf_1y = np.sqrt(poids @ ret_1y.cov().values @ poids) * np.sqrt(252)
vol_pf_5y = np.sqrt(poids @ ret_5y.cov().values @ poids) * np.sqrt(252)
vol_b_1y = rb_1y.std() * np.sqrt(252)
vol_b_5y = rb_5y.std() * np.sqrt(252)
Rp_1y = (ret_1y.mean() @ poids) * 252
Rp_5y = (ret_5y.mean() @ poids) * 252
Rp_b_1y = rb_1y.mean() * 252
Rp_b_5y = rb_5y.mean() * 252
sharpe_pf_1y = (Rp_1y - Rf) / vol_pf_1y
sharpe_pf_5y = (Rp_5y - Rf) / vol_pf_5y
sharpe_b_1y = (Rp_b_1y - Rf) / vol_b_1y
sharpe_b_5y = (Rp_b_5y - Rf) / vol_b_5y

def max_drawdown(rendements):
    cumul = (1 + rendements).cumprod()
    pic = cumul.cummax()
    drawdown = (cumul - pic) / pic
    return drawdown.min()

mdd_pf_1y = max_drawdown(ret_1y @ poids)
mdd_pf_5y = max_drawdown(ret_5y @ poids)
mdd_b_1y = max_drawdown(rb_1y)
mdd_b_5y = max_drawdown(rb_5y)
corr_pf_msci_1y = (ret_1y @ poids).corr(rb_1y)
corr_pf_msci_5y = (ret_5y @ poids).corr(rb_5y)
beta_1y = (ret_1y @ poids).cov(rb_1y) / rb_1y.var()
beta_5y = (ret_5y @ poids).cov(rb_5y) / rb_5y.var()
alpha_1y = Rp_1y - (Rf + beta_1y * (Rp_b_1y - Rf))
alpha_5y = Rp_5y - (Rf + beta_5y * (Rp_b_5y - Rf))

# --- AFFICHAGE STREAMLIT ---
st.divider()

st.subheader("1. Résumé du Portefeuille Témoin & Performance")
# PV Globale = Latente + Encaissée (Ventes + Div)
pv_globale = plus_value_totale + total_encaisse
performance_globale_pct = (pv_globale / (investi_total + (150 * 29.23))) * 100 # Calcul indicatif sur capital total engagé

col1, col2, col3, col4 = st.columns(4)
col1.metric("Valeur Portefeuille", f"{valeur_totale:,.2f} €")
col2.metric("PV Latente (En cours)", f"{plus_value_totale:,.2f} €")
col3.metric("Total Encaissé (Div + Ventes)", f"{total_encaisse:,.2f} €")
col4.metric("Gain Total (Absolu)", f"{pv_globale:,.2f} €", f"{performance_globale_pct:.2f} %")

st.divider()

st.subheader("2. Analyse de Risque & Performance Active (vs MSCI World)")
df_risque = pd.DataFrame({
    "Métrique": ["Volatilité (1 an)", "Volatilité (5 ans)", "Max Drawdown (1 an)", "Max Drawdown (5 ans)", "Ratio de Sharpe (1 an)", "Ratio de Sharpe (5 ans)", "Corrélation au marché (1 an)", "Corrélation au marché (5 ans)", "Bêta (1 an)", "Bêta (5 ans)", "Alpha de Jensen (1 an)", "Alpha de Jensen (5 ans)"],
    "Portefeuille Témoin": [f"{vol_pf_1y * 100:.2f} %", f"{vol_pf_5y * 100:.2f} %", f"{mdd_pf_1y * 100:.2f} %", f"{mdd_pf_5y * 100:.2f} %", f"{sharpe_pf_1y:.2f}", f"{sharpe_pf_5y:.2f}", f"{corr_pf_msci_1y:.2f}", f"{corr_pf_msci_5y:.2f}", f"{beta_1y:.2f}", f"{beta_5y:.2f}", f"{alpha_1y * 100:.2f} %", f"{alpha_5y * 100:.2f} %"],
    "MSCI World (Benchmark)": [f"{vol_b_1y * 100:.2f} %", f"{vol_b_5y * 100:.2f} %", f"{mdd_b_1y * 100:.2f} %", f"{mdd_b_5y * 100:.2f} %", f"{sharpe_b_1y:.2f}", f"{sharpe_b_5y:.2f}", "1.00", "1.00", "1.00", "1.00", "0.00 %", "0.00 %"]
})
st.dataframe(df_risque, width='stretch', hide_index=True)

st.divider()

col_gauche, col_droite = st.columns([2, 1])
with col_gauche:
    st.subheader("3. Composition détaillée")
    st.dataframe(df.style.format({"Prix actuel (€)": "{:.2f}", "Valeur (€)": "{:.2f}", "Plus-value (€)": "{:.2f}", "Performance (%)": "{:.2f}%", "Pondération (%)": "{:.2f}%"}), width='stretch', hide_index=True)

with col_droite:
    st.subheader("4. Historique & Dividendes")
    
    # Détail des ventes
    rows_realise = []
    for ticker, info in historique_gains.items():
        perf_bouygues = ((info["prix_vente"] - info["prix_achat"]) / info["prix_achat"]) * 100
        rows_realise.append({"Actif": info["nom"], "PV (€)": info["pv_realisee"], "Perf (%)": f"{perf_bouygues:.1f}%"})
    
    st.write("**Positions soldées (Ventes)**")
    st.table(pd.DataFrame(rows_realise))
    
    # Détail des dividendes
    st.write("**Dividendes perçus**")
    st.table(pd.DataFrame(list(dividendes_recus.items()), columns=["Source", "Montant (€)"]))
    
    # Rappel du total encaissé en bas du tableau
    st.info(f"**Total encaissé réel : {total_encaisse:.2f} €**")

st.divider()
st.subheader("5. Matrice de Corrélation")
corr_matrix = ret_5y.corr()
corr_matrix.columns = [noms_affichage.get(col, col) for col in corr_matrix.columns]
corr_matrix.index = [noms_affichage.get(idx, idx) for idx in corr_matrix.index]
st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"), width='stretch')

st.subheader("6. Performance cumulée (5 ans)")
ret_pf_5y_pondere = ret_5y @ poids
perf_pf = (1 + ret_pf_5y_pondere).cumprod() * 100
perf_benchmark = (1 + rb_5y).cumprod() * 100
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(perf_pf.index, perf_pf.values, label="Portefeuille Témoin", linewidth=2, color="#1f77b4")
ax.plot(perf_benchmark.index, perf_benchmark.values, label="MSCI World (IWDA)", linestyle="--", color="#ff7f0e")
ax.legend()
st.pyplot(fig)
