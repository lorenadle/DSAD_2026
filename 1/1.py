import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype

# =================================================
# FOLDERE
# =================================================

INPUT_DIR = "datain"
OUTPUT_DIR = "dataout"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================
# CITIRE DATE
# =================================================

df = pd.read_csv(f"{INPUT_DIR}/MiseNatPopTari.csv")
coduri = pd.read_csv(f"{INPUT_DIR}/CoduriTariExtins.csv")

# =================================================
# A.1 ȚĂRI CU SPOR NATURAL NEGATIV
# =================================================

cerinta1 = df[df["RS"] < 0]
cerinta1.to_csv(f"{OUTPUT_DIR}/cerinta1.csv", index=False)

# =================================================
# A.2 MEDIA INDICATORILOR PE CONTINENTE
# =================================================
# MERGE CORECT: Country_Number

df_extins = df.merge(
    coduri,
    on="Country_Number",
    how="left"
)

indicatori = ["RS", "FR", "LM", "MMR", "LE", "LEM", "LEF"]

cerinta2 = (
    df_extins
    .groupby("Continent")[indicatori]
    .mean()
)

cerinta2.to_csv(f"{OUTPUT_DIR}/cerinta2.csv")

# =================================================
# B. ANALIZA ÎN COMPONENTE PRINCIPALE (PCA)
# =================================================

X = df[indicatori].copy()

# Tratarea valorilor lipsă – media
for col in X.columns:
    if is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].mean())

# Standardizare
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_std)

# =================================================
# B.1 VARIANTELE COMPONENTELOR PRINCIPALE
# =================================================

print("VARIANTELE COMPONENTELOR PRINCIPALE")
print("-----------------------------------")

variante = pca.explained_variance_
ponderi = pca.explained_variance_ratio_
cumul = ponderi.cumsum()

for i in range(len(variante)):
    print(
        f"PC{i+1}: "
        f"Varianta = {variante[i]:.4f}, "
        f"Pondere = {ponderi[i]*100:.2f}%, "
        f"Cumul = {cumul[i]*100:.2f}%"
    )

# =================================================
# B.2 SCORURI PCA
# =================================================

scoruri = pd.DataFrame(
    X_pca,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])],
    index=df["Country_Name"]
)

scoruri.to_csv(f"{OUTPUT_DIR}/scoruri.csv")

# =================================================
# B.3 GRAFICUL SCORURILOR (PC1 vs PC2)
# =================================================

plt.figure(figsize=(8, 6))
plt.scatter(scoruri["PC1"], scoruri["PC2"], alpha=0.7)

plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel("Componenta principală 1 (PC1)")
plt.ylabel("Componenta principală 2 (PC2)")
plt.title("Scorurile instanțelor în primele două componente principale")
plt.grid(True)

plt.show()
