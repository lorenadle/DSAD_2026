import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("GlobalIndicatorsPerCapita_2021.csv")
df_coduri=pd.read_csv("CoduriTari.csv")
print( df)
print(df_coduri)
df_merge=df.merge(df_coduri,how='inner',on='CountryID')
print(df_merge)
coloane=df_merge.select_dtypes(include=[np.number]).columns
df[coloane]=df[coloane].fillna(df[coloane].mean())
# Să se salveze în fișierul Cerinta1.csv, pentru fiecare țară, ramura cu valoarea adăugată cea mai mare.
# Se va salva codul de țară, numele țării și denumirea indicatorului pentru ramura cu valoarea adăugată cea mai mare.
# Indicatorii cu valoarea adăugată pe ramuri sunt speciaficați pe ultima linie din tabelul de mai sus
ramuri = [
    "AgrHuntForFish", "Construction", "Manufacturing",
    "MiningManUt", "TradeT", "TransportComm", "Other"
]
df["Ramura Dominanta"] = df[ramuri].idxmax(axis=1)
df_c1 = df[["CountryID", "Country", "Ramura Dominanta"]]
df_c1.to_csv("Cerinta1.csv", index=False)

#Salvarea în fișierul Cerinta2.csv a țărilor la care s-au înregistrat cele mai mari valori pentru indicatorii de mai sus
#(de la GNI până la sfârșit), la nivel de continent. Pentru fiecare continent se va afișa denumirea continentului și
# id-urile țărilor cu valorile cele mai mari pentru indicatori

indicatori = [
    "GNI", "ChangesInv", "Exports", "Imports",
    "FinalConsExp", "GrossCF", "HouseholdConsExp"
] + ramuri

rez = df_merge[["Continent"]].drop_duplicates().set_index("Continent")
for ind in indicatori:
    idx = df_merge.groupby("Continent")[ind].idxmax()
    rez[ind] = df_merge.loc[idx, "CountryID"].values

rez.reset_index().to_csv("Cerinta2.csv", index=False)

#Să se efectueze analiza de clusteri prin metoda Ward, pentru indicatorii macroeconomici de mai sus și să se furnizeze următoarele rezultate:
#1. Dendrograma partiției cu 3 clusteri.
#2. Componența partiției formate din 3 clusteri. Pentru fiecare instanță se va specifica clusterul din care face parte și
# scorul Silhouette al instanței. Partiția va fi salvată în fișierul p3.csv, pe 4 coloane: codul de țară, denumirea țării,
# clusterul din care face parte și scorul Sihouette.
#3. Trasare plot partiție în axe principale pentru partiția din 3 clusteri.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# ============================================================
# 1. SELECTAREA INDICATORILOR MACROECONOMICI
# ============================================================

indicatori = [
    "GNI", "ChangesInv", "Exports", "Imports",
    "FinalConsExp", "GrossCF", "HouseholdConsExp",
    "AgrHuntForFish", "Construction", "Manufacturing",
    "MiningManUt", "TradeT", "TransportComm", "Other"
]

X = df_merge[indicatori].values


# ============================================================
# 2. STANDARDIZAREA DATELOR
# ============================================================

scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# ============================================================
# 3. CLUSTERING IERARHIC – METODA WARD
# ============================================================

Z = linkage(X_std, method="ward")


# ============================================================
# 4. DENDROGRAMA
# ============================================================

plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title("Dendrograma – metoda Ward")
plt.xlabel("Instanțe")
plt.ylabel("Distanța")
plt.show()


# ============================================================
# 5. PARTIȚIA CU 3 CLUSTERI
# ============================================================

clusteri = fcluster(Z, t=3, criterion="maxclust")


# ============================================================
# 6. SCOR SILHOUETTE PE INSTANȚĂ
# ============================================================

sil = silhouette_samples(X_std, clusteri)


# ============================================================
# 7. SALVAREA PARTIȚIEI – p3.csv
# ============================================================

df_p3 = pd.DataFrame({
    "CountryID": df_merge["CountryID"],
    "Country": df_merge["Country"],
    "Cluster": clusteri,
    "Silhouette": sil
})

df_p3.to_csv("p3.csv", index=False)


# ============================================================
# 8. PLOT PARTIȚIE ÎN AXE PRINCIPALE (PC1–PC2)
# ============================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusteri,
    cmap="tab10"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Partiția în planul PC1–PC2")
plt.colorbar(label="Cluster")
plt.show()