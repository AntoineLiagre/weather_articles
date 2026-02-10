import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les indices climatiques
enso_path = r"chemin/vers/les/donnees/ONI.csv"
iod_path = r"chemin/vers/les/donnees/IOD.csv"
pdo_path = r"chemin/vers/les/donnees/PDO.csv"
pmm_path = r"chemin/vers/les/donnees/PMM.csv"
amm_path = r"chemin/vers/les/donnees/AMM.csv"

# Lire les fichiers CSV
enso_data = pd.read_csv(enso_path)
iod_data = pd.read_csv(iod_path)
pdo_data = pd.read_csv(pdo_path)
pmm_data = pd.read_csv(pmm_path)
amm_data = pd.read_csv(amm_path)

# Afficher les premières lignes pour vérifier
print(enso_data.head())
print(iod_data.head())
print(pdo_data.head())
print(pmm_data.head())
print(amm_data.head())

# Exemple pour ENSO : Sélectionner les données autour des pics saisonniers
enso_data['Date'] = pd.to_datetime(enso_data['Year'].astype(str) + '-' + enso_data['Month'].astype(str) + '-01')
enso_data.set_index('Date', inplace=True)

# Filtrer pour les mois d'octobre à mars (fenêtre de -3 à +3 mois autour du pic de décembre)
enso_seasonal = enso_data[enso_data.index.month.isin([10, 11, 12, 1, 2, 3])]

# Visualiser les données ENSO
plt.figure(figsize=(12, 6))
plt.plot(enso_seasonal.index, enso_seasonal['ONI'], label='ONI')
plt.title('ENSO Index (ONI) - Octobre à Mars')
plt.xlabel('Date')
plt.ylabel('ONI')
plt.legend()
plt.grid(True)
plt.show()
