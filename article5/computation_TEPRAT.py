import xarray as xr
import numpy as np
import pandas as pd

# Charger les données de vent à 200 hPa (u200 et v200)
file_path_u200 = r"chemin/vers/les/donnees/u200.nc"
file_path_v200 = r"chemin/vers/les/donnees/v200.nc"

ds_u200 = xr.open_dataset(file_path_u200)
ds_v200 = xr.open_dataset(file_path_v200)

# Extraire les variables u200 et v200
u200 = ds_u200['u']
v200 = ds_v200['v']

# Calculer l'énergie cinétique (KE)
rho = 1.0  # Densité de l'air à 200 hPa (on peut considérer une valeur constante pour simplifier)
KE = 0.5 * rho * (u200**2 + v200**2)

# Calculer la fonction de courant (psi)
# Pour simplifier, on peut utiliser une approximation de la fonction de courant
# en intégrant les composantes du vent
# Note: Ce calcul peut être complexe et nécessite des données sur une grille régulière
# Voici une méthode simplifiée pour obtenir une approximation de la fonction de courant
psi = np.zeros_like(u200)
for lat in range(u200.shape[1]):
    psi[:, lat, :] = np.cumsum(u200[:, lat, :], axis=2)  # Intégration simplifiée

# Calculer la variance de la fonction de courant
psi_var = psi.var(dim=('longitude', 'latitude'), skipna=True)

# Calculer TEPRAT (variance de la fonction de courant ou énergie cinétique)
TEPRAT = psi_var

# Sauvegarder les résultats
TEPRAT.to_netcdf(r"chemin/vers/les/resultats/TEPRAT.nc")
TEPRAT.to_dataframe().to_csv(r"chemin/vers/les/resultats/TEPRAT.csv")
import xarray as xr
import numpy as np
import pandas as pd

# Charger les données de vent à 200 hPa (u200 et v200)
file_path_u200 = r"chemin/vers/les/donnees/u200.nc"
file_path_v200 = r"chemin/vers/les/donnees/v200.nc"

ds_u200 = xr.open_dataset(file_path_u200)
ds_v200 = xr.open_dataset(file_path_v200)

# Extraire les variables u200 et v200
u200 = ds_u200['u']
v200 = ds_v200['v']

# Calculer l'énergie cinétique (KE)
rho = 1.0  # Densité de l'air à 200 hPa (on peut considérer une valeur constante pour simplifier)
KE = 0.5 * rho * (u200**2 + v200**2)

# Calculer la fonction de courant (psi)
# Pour simplifier, on peut utiliser une approximation de la fonction de courant
# en intégrant les composantes du vent
# Note: Ce calcul peut être complexe et nécessite des données sur une grille régulière
# Voici une méthode simplifiée pour obtenir une approximation de la fonction de courant
psi = np.zeros_like(u200)
for lat in range(u200.shape[1]):
    psi[:, lat, :] = np.cumsum(u200[:, lat, :], axis=2)  # Intégration simplifiée

# Calculer la variance de la fonction de courant
psi_var = psi.var(dim=('longitude', 'latitude'), skipna=True)

# Calculer TEPRAT (variance de la fonction de courant ou énergie cinétique)
TEPRAT = psi_var

# Sauvegarder les résultats
TEPRAT.to_netcdf(r"chemin/vers/les/resultats/TEPRAT.nc")
TEPRAT.to_dataframe().to_csv(r"chemin/vers/les/resultats/TEPRAT.csv")
