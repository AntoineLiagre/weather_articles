import xarray as xr
import numpy as np
import gsw  # Gibbs SeaWater library

# Charger les données Argo (exemple avec un fichier netCDF)
file_path = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\argo_data.nc"
ds = xr.open_dataset(file_path)

# Extraire les variables nécessaires
temp = ds['TEMP']  # Température
salt = ds['PSAL']  # Salinité
pressure = ds['PRES']  # Pression

# Calculer la densité potentielle
SA = gsw.SA_from_SP(salt, pressure, ds['LONGITUDE'], ds['LATITUDE'])
CT = gsw.CT_from_t(SA, temp, pressure)
rho = gsw.rho(SA, CT, pressure)

# Calculer N^2
g = 9.81  # Accélération due à la gravité
N2 = - (g / rho) * rho.differentiate('PRES')

# Intégrer N^2 sur la profondeur pour obtenir ΔN
dN = N2.integrate('PRES')

# Sauvegarder les résultats
dN.to_netcdf(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\stratification.nc")
