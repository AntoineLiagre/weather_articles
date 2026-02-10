import os
import pygrib
import numpy as np
import pandas as pd
from scipy.signal import detrend
from datetime import datetime

# Configuration des chemins de fichiers
output_dir = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data"
os.makedirs(output_dir, exist_ok=True)

# 1. Charger les données SSTa
ssta_file_path = os.path.join(output_dir, "SSTa.xlsx")
if not os.path.exists(ssta_file_path):
    raise FileNotFoundError(f"Le fichier {ssta_file_path} est introuvable.")

data_ssta = pd.read_excel(ssta_file_path)
years = data_ssta['Year'].values

# 2. Charger les valeurs de référence
reference_values_path = os.path.join(output_dir, "reference_values_1940_1970_adapted.csv")
if not os.path.exists(reference_values_path):
    raise FileNotFoundError(f"Le fichier {reference_values_path} est introuvable.")

reference_values_df = pd.read_csv(reference_values_path)

# Extraire les valeurs de référence
wind_speed_0_ref = reference_values_df.loc[reference_values_df['Parameter'] == 'wind_speed_0', 'Value'].values[0]
wind_direction_0_ref = reference_values_df.loc[reference_values_df['Parameter'] == 'wind_direction_0', 'Value'].values[0]
cape_0_ref = reference_values_df.loc[reference_values_df['Parameter'] == 'cape_0', 'Value'].values[0]

# 3. Fonction pour lire les données GRIB
def load_grib_data(file_path, short_name, level=None):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
        grbs = pygrib.open(file_path)
        data = []
        times = []
        for grb in grbs:
            if grb.shortName == short_name and (level is None or grb.level == level):
                data.append(grb.values)
                times.append(grb.validityDate)
        grbs.close()
        return np.array(data), np.array(times)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {file_path}: {str(e)}")
        return None, None

# 4. Charger les données u850, v850 et cape directement depuis les fichiers GRIB
u850_file_path = os.path.join(output_dir, "1f6ad324afe41a513fb0751df9d5d5d2.grib")
v850_file_path = os.path.join(output_dir, "1f6ad324afe41a513fb0751df9d5d5d2.grib")
cape_file_path = os.path.join(output_dir, "99441df5285ae811326b739643667aa6.grib")

u850_data, u850_times = load_grib_data(u850_file_path, 'u', level=850)
v850_data, v850_times = load_grib_data(v850_file_path, 'v', level=850)
cape_data, cape_times = load_grib_data(cape_file_path, 'cape')

if u850_data is None or v850_data is None or cape_data is None:
    raise ValueError("Erreur lors du chargement des données GRIB.")

# 5. Définir les masques régionaux
def create_regional_masks(lats, lons):
    pacific_mask = (lats[:, None] >= -5) & (lats[:, None] <= 5) & (lons >= -120) & (lons <= -80)
    atlantic_mask = (lats[:, None] >= 0) & (lats[:, None] <= 20) & (lons >= -60) & (lons <= 20)
    indian_mask = (lats[:, None] >= -10) & (lats[:, None] <= 10) & (lons >= 50) & (lons <= 110)

    return {
        'pacific': pacific_mask,
        'atlantic': atlantic_mask,
        'indian': indian_mask
    }

# 6. Fonction pour calculer les moyennes annuelles globales et régionales
def calculate_annual_means(data, times, region_mask=None):
    annual_means = []
    years = np.unique([int(str(t)[:4]) for t in times])
    for year in years:
        year_mask = np.array([int(str(t)[:4]) == year for t in times])
        if np.any(year_mask):
            data_year = data[year_mask]
            if region_mask is not None:
                data_year = data_year[:, region_mask]
            annual_mean = np.nanmean(data_year)
            annual_means.append(annual_mean)
    return np.array(annual_means)

# 7. Calcul des moyennes annuelles globales et régionales
lats = np.linspace(-90, 90, u850_data.shape[1])  # Exemple : 41 latitudes
lons = np.linspace(-180, 180, u850_data.shape[2])  # Exemple : 1440 longitudes

regional_masks = create_regional_masks(lats, lons)

u850_global_annual_means = calculate_annual_means(u850_data, u850_times)
u850_pacific_annual_means = calculate_annual_means(u850_data, u850_times, regional_masks['pacific'])
u850_atlantic_annual_means = calculate_annual_means(u850_data, u850_times, regional_masks['atlantic'])
u850_indian_annual_means = calculate_annual_means(u850_data, u850_times, regional_masks['indian'])

v850_global_annual_means = calculate_annual_means(v850_data, v850_times)
v850_pacific_annual_means = calculate_annual_means(v850_data, v850_times, regional_masks['pacific'])
v850_atlantic_annual_means = calculate_annual_means(v850_data, v850_times, regional_masks['atlantic'])
v850_indian_annual_means = calculate_annual_means(v850_data, v850_times, regional_masks['indian'])

cape_global_annual_means = calculate_annual_means(cape_data, cape_times)
cape_pacific_annual_means = calculate_annual_means(cape_data, cape_times, regional_masks['pacific'])
cape_atlantic_annual_means = calculate_annual_means(cape_data, cape_times, regional_masks['atlantic'])
cape_indian_annual_means = calculate_annual_means(cape_data, cape_times, regional_masks['indian'])

# 8. Calcul des paramètres de la cellule de Walker
def compute_walker_parameters(u850, v850, cape):
    wind_speed = np.sqrt(u850**2 + v850**2)
    wind_direction = np.arctan2(v850, u850) * 180 / np.pi

    return wind_speed, wind_direction, cape

# 9. Calcul des paramètres annuels globaux et régionaux
def compute_annual_walker_parameters(u850_annual, v850_annual, cape_annual):
    wind_speed_annual, wind_direction_annual, cape_annual = compute_walker_parameters(
        u850_annual, v850_annual, cape_annual)

    wind_speed_detrend = detrend(wind_speed_annual)
    wind_direction_detrend = detrend(wind_direction_annual)
    cape_detrend = detrend(cape_annual)

    return wind_speed_annual, wind_speed_detrend, wind_direction_annual, wind_direction_detrend, cape_annual, cape_detrend

# 10. Calcul des paramètres annuels pour chaque région
global_params = compute_annual_walker_parameters(u850_global_annual_means, v850_global_annual_means, cape_global_annual_means)
pacific_params = compute_annual_walker_parameters(u850_pacific_annual_means, v850_pacific_annual_means, cape_pacific_annual_means)
atlantic_params = compute_annual_walker_parameters(u850_atlantic_annual_means, v850_atlantic_annual_means, cape_atlantic_annual_means)
indian_params = compute_annual_walker_parameters(u850_indian_annual_means, v850_indian_annual_means, cape_indian_annual_means)

# 11. Sauvegarder les résultats dans un fichier CSV
years = np.unique([int(str(t)[:4]) for t in u850_times])

# Normaliser les résultats par rapport aux valeurs de référence
def normalize_by_reference(params, ref_values):
    wind_speed_norm = params[0] / ref_values['wind_speed_0']
    wind_direction_norm = params[2] - ref_values['wind_direction_0']
    cape_norm = params[4] / ref_values['cape_0']

    return wind_speed_norm, params[1], wind_direction_norm, params[3], cape_norm, params[5]

ref_values = {
    'wind_speed_0': wind_speed_0_ref,
    'wind_direction_0': wind_direction_0_ref,
    'cape_0': cape_0_ref
}

global_params_norm = normalize_by_reference(global_params, ref_values)
pacific_params_norm = normalize_by_reference(pacific_params, ref_values)
atlantic_params_norm = normalize_by_reference(atlantic_params, ref_values)
indian_params_norm = normalize_by_reference(indian_params, ref_values)

# Préparation des données pour le DataFrame
data = {
    'Year': years,
    'wind_speed_global': global_params[0],
    'wind_speed_detrend_global': global_params[1],
    'wind_direction_global': global_params[2],
    'wind_direction_detrend_global': global_params[3],
    'cape_global': global_params[4],
    'cape_detrend_global': global_params[5],

    'wind_speed_pacific': pacific_params[0],
    'wind_speed_detrend_pacific': pacific_params[1],
    'wind_direction_pacific': pacific_params[2],
    'wind_direction_detrend_pacific': pacific_params[3],
    'cape_pacific': pacific_params[4],
    'cape_detrend_pacific': pacific_params[5],

    'wind_speed_atlantic': atlantic_params[0],
    'wind_speed_detrend_atlantic': atlantic_params[1],
    'wind_direction_atlantic': atlantic_params[2],
    'wind_direction_detrend_atlantic': atlantic_params[3],
    'cape_atlantic': atlantic_params[4],
    'cape_detrend_atlantic': atlantic_params[5],

    'wind_speed_indian': indian_params[0],
    'wind_speed_detrend_indian': indian_params[1],
    'wind_direction_indian': indian_params[2],
    'wind_direction_detrend_indian': indian_params[3],
    'cape_indian': indian_params[4],
    'cape_detrend_indian': indian_params[5],

    'wind_speed_global_norm': global_params_norm[0],
    'wind_speed_detrend_global_norm': global_params_norm[1],
    'wind_direction_global_norm': global_params_norm[2],
    'wind_direction_detrend_global_norm': global_params_norm[3],
    'cape_global_norm': global_params_norm[4],
    'cape_detrend_global_norm': global_params_norm[5],

    'wind_speed_pacific_norm': pacific_params_norm[0],
    'wind_speed_detrend_pacific_norm': pacific_params_norm[1],
    'wind_direction_pacific_norm': pacific_params_norm[2],
    'wind_direction_detrend_pacific_norm': pacific_params_norm[3],
    'cape_pacific_norm': pacific_params_norm[4],
    'cape_detrend_pacific_norm': pacific_params_norm[5],

    'wind_speed_atlantic_norm': atlantic_params_norm[0],
    'wind_speed_detrend_atlantic_norm': atlantic_params_norm[1],
    'wind_direction_atlantic_norm': atlantic_params_norm[2],
    'wind_direction_detrend_atlantic_norm': atlantic_params_norm[3],
    'cape_atlantic_norm': atlantic_params_norm[4],
    'cape_detrend_atlantic_norm': atlantic_params_norm[5],

    'wind_speed_indian_norm': indian_params_norm[0],
    'wind_speed_detrend_indian_norm': indian_params_norm[1],
    'wind_direction_indian_norm': indian_params_norm[2],
    'wind_direction_detrend_indian_norm': indian_params_norm[3],
    'cape_indian_norm': indian_params_norm[4],
    'cape_detrend_indian_norm': indian_params_norm[5],
}

walker_parameters = pd.DataFrame(data)

output_path = os.path.join(output_dir, "walker_parameters_annual_regional.csv")
walker_parameters.to_csv(output_path, index=False)

print(f"\nLes paramètres annuels de la cellule de Walker ont été sauvegardés dans '{output_path}'.")
