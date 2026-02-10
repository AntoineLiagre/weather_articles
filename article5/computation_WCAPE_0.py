import numpy as np
import pandas as pd
import os

# Configuration des chemins de fichiers
output_dir = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data"
os.makedirs(output_dir, exist_ok=True)

# Charger les données nécessaires
data_file_path = os.path.join(output_dir, "walker_parameters_annual.csv")
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"Le fichier {data_file_path} est introuvable.")

data = pd.read_csv(data_file_path)

# Afficher les colonnes disponibles pour vérifier
print("Colonnes disponibles dans le fichier :")
print(data.columns.tolist())

# Vérifier qu'il y a des données pour la période de référence (1940-1970)
reference_period_data = data[(data['Year'] >= 1940) & (data['Year'] <= 1970)]
if reference_period_data.empty:
    raise ValueError("Aucune donnée disponible pour la période de référence (1940-1970).")

# Calcul des valeurs de référence avec les colonnes disponibles
wind_speed_0 = reference_period_data['wind_speed'].mean()
wind_speed_detrend_0 = reference_period_data['wind_speed_detrend'].mean()
wind_direction_0 = reference_period_data['wind_direction'].mean()
wind_direction_detrend_0 = reference_period_data['wind_direction_detrend'].mean()
cape_0 = reference_period_data['cape'].mean()
cape_detrend_0 = reference_period_data['cape_detrend'].mean()

# Afficher les valeurs de référence
print(f"wind_speed_0 (1940-1970): {wind_speed_0}")
print(f"wind_speed_detrend_0 (1940-1970): {wind_speed_detrend_0}")
print(f"wind_direction_0 (1940-1970): {wind_direction_0}")
print(f"wind_direction_detrend_0 (1940-1970): {wind_direction_detrend_0}")
print(f"cape_0 (1940-1970): {cape_0}")
print(f"cape_detrend_0 (1940-1970): {cape_detrend_0}")

# Sauvegarder les valeurs de référence dans un fichier CSV
reference_values = pd.DataFrame({
    'Parameter': ['wind_speed_0', 'wind_speed_detrend_0', 'wind_direction_0', 'wind_direction_detrend_0', 'cape_0', 'cape_detrend_0'],
    'Value': [wind_speed_0, wind_speed_detrend_0, wind_direction_0, wind_direction_detrend_0, cape_0, cape_detrend_0]
})

reference_values.to_csv(os.path.join(output_dir, "reference_values_1940_1970_adapted.csv"), index=False)

print("\nLes valeurs de référence (1940-1970) ont été sauvegardées dans 'reference_values_1940_1970_adapted.csv'.")
