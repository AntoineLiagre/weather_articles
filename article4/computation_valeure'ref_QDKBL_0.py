import pandas as pd
import os

# Configuration des chemins de fichiers
output_dir = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data"
results_path = os.path.join(output_dir, "annual_spectral_parameters.csv")

# Vérifier que le fichier de résultats existe
if not os.path.exists(results_path):
    raise FileNotFoundError(f"Le fichier {results_path} est introuvable.")

# Charger les résultats calculés
results_df = pd.read_csv(results_path)

# Afficher les colonnes disponibles pour vérifier
print("Colonnes disponibles dans le fichier de résultats :")
print(results_df.columns.tolist())

# Vérifier si les colonnes nécessaires existent
required_columns = ['D_max', 'Q_max', 'kappa_B_global', 'lambda']
missing_columns = []
for column in required_columns:
    if column not in results_df.columns:
        missing_columns.append(column)

if missing_columns:
    raise ValueError(f"Les colonnes suivantes sont manquantes dans les données: {', '.join(missing_columns)}")

# Filtrer les données pour la période de référence (1940-1970)
reference_period_data = results_df[(results_df['Year'] >= 1940) & (results_df['Year'] <= 1970)]

if reference_period_data.empty:
    raise ValueError("Aucune donnée disponible pour la période de référence (1940-1970).")

# Calcul des valeurs de référence
D0 = reference_period_data['D_max'].mean()
Q0 = reference_period_data['Q_max'].mean()
kappa_B0 = reference_period_data['kappa_B_global'].mean()
lambda_0 = reference_period_data['lambda'].mean()

# Afficher les valeurs de référence
print(f"D0 (1940-1970): {D0}")
print(f"Q0 (1940-1970): {Q0}")
print(f"kappa_B0 (1940-1970): {kappa_B0}")
print(f"lambda_0 (1940-1970): {lambda_0}")

# Sauvegarder les valeurs de référence dans un fichier CSV
reference_values = pd.DataFrame({
    'Parameter': ['D0', 'Q0', 'kappa_B0', 'lambda_0'],
    'Value': [D0, Q0, kappa_B0, lambda_0]
})

reference_values.to_csv(os.path.join(output_dir, "reference_values_DQKL_1940_1970.csv"), index=False)

print("\nLes valeurs de référence (D0, Q0, kappa_B0, lambda_0) ont été sauvegardées dans 'reference_values_DQKL_1940_1970.csv'.")
