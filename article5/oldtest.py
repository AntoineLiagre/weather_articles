import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

# 1. Charger les données
data = pd.read_csv(r"C:\Users\antoi\Downloads\data_Global_n.csv", sep=';')
print(data.columns)

# 2. Extraire les colonnes
N_Pacific_Blob = data['NPB'].values
enso = data['ENSO 3.4'].values
Indian_Ocean = data['Indian Ocean'].values
South_Pacific = data['South Pacific'].values
years = np.arange(1940, 2026)

# 3. Fonction pour calculer n(t) avec les règles de relaxation
def calculate_n(ssta, threshold_inc=0.15, threshold_dec=-0.10, min_cooling_years=2):
    ssta_detrend = detrend(ssta)
    yoY_diff = np.diff(ssta_detrend)
    n = 0
    n_list = [0]
    cooling_counter = 0

    for diff in yoY_diff:
        if diff >= threshold_inc:
            n += 1
            cooling_counter = 0
        elif diff <= threshold_dec:
            cooling_counter += 1
            if cooling_counter >= min_cooling_years and n > 0:
                n -= 1
                cooling_counter = 0
        else:
            cooling_counter = 0

        n_list.append(n)

    return np.array(n_list)

# 4. Calculer n(t) pour chaque région avec leurs seuils spécifiques
n_blob = calculate_n(N_Pacific_Blob, threshold_inc=0.20, threshold_dec=-0.15, min_cooling_years=2)
n_enso = calculate_n(enso, threshold_inc=0.50, threshold_dec=-0.50, min_cooling_years=1)
n_iod = calculate_n(Indian_Ocean, threshold_inc=0.20, threshold_dec=-0.15, min_cooling_years=2)
n_sp = calculate_n(South_Pacific, threshold_inc=0.20, threshold_dec=-0.15, min_cooling_years=2)

# 5. Tracer les résultats
plt.figure(figsize=(12, 6))
plt.plot(years[1:], n_blob[:-1], label='N Pacific Blob', color='blue')
plt.plot(years[1:], n_enso[:-1], label='ENSO 3.4', color='red')
plt.plot(years[1:], n_iod[:-1], label='Indian Ocean', color='green')
plt.plot(years[1:], n_sp[:-1], label='South Pacific', color='purple')
plt.xlabel('Year')
plt.ylabel('n(t) - Cumulative Equilibria')
plt.title('Cumulative Equilibria n(t) for Different Regions (1941-2025)')
plt.legend()
plt.grid()
plt.savefig('Article4_n_t_Regions.png')
plt.show()

# Afficher les valeurs finales de n(t) en 2025
print(f"n_blob (2025): {n_blob[-1]}")
print(f"n_enso (2025): {n_enso[-1]}")
print(f"n_iod (2025): {n_iod[-1]}")
print(f"n_sp (2025): {n_sp[-1]}")