from scipy.stats import linregress
import matplotlib.pyplot as plt

def validate_triad_partition(q_abs, q_max, teprat_index):
    """Prouve que TEPRAT compense la perte de capacité du puits"""
    residual_capacity = 1 - (q_abs / q_max)

    # Régression linéaire
    slope, intercept, r_value, p_value, std_err = linregress(residual_capacity, teprat_index)

    plt.figure(figsize=(12, 6))
    plt.scatter(residual_capacity, teprat_index, alpha=0.5)
    plt.plot(residual_capacity, intercept + slope*residual_capacity, 'r', label=f'R² = {r_value**2:.2f}')
    plt.xlabel('Capacité Résiduelle Océanique (1 - Q_abs/Q_max)')
    plt.ylabel('Indice TEPRAT (Cinétique 200hPa)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemple d'utilisation
q_abs_series = data['Q_absorbed']
q_max_series = data['Q_max']
teprat_series = data['TEPRAT']  # Assurez-vous que cette colonne existe dans vos données

validate_triad_partition(q_abs_series, q_max_series, teprat_series)
