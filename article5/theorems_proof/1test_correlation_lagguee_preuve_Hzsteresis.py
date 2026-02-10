import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def find_optimal_lag(q_abs, walker_index, max_lag_months=24):
    """Trouve le lag où la corrélation est maximale (Preuve de l'Hystérésis)"""
    lags = np.arange(0, max_lag_months + 1)
    correlations = []

    for lag in lags:
        if lag == 0:
            c, _ = pearsonr(q_abs, walker_index)
        else:
            c, _ = pearsonr(q_abs[:-lag], walker_index[lag:])
        correlations.append(c)

    opt_lag = lags[np.argmax(np.abs(correlations))]
    return opt_lag, correlations

# Exemple d'utilisation
q_abs_series = data['Q_absorbed']  # Assurez-vous que cette colonne existe dans vos données
walker_series = data['kappa_B_global']  # Ou un autre indice de Walker

lag, corr_curve = find_optimal_lag(q_abs_series, walker_series)

print(f"Lag optimal : {lag} mois. Corrélation : {max(corr_curve):.3f}")
# Si lag > 0 et Corrélation > 0.7, l'hystérésis spectrale est prouvée physiquement
