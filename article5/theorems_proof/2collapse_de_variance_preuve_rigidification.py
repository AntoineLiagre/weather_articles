def test_variance_collapse(walker_series, window_years=5):
    """Calcule la perte de résilience dynamique (Stiffening)"""
    window = window_years * 12  # Convertir les années en mois
    rolling_var = walker_series.rolling(window=window, center=False).var()
    return rolling_var

# Exemple d'utilisation
walker_series = data['kappa_B_global']
var_w = test_variance_collapse(walker_series)

# Visualisation
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
var_w.plot(label="Variance de Walker (Fenêtre 5 ans)")
plt.title("Variance Glissante de l'Indice de Walker")
plt.xlabel("Année")
plt.ylabel("Variance")
plt.legend()
plt.grid(True)
plt.show()
# Un déclin post-2020 dans l'Indien valide votre prédiction de "Bascule".
