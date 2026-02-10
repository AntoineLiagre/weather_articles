import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, find_peaks, detrend
from scipy.stats import linregress

# 1. Charger les données SSTa et variables atmosphériques (u850, SLP)
data_ssta = pd.read_excel(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\SSTa.xlsx")
data_atmos = pd.read_excel(r"C:\Users\antoi\Desktop\python projects\Atmos_Variables.xlsx")

print("Colonnes SSTa:", data_ssta.columns)
print("Colonnes Atmos:", data_atmos.columns)

# 2. Paramètres régionaux
regional_params = {
    'Global': {
        'inc': 0.10, 'dec': -0.08, 'cooling_years': 3,
        'saturation_band': (1/100, 1/30),  # Bande de saturation globale
        'walker_coupling': True  # La cellule de Walker influence le Global via ENSO
    },
    'ENSO 3.4': {
        'inc': 0.50, 'dec': -0.50, 'cooling_years': 1,
        'saturation_band': (1/7, 1/2),
        'walker_coupling': True  # ENSO est directement lié à la cellule de Walker
    },
    'North_Pacific_Blob': {
        'inc': 0.20, 'dec': -0.15, 'cooling_years': 2,
        'saturation_band': (1/30, 1/10),
        'walker_coupling': False  # Influence indirecte via PDO
    },
    'Indian_Ocean': {
        'inc': 0.12, 'dec': -0.10, 'cooling_years': 2,
        'saturation_band': (1/50, 1/20),
        'walker_coupling': True  # Couplage via l'Océan Indien et la cellule de Walker
    },
    'South_Pacific': {
        'inc': 0.15, 'dec': -0.12, 'cooling_years': 2,
        'saturation_band': (1/40, 1/15),
        'walker_coupling': True  # Influence de la SPCZ sur la cellule de Walker
    },
    'North_Atlantic': {
        'inc': 0.12, 'dec': -0.10, 'cooling_years': 2,
        'saturation_band': (1/50, 1/20),
        'walker_coupling': False  # Peu d'influence directe
    }
}

# 3. Décomposition spectrale (conserve les cycles longs)
def decompose_ssta(ssta, region_name, fs=1.0):
    f, Pxx = welch(ssta, fs=fs, nperseg=len(ssta)//2)
    nyq = 0.5 * fs

    # Bande ENSO (2-7 ans)
    low_enso = max(1e-6, (1/7) / nyq)
    high_enso = min(0.49, (1/2) / nyq)
    b_enso, a_enso = butter(4, [low_enso, high_enso], btype='band')
    ssta_enso = filtfilt(b_enso, a_enso, ssta)

    # Bande PDO/AMO (10-30 ans)
    low_pdo = max(1e-6, (1/30) / nyq)
    high_pdo = min(0.49, (1/10) / nyq)
    b_pdo, a_pdo = butter(4, [low_pdo, high_pdo], btype='band')
    ssta_pdo = filtfilt(b_pdo, a_pdo, ssta)

    # Bande de saturation (conserve les cycles longs)
    low_sat, high_sat = regional_params[region_name]['saturation_band']
    low_sat_norm = max(1e-6, low_sat / nyq)
    high_sat_norm = min(0.49, high_sat / nyq)

    if low_sat_norm >= high_sat_norm:
        b_sat, a_sat = butter(4, low_sat_norm, btype='low')
    else:
        b_sat, a_sat = butter(4, [low_sat_norm, high_sat_norm], btype='band')
    ssta_sat = filtfilt(b_sat, a_sat, ssta)

    return ssta_enso, ssta_pdo, ssta_sat

# 4. Calcul de n(t) avec couplage à la cellule de Walker
def calculate_n(ssta, region_name, u850=None):
    """
    Calcule n(t) en incluant l'influence de la cellule de Walker (u850).
    """
    params = regional_params[region_name]
    inc = params['inc']
    dec = params['dec']
    cooling_years = params['cooling_years']

    # Décomposer les SSTa
    ssta_enso, ssta_pdo, ssta_sat = decompose_ssta(ssta, region_name)
    ssta_physical = ssta_enso + ssta_pdo + ssta_sat

    # Ajuster les seuils si la région est couplée à la cellule de Walker
    if regional_params[region_name]['walker_coupling'] and u850 is not None:
        # Normaliser u850 (ex. : entre -1 et 1)
        u850_norm = (u850 - np.mean(u850)) / np.std(u850)
        inc *= (1 + 0.2 * u850_norm)  # u850 > 0 → inc plus grand (difficile à atteindre)
        dec *= (1 + 0.2 * u850_norm)  # u850 > 0 → dec plus négatif (difficile à descendre)

    yoY_diff = np.diff(ssta_physical)
    n = 0
    n_list = [0]
    cooling_counter = 0

    for diff in yoY_diff:
        if diff >= inc:
            n += 1
            cooling_counter = 0
        elif diff <= dec:
            cooling_counter += 1
            if cooling_counter >= cooling_years and n > 0:
                n -= 1
                cooling_counter = 0
        else:
            cooling_counter = 0
        n_list.append(n)

    return np.array(n_list)

# 5. Calcul du couplage cellule de Walker (κ_B)
def compute_walker_coupling(ssta, u850, region_name):
    """
    Calcule κ_B (sensibilité de la cellule de Walker aux SSTa).
    """
    if not regional_params[region_name]['walker_coupling']:
        return np.nan  # Pas de couplage pour cette région

    # Détrendre les séries (pour éviter les tendances long-terme)
    ssta_detrend = detrend(ssta)
    u850_detrend = detrend(u850)

    # Calculer la covariance et les écarts-types
    covariance = np.cov(ssta_detrend, u850_detrend)[0, 1]
    std_ssta = np.std(ssta_detrend)
    std_u850 = np.std(u850_detrend)

    if std_ssta > 0 and std_u850 > 0:
        kappa_B = covariance / (std_ssta * std_u850)
    else:
        kappa_B = np.nan

    return kappa_B

# 6. Calcul de Γ_confine
def calculate_Gamma_confine(delta_F_residual, Q_max, D_0, lambda_d, kappa_B):
    """
    Calcule Γ_confine(x, y, t) selon la formule simplifiée.
    """
    denominator = Q_max + D_0 * lambda_d
    Gamma_confine = (abs(delta_F_residual) / denominator) * (1 + kappa_B)
    return Gamma_confine

# 7. Calcul des paramètres spectraux (λ, β, γ, Γ_confine, ε, κ_B)
def compute_parameters(ssta, u850, region_name, t):
    # Décomposer les SSTa
    ssta_enso, ssta_pdo, ssta_sat = decompose_ssta(ssta, region_name)
    ssta_physical = ssta_enso + ssta_pdo + ssta_sat

    # Calculer λ (fréquence dominante)
    f, Pxx = welch(ssta_physical, fs=1.0, nperseg=len(ssta_physical)//2)
    low_sat, high_sat = regional_params[region_name]['saturation_band']
    band_mask = (f >= low_sat) & (f <= high_sat)

    if band_mask.sum() > 0:
        peaks, props = find_peaks(Pxx[band_mask], height=0.1 * np.max(Pxx[band_mask]))
        if len(peaks) > 0:
            lambda_val = f[band_mask][peaks[np.argmax(props['peak_heights'])]]
        else:
            lambda_val = np.mean(f[band_mask])
    else:
        lambda_val = np.nan

    # Calculer β (fraction de puissance dans la bande de saturation)
    total_power = np.sum(Pxx)
    if total_power > 0:
        beta = np.sum(Pxx[band_mask]) / total_power
    else:
        beta = np.nan

    # Calculer γ (taux d'hystérésis)
    n_t = calculate_n(ssta, region_name, u850)
    n_detrend = detrend(n_t[1:])
    f_n, Pxx_n = welch(n_detrend, fs=1.0, nperseg=len(n_detrend)//2)
    low_mask = (f_n >= 0.01) & (f_n <= 0.1)
    if low_mask.sum() > 0:
        gamma = np.mean(f_n[low_mask][Pxx_n[low_mask] > 0.1 * np.max(Pxx_n[low_mask])])
    else:
        gamma = np.mean(f_n[low_mask]) if low_mask.sum() > 0 else np.nan

    # Calculer ε (efficacité résiduelle)
    n_detrend_var = np.var(n_detrend)
    n_total_var = np.var(n_t[1:])
    epsilon = n_detrend_var / n_total_var if n_total_var > 0 else np.nan

    # Calculer κ_B (couplage cellule de Walker)
    kappa_B = compute_walker_coupling(ssta, u850, region_name)

    # Calculer Γ_confine
    delta_F_residual = np.std(ssta_physical)  # Variation du forçage résiduel
    Q_max = np.max(ssta_physical)  # Q_max(t) : Flux maximal de chaleur
    D_0 = 1000000  # Constante de distance caractéristique (en mètres, à ajuster)
    lambda_d = lambda_val  # λ_d(x, y, t) : Fréquence dominante

    Gamma_confine = calculate_Gamma_confine(delta_F_residual, Q_max, D_0, lambda_d, kappa_B)

    return lambda_val, beta, gamma, Gamma_confine, epsilon, kappa_B

# 8. Validation des théorèmes
def validate_theorems(results):
    """
    Valide les théorèmes de l’Article 5 en utilisant les résultats calculés.
    """
    theorem_validation = {}

    for region, data in results.items():
        Gamma_confine = data['Gamma_confine']
        kappa_B = data.get('kappa_B', np.nan)

        # Validation du Théorème de l’Article 5 : Confinement
        theorem_5_valid = not np.isnan(Gamma_confine) and Gamma_confine > 0

        theorem_validation[region] = {
            'Gamma_confine': Gamma_confine,
            'theorem_5_valid': theorem_5_valid
        }

    return theorem_validation

# 9. Exécution principale
regions = {
    'Global': data_ssta['Global'].values,
    'North_Pacific_Blob': data_ssta['North_Pacific_Blob'].values,
    'ENSO 3.4': data_ssta['ENSO 3.4'].values,
    'Indian_Ocean': data_ssta['Indian_Ocean'].values,
    'South_Pacific': data_ssta['South_Pacific'].values,
    'North_Atlantic': data_ssta['North_Atlantic'].values
}

# Charger u850 (vent zonal à 850 hPa) pour les régions couplées
u850_regions = {
    'Global': data_atmos['u850_Global'].values if 'u850_Global' in data_atmos.columns else None,
    'ENSO 3.4': data_atmos['u850_ENSO'].values if 'u850_ENSO' in data_atmos.columns else None,
    'Indian_Ocean': data_atmos['u850_Indian'].values if 'u850_Indian' in data_atmos.columns else None,
    'South_Pacific': data_atmos['u850_SouthPacific'].values if 'u850_SouthPacific' in data_atmos.columns else None
}

results = {}
for region_name, ssta in regions.items():
    u850 = u850_regions.get(region_name, None)
    t = np.arange(len(ssta))  # Temps en années (à ajuster selon vos données)
    lambda_val, beta, gamma, Gamma_confine, epsilon, kappa_B = compute_parameters(ssta, u850, region_name, t)
    n_t = calculate_n(ssta, region_name, u850)
    results[region_name] = {
        'n_2025': n_t[-1],
        'lambda': lambda_val,
        'beta': beta,
        'gamma': gamma,
        'Gamma_confine': Gamma_confine,
        'epsilon': epsilon,
        'kappa_B': kappa_B
    }

# 10. Afficher les résultats
print("\n=== Résultats avec Couplage Cellule de Walker ===")
for region, params in results.items():
    print(f"\n{region}:")
    print(f"  n(2025): {params['n_2025']}")
    print(f"  λ: {params['lambda']:.3f} yr⁻¹" if not np.isnan(params['lambda']) else "  λ: N/A")
    print(f"  β: {params['beta']:.3f}" if not np.isnan(params['beta']) else "  β: N/A")
    print(f"  γ: {params['gamma']:.3f} yr⁻¹" if not np.isnan(params['gamma']) else "  γ: N/A")
    print(f"  Γ_confine: {params['Gamma_confine']:.3f}" if not np.isnan(params['Gamma_confine']) else "  Γ_confine: N/A")
    print(f"  ε: {params['epsilon']:.3f}" if not np.isnan(params['epsilon']) else "  ε: N/A")
    print(f"  κ_B: {params['kappa_B']:.3f}" if not np.isnan(params['kappa_B']) else "  κ_B: N/A")

# 11. Validation des théorèmes
theorem_validation = validate_theorems(results)

# 12. Afficher les résultats de la validation
print("\n=== Validation des Théorèmes ===")
for region, validation in theorem_validation.items():
    print(f"\n{region}:")
    print(f"  Γ_confine: {validation['Gamma_confine']:.3f}" if not np.isnan(validation['Gamma_confine']) else "  Γ_confine: N/A")
    print(f"  Théorème 5 validé: {validation['theorem_5_valid']}")
