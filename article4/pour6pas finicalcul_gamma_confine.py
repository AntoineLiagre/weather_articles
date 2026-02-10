import pygrib
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, find_peaks, detrend
from scipy.stats import linregress

# 1. Charger les données SSTa
data_ssta = pd.read_excel(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\SSTa.xlsx")
print("Colonnes SSTa:", data_ssta.columns)

# 2. Paramètres régionaux
regional_params = {
    'Global': {
        'inc': 0.10, 'dec': -0.08, 'cooling_years': 3,
        'saturation_band': (1/100, 1/30),
        'walker_coupling': True
    },
    'ENSO 3.4': {
        'inc': 0.50, 'dec': -0.50, 'cooling_years': 1,
        'saturation_band': (1/7, 1/2),
        'walker_coupling': True
    },
    'North_Pacific_Blob': {
        'inc': 0.20, 'dec': -0.15, 'cooling_years': 2,
        'saturation_band': (1/30, 1/10),
        'walker_coupling': False
    },
    'Indian_Ocean': {
        'inc': 0.12, 'dec': -0.10, 'cooling_years': 2,
        'saturation_band': (1/50, 1/20),
        'walker_coupling': True
    },
    'South_Pacific': {
        'inc': 0.15, 'dec': -0.12, 'cooling_years': 2,
        'saturation_band': (1/40, 1/15),
        'walker_coupling': True
    },
    'North_Atlantic': {
        'inc': 0.12, 'dec': -0.10, 'cooling_years': 2,
        'saturation_band': (1/50, 1/20),
        'walker_coupling': False
    }
}

# 3. Fonction pour lire les données GRIB
def load_grib_data(file_path, short_name, level=None):
    try:
        grbs = pygrib.open(file_path)
        for grb in grbs:
            if grb.shortName == short_name and (level is None or grb.level == level):
                data = grb.values
                grbs.close()
                return data
        grbs.close()
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {file_path}: {str(e)}")
        return None

# 4. Charger les données u850 directement depuis les fichiers GRIB
u850_regions = {
    'Global': load_grib_data(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\1f6ad324afe41a513fb0751df9d5d5d2.grib", 'u', level=850),
    'ENSO 3.4': load_grib_data(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\1f6ad324afe41a513fb0751df9d5d5d2.grib", 'u', level=850),
    'Indian_Ocean': load_grib_data(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\1f6ad324afe41a513fb0751df9d5d5d2.grib", 'u', level=850),
    'South_Pacific': load_grib_data(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\1f6ad324afe41a513fb0751df9d5d5d2.grib", 'u', level=850)
}

# 5. Décomposition spectrale
def decompose_ssta(ssta, region_name, fs=1.0):
    f, Pxx = welch(ssta, fs=fs, nperseg=len(ssta)//2)
    nyq = 0.5 * fs

    low_enso = max(1e-6, (1/7) / nyq)
    high_enso = min(0.49, (1/2) / nyq)
    b_enso, a_enso = butter(4, [low_enso, high_enso], btype='band')
    ssta_enso = filtfilt(b_enso, a_enso, ssta)

    low_pdo = max(1e-6, (1/30) / nyq)
    high_pdo = min(0.49, (1/10) / nyq)
    b_pdo, a_pdo = butter(4, [low_pdo, high_pdo], btype='band')
    ssta_pdo = filtfilt(b_pdo, a_pdo, ssta)

    low_sat, high_sat = regional_params[region_name]['saturation_band']
    low_sat_norm = max(1e-6, low_sat / nyq)
    high_sat_norm = min(0.49, high_sat / nyq)

    if low_sat_norm >= high_sat_norm:
        b_sat, a_sat = butter(4, low_sat_norm, btype='low')
    else:
        b_sat, a_sat = butter(4, [low_sat_norm, high_sat_norm], btype='band')
    ssta_sat = filtfilt(b_sat, a_sat, ssta)

    return ssta_enso, ssta_pdo, ssta_sat

# 6. Calcul de n(t) avec couplage à la cellule de Walker
def calculate_n(ssta, region_name, u850=None):
    params = regional_params[region_name]
    inc = params['inc']
    dec = params['dec']
    cooling_years = params['cooling_years']

    ssta_enso, ssta_pdo, ssta_sat = decompose_ssta(ssta, region_name)
    ssta_physical = ssta_enso + ssta_pdo + ssta_sat

    if regional_params[region_name]['walker_coupling'] and u850 is not None:
        u850_norm = (u850 - np.nanmean(u850)) / (np.nanstd(u850) + 1e-6)
        inc *= (1 + 0.2 * u850_norm)
        dec *= (1 + 0.2 * u850_norm)

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

# 7. Calcul du couplage cellule de Walker (κ_B)
def compute_walker_coupling(ssta, u850, region_name):
    # Vérification si le couplage est activé
    if not regional_params.get(region_name, {}).get('walker_coupling', False) or u850 is None:
        return 0.0

    # FORCE LA CONVERSION EN ARRAY NUMPY
    s_arr = np.atleast_1d(np.asarray(ssta))
    u_arr = np.atleast_1d(np.asarray(u850))

    # SI C'EST UN SCALAIRE (UN SEUL NOMBRE), DETREND EST IMPOSSIBLE
    if s_arr.size <= 1 or u_arr.size <= 1:
        # On retourne 0 au lieu de faire planter le script
        return 0.0

    try:
        # On force l'axe 0 pour éviter le tuple index out of range
        ssta_detrend = detrend(s_arr, axis=0)
        u850_detrend = detrend(u_arr, axis=0)
        
        # Calcul de corrélation simple pour kappa_B
        # (La covariance sur des données normalisées/detrendées)
        if len(ssta_detrend) != len(u850_detrend):
            min_l = min(len(ssta_detrend), len(u850_detrend))
            ssta_detrend, u850_detrend = ssta_detrend[:min_l], u850_detrend[:min_l]
            
        correlation = np.corrcoef(ssta_detrend, u850_detrend)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    except Exception:
        # En cas d'autre erreur bizarre, on renvoie 0 pour ne pas bloquer l'Excel
        return 0.0
    
# 8. Calcul des paramètres spectraux (λ, β, γ, Γ_confine, ε, κ_B)
def compute_parameters(ssta, u850, region_name, t):
    ssta_enso, ssta_pdo, ssta_sat = decompose_ssta(ssta, region_name)
    ssta_physical = ssta_enso + ssta_pdo + ssta_sat

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

    total_power = np.sum(Pxx)
    if total_power > 0:
        beta = np.sum(Pxx[band_mask]) / total_power
    else:
        beta = np.nan

    n_t = calculate_n(ssta, region_name, u850)
    n_detrend = detrend(n_t[1:])
    f_n, Pxx_n = welch(n_detrend, fs=1.0, nperseg=len(n_detrend)//2)
    low_mask = (f_n >= 0.01) & (f_n <= 0.1)
    if low_mask.sum() > 0:
        gamma = np.mean(f_n[low_mask][Pxx_n[low_mask] > 0.1 * np.max(Pxx_n[low_mask])])
    else:
        gamma = np.mean(f_n[low_mask]) if low_mask.sum() > 0 else np.nan

    n_detrend_var = np.var(n_detrend)
    n_total_var = np.var(n_t[1:])
    epsilon = n_detrend_var / n_total_var if n_total_var > 0 else np.nan

    kappa_B = compute_walker_coupling(ssta, u850, region_name)

    delta_F_residual = np.std(ssta_physical)
    Q_max = np.max(ssta_physical)
    D_0 = 1000000
    lambda_d = lambda_val

    Gamma_confine = (abs(delta_F_residual) / (Q_max + D_0 * lambda_d)) * (1 + kappa_B) if (Q_max + D_0 * lambda_d) > 0 else np.nan

    return lambda_val, beta, gamma, Gamma_confine, epsilon, kappa_B

# 9. Exécution principale
regions = {
    'Global': data_ssta['Global'].values,
    'North_Pacific_Blob': data_ssta['North_Pacific_Blob'].values,
    'ENSO 3.4': data_ssta['ENSO 3.4'].values,
    'Indian_Ocean': data_ssta['Indian_Ocean'].values,
    'South_Pacific': data_ssta['South_Pacific'].values,
    'North_Atlantic': data_ssta['North_Atlantic'].values
}

results = {}
results_df = pd.DataFrame(columns=['Region', 'n_2025', 'lambda', 'beta', 'gamma', 'Gamma_confine', 'epsilon', 'kappa_B'])

for region_name, ssta in regions.items():
    u850 = u850_regions.get(region_name, None)
    t = np.arange(len(ssta))
    lambda_val, beta, gamma, Gamma_confine, epsilon, kappa_B = compute_parameters(ssta, u850, region_name, t)
    n_t = calculate_n(ssta, region_name, u850)
    results[region_name] = {
        'n_2025': n_t[-1],
        'lambda': lambda_val,
        'beta': beta,
        'gamma': gamma,
        'Gamma_confine': Gamma_confine,
        'epsilon': epsilon,
        'kappa_B': kappa_B,
        'n_t': n_t
    }

    results_df = pd.concat([results_df, pd.DataFrame([{
        'Region': region_name,
        'n_2025': n_t[-1],
        'lambda': lambda_val,
        'beta': beta,
        'gamma': gamma,
        'Gamma_confine': Gamma_confine,
        'epsilon': epsilon,
        'kappa_B': kappa_B
    }])], ignore_index=True)

# 10. Sauvegarder les résultats dans un fichier Excel
results_df.to_excel(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\results_spectral_parameters.xlsx", index=False)

# 11. Afficher les résultats
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

print("\nLes résultats ont été sauvegardés dans 'results_spectral_parameters.xlsx'.")
