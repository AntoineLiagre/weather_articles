import os
import pygrib
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, find_peaks, detrend

# Configuration des chemins de fichieer test commit
output_dir = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data"
os.makedirs(output_dir, exist_ok=True)

# 1. Charger les données SSTa
ssta_file_path = os.path.join(output_dir, "SSTa.xlsx")
if not os.path.exists(ssta_file_path):
    raise FileNotFoundError(f"Le fichier {ssta_file_path} est introuvable.")

data_ssta = pd.read_excel(ssta_file_path)
years = data_ssta['Year'].values

# Charger les valeurs de référence
reference_values_path = os.path.join(output_dir, "reference_values_DQKL_1940_1970.csv")
if not os.path.exists(reference_values_path):
    raise FileNotFoundError(f"Le fichier {reference_values_path} est introuvable.")

reference_values_df = pd.read_csv(reference_values_path)

# Extraire les valeurs de référence
D0_ref = reference_values_df.loc[reference_values_df['Parameter'] == 'D0', 'Value'].values[0]
Q0_ref = reference_values_df.loc[reference_values_df['Parameter'] == 'Q0', 'Value'].values[0]
kappa_B0_ref = reference_values_df.loc[reference_values_df['Parameter'] == 'kappa_B0', 'Value'].values[0]
lambda_0_ref = reference_values_df.loc[reference_values_df['Parameter'] == 'lambda_0', 'Value'].values[0]

# 2. Paramètres régionaux (avec les valeurs de référence)
regional_params = {
    'Global': {
        'inc': 0.10, 'dec': -0.08, 'cooling_years': 3,
        'saturation_band': (1/100, 1/30),
        'walker_coupling': True,
        'D0': D0_ref, 'gamma': 0.1, 'Tv0': 1, 'q': 2, 'P': 1, 'R': 1, 'tc': 10, 'lambda_d': 0.1,
        'Q0': Q0_ref, 'alpha': 0.1, 'N0': 1, 'p': 2, 'lambda': lambda_0_ref
    },
    'ENSO 3.4': {
        'inc': 0.50, 'dec': -0.50, 'cooling_years': 1,
        'saturation_band': (1/7, 1/2),
        'walker_coupling': True,
        'D0': D0_ref, 'gamma': 0.1, 'Tv0': 1, 'q': 2, 'P': 1, 'R': 1, 'tc': 10, 'lambda_d': 0.1,
        'Q0': Q0_ref, 'alpha': 0.1, 'N0': 1, 'p': 2, 'lambda': lambda_0_ref
    },
    'North_Pacific_Blob': {
        'inc': 0.20, 'dec': -0.15, 'cooling_years': 2,
        'saturation_band': (1/30, 1/10),
        'walker_coupling': False,
        'D0': D0_ref, 'gamma': 0.1, 'Tv0': 1, 'q': 2, 'P': 1, 'R': 1, 'tc': 10, 'lambda_d': 0.1,
        'Q0': Q0_ref, 'alpha': 0.1, 'N0': 1, 'p': 2, 'lambda': lambda_0_ref
    },
    'Indian_Ocean': {
        'inc': 0.12, 'dec': -0.10, 'cooling_years': 2,
        'saturation_band': (1/50, 1/20),
        'walker_coupling': True,
        'D0': D0_ref, 'gamma': 0.1, 'Tv0': 1, 'q': 2, 'P': 1, 'R': 1, 'tc': 10, 'lambda_d': 0.1,
        'Q0': Q0_ref, 'alpha': 0.1, 'N0': 1, 'p': 2, 'lambda': lambda_0_ref
    },
    'South_Pacific': {
        'inc': 0.15, 'dec': -0.12, 'cooling_years': 2,
        'saturation_band': (1/40, 1/15),
        'walker_coupling': True,
        'D0': D0_ref, 'gamma': 0.1, 'Tv0': 1, 'q': 2, 'P': 1, 'R': 1, 'tc': 10, 'lambda_d': 0.1,
        'Q0': Q0_ref, 'alpha': 0.1, 'N0': 1, 'p': 2, 'lambda': lambda_0_ref
    },
    'North_Atlantic': {
        'inc': 0.12, 'dec': -0.10, 'cooling_years': 2,
        'saturation_band': (1/50, 1/20),
        'walker_coupling': False,
        'D0': D0_ref, 'gamma': 0.1, 'Tv0': 1, 'q': 2, 'P': 1, 'R': 1, 'tc': 10, 'lambda_d': 0.1,
        'Q0': Q0_ref, 'alpha': 0.1, 'N0': 1, 'p': 2, 'lambda': lambda_0_ref
    }
}

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
    # Exemple de masques régionaux (ajustez selon vos données)
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
        year_mask = [int(str(t)[:4]) == year for t in times]
        if np.any(year_mask):
            data_year = data[year_mask]
            if region_mask is not None:
                data_year = data_year[:, region_mask]
            annual_mean = np.nanmean(data_year)
            annual_means.append(annual_mean)
    return np.array(annual_means)

# 7. Calcul des moyennes annuelles globales et régionales
# Supposons que les données GRIB ont une forme (temps, latitude, longitude)
# On extrait les latitudes et longitudes à partir des données
lats = np.linspace(-90, 90, u850_data.shape[1])  # Exemple : 41 latitudes
lons = np.linspace(-180, 180, u850_data.shape[2])  # Exemple : 1440 longitudes

regional_masks = create_regional_masks(lats, lons)

u850_global_annual_means = calculate_annual_means(u850_data, u850_times)
u850_pacific_annual_means = calculate_annual_means(u850_data, u850_times, regional_masks['pacific'])
u850_atlantic_annual_means = calculate_annual_means(u850_data, u850_times, regional_masks['atlantic'])
u850_indian_annual_means = calculate_annual_means(u850_data, u850_times, regional_masks['indian'])

# 8. Décomposition spectrale
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

# 9. Calcul de n(t) avec couplage à la cellule de Walker
def calculate_n(ssta, region_name, u850=None):
    params = regional_params[region_name]
    base_inc = params['inc']
    base_dec = params['dec']
    cooling_years = params['cooling_years']

    ssta_enso, ssta_pdo, ssta_sat = decompose_ssta(ssta, region_name)
    ssta_physical = ssta_enso + ssta_pdo + ssta_sat

    yoY_diff = np.diff(ssta_physical)

    if params['walker_coupling'] and u850 is not None:
        u850_norm = (u850 - np.nanmean(u850)) / (np.nanstd(u850) + 1e-6)
        inc_array = base_inc * (1 + 0.2 * u850_norm)
        dec_array = base_dec * (1 + 0.2 * u850_norm)
    else:
        inc_array = np.full(len(yoY_diff), base_inc)
        dec_array = np.full(len(yoY_diff), base_dec)

    n = 0
    n_list = [0]
    cooling_counter = 0

    for i, diff in enumerate(yoY_diff):
        if diff >= inc_array[i]:
            n += 1
            cooling_counter = 0
        elif diff <= dec_array[i]:
            cooling_counter += 1
            if cooling_counter >= cooling_years and n > 0:
                n -= 1
                cooling_counter = 0
        else:
            cooling_counter = 0
        n_list.append(n)

    return np.array(n_list)

# 10. Calcul du couplage cellule de Walker (κ_B)
def compute_walker_coupling(ssta, u850, region_name):
    if not regional_params[region_name]['walker_coupling'] or u850 is None:
        return np.nan

    try:
        ssta_detrend = detrend(ssta)
        u850_detrend = detrend(u850)
    except Exception as e:
        print(f"Erreur lors du detrend: {e}")
        return np.nan

    covariance = np.cov(ssta_detrend, u850_detrend)[0, 1]
    std_ssta = np.std(ssta_detrend)
    std_u850 = np.std(u850_detrend)

    if std_ssta > 0 and std_u850 > 0:
        kappa_B = covariance / (std_ssta * std_u850)
    else:
        kappa_B = np.nan

    return kappa_B

# 11. Calcul de D_max
def compute_D_max(t, delta_Tv, region_name):
    params = regional_params[region_name]
    D0 = params['D0']
    gamma = params['gamma']
    Tv0 = params['Tv0']
    q = params['q']
    P = params['P']
    R = params['R']
    tc = params['tc']
    lambda_d = params['lambda_d']

    term1 = D0 * np.exp(-gamma * t)
    term2 = 1 / (1 + (delta_Tv / Tv0) ** q)
    term3 = P / (R * delta_Tv)

    mask = t >= tc
    term4 = np.ones_like(t, dtype=float)
    if np.any(mask):
        term4[mask] = 1 - np.exp(-lambda_d * (t[mask] - tc))

    D_max = term1 * term2 * term3 * term4
    return D_max

# 12. Calcul de Q_max
def compute_Q_max(t, delta_N, region_name):
    params = regional_params[region_name]
    Q0 = params['Q0']
    alpha = params['alpha']
    N0 = params['N0']
    p = params['p']
    tc = params['tc']
    lambda_val = params['lambda']

    term1 = Q0 * np.exp(-alpha * t)
    term2 = 1 / (1 + (delta_N / N0) ** p)

    mask = t >= tc
    term3 = np.ones_like(t, dtype=float)
    if np.any(mask):
        term3[mask] = 1 - np.exp(-lambda_val * (t[mask] - tc))

    Q_max = term1 * term2 * term3
    return Q_max

# 13. Calcul des paramètres spectraux (λ, β, γ, κ_B, D_max, Q_max)
def compute_parameters(ssta, u850, region_name, t, years, u850_regional=None):
    ssta_enso, ssta_pdo, ssta_sat = decompose_ssta(ssta, region_name)
    ssta_physical = ssta_enso + ssta_pdo + ssta_sat

    n_seg = max(4, len(ssta_physical)//2)
    f, Pxx = welch(ssta_physical, fs=1.0, nperseg=n_seg)

    low_sat, high_sat = regional_params[region_name]['saturation_band']
    band_mask = (f >= low_sat) & (f <= high_sat)

    lambda_val = np.nan
    if np.any(band_mask):
        peaks, _ = find_peaks(Pxx[band_mask], height=0.1 * np.max(Pxx[band_mask]))
        if len(peaks) > 0:
            lambda_val = f[band_mask][peaks[0]]
        else:
            lambda_val = np.mean(f[band_mask])

    total_power = np.sum(Pxx)
    beta = np.sum(Pxx[band_mask]) / total_power if total_power > 0 else np.nan

    n_t = calculate_n(ssta, region_name, u850)

    gamma = np.nan
    epsilon = np.nan

    if len(n_t) > 2:
        try:
            data_to_detrend = n_t[1:]
            n_detrend = detrend(data_to_detrend)

            n_seg_n = max(4, len(n_detrend)//2)
            f_n, Pxx_n = welch(n_detrend, fs=1.0, nperseg=n_seg_n)

            low_mask = (f_n >= 0.01) & (f_n <= 0.1)
            if np.any(low_mask):
                gamma = np.mean(f_n[low_mask])

            n_total_var = np.var(data_to_detrend)
            epsilon = np.var(n_detrend) / n_total_var if n_total_var > 0 else np.nan
        except Exception as e:
            print(f"Skipping detrend for {region_name} due to: {e}")

    # Calcul de kappa_B pour chaque région
    kappa_B_global = compute_walker_coupling(ssta_physical, u850, region_name)
    kappa_B_regional = np.nan
    if u850_regional is not None:
        kappa_B_regional = compute_walker_coupling(ssta_physical, u850_regional, region_name)

    delta_Tv = np.std(ssta_physical)

    t_val = t[-1] if hasattr(t, '__len__') else t
    D_max = compute_D_max(t_val, delta_Tv, region_name)

    delta_N = n_t[-1] - n_t[0]
    Q_max = compute_Q_max(t_val, delta_N, region_name)

    res_len = len(years) - 1
    return pd.DataFrame({
        'Year': years[1:],
        'n_t': n_t[1:],
        'lambda': [lambda_val] * res_len,
        'beta': [beta] * res_len,
        'gamma': [gamma] * res_len,
        'epsilon': [epsilon] * res_len,
        'kappa_B_global': [kappa_B_global] * res_len,
        'kappa_B_regional': [kappa_B_regional] * res_len,
        'D_max': [D_max] * res_len,
        'Q_max': [Q_max] * res_len
    })

# 14. Exécution principale
def main():
    regions_cols = [col for col in data_ssta.columns if col != 'Year']

    all_results = {}
    output_path = os.path.join(output_dir, "annual_spectral_parameters.csv")

    for region_name in regions_cols:
        print(f"Processing {region_name}...")
        ssta = data_ssta[region_name].values
        t = np.arange(len(ssta))

        try:
            # Calcul des paramètres avec les indices globaux et régionaux
            res_global = compute_parameters(ssta, u850_global_annual_means, region_name, t, years)
            res_pacific = compute_parameters(ssta, u850_global_annual_means, region_name, t, years, u850_pacific_annual_means)
            res_atlantic = compute_parameters(ssta, u850_global_annual_means, region_name, t, years, u850_atlantic_annual_means)
            res_indian = compute_parameters(ssta, u850_global_annual_means, region_name, t, years, u850_indian_annual_means)

            # Ajout des résultats au dictionnaire
            all_results[f"{region_name}_global"] = res_global
            all_results[f"{region_name}_pacific"] = res_pacific
            all_results[f"{region_name}_atlantic"] = res_atlantic
            all_results[f"{region_name}_indian"] = res_indian
        except Exception as e:
            print(f"❌ Erreur lors du calcul pour {region_name}: {e}")

    if all_results:
        try:
            # On combine tous les DataFrames des régions en un seul
            combined_df = pd.concat(all_results.values(), keys=all_results.keys())
            combined_df.to_csv(output_path)
            print(f"\n✅ Succès ! Fichier sauvegardé ici : {output_path}")
        except Exception as e:
            print(f"❌ Échec de la sauvegarde CSV : {e}")
    else:
        print("⚠️ Aucun résultat à sauvegarder. Vérifiez vos données d'entrée.")

if __name__ == "__main__":
    main()
