import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, detrend, find_peaks

# 1. Charger les données
data = pd.read_excel(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\SSTa.xlsx")
print("Colonnes disponibles:", data.columns)

# 2. Paramètres régionaux ajustés pour toutes les régions
regional_params = {
    'Global': {
        'inc': 0.10,  # Seuil réduit pour capturer les petites anomalies globales
        'dec': -0.08,  # Seuil asymétrique pour l'hystérésis
        'cooling_years': 3,  # Persistance plus longue pour le global (réchauffement climatique)
        'saturation_band': (1/100, 1/30)  # Bande de saturation globale (30-100 ans)
    },
    'ENSO 3.4': {
        'inc': 0.50,
        'dec': -0.50,
        'cooling_years': 1,
        'saturation_band': (1/7, 1/2)  # Bande ENSO standard
    },
    'North_Pacific_Blob': {
        'inc': 0.20,
        'dec': -0.15,
        'cooling_years': 2,
        'saturation_band': (1/30, 1/10)  # Bande PDO
    },
    'Indian_Ocean': {
        'inc': 0.12,  # Seuil réduit pour capturer les anomalies modérées
        'dec': -0.10,
        'cooling_years': 2,
        'saturation_band': (1/50, 1/20)  # Bande IOD + trend long
    },
    'South_Pacific': {
        'inc': 0.15,  # Seuil ajusté pour la SPCZ
        'dec': -0.12,
        'cooling_years': 2,
        'saturation_band': (1/40, 1/15)  # Bande SPCZ (15-40 ans)
    },
    'North_Atlantic': {
        'inc': 0.12,  # Seuil réduit pour l'AMO
        'dec': -0.10,
        'cooling_years': 2,
        'saturation_band': (1/50, 1/20)  # Bande AMO
    }
}

# 3. Décomposition spectrale améliorée pour toutes les régions
def decompose_ssta(ssta, region_name, fs=1.0):
    f, Pxx = welch(ssta, fs=fs, nperseg=len(ssta)//2)
    nyq = 0.5 * fs  # Fréquence de Nyquist

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

    # Bande de saturation spécifique à la région
    low_sat, high_sat = regional_params[region_name]['saturation_band']
    low_sat_norm = max(1e-6, low_sat / nyq)
    high_sat_norm = min(0.49, high_sat / nyq)

    if low_sat_norm >= high_sat_norm:
        b_sat, a_sat = butter(4, low_sat_norm, btype='low')
        ssta_sat = filtfilt(b_sat, a_sat, ssta)
    else:
        b_sat, a_sat = butter(4, [low_sat_norm, high_sat_norm], btype='band')
        ssta_sat = filtfilt(b_sat, a_sat, ssta)

    # Trend très long (>100 ans) - optionnel
    cutoff_trend = max(1e-6, (1/100) / nyq)
    b_trend, a_trend = butter(4, cutoff_trend, btype='low')
    ssta_trend = filtfilt(b_trend, a_trend, ssta)

    return ssta_enso, ssta_pdo, ssta_sat, ssta_trend

# 4. Calcul de n(t) avec inclusion des bandes de saturation
def calculate_n(ssta, region_name):
    params = regional_params[region_name]
    inc = params['inc']
    dec = params['dec']
    cooling_years = params['cooling_years']

    ssta_enso, ssta_pdo, ssta_sat, _ = decompose_ssta(ssta, region_name)
    ssta_physical = ssta_enso + ssta_pdo + ssta_sat

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

# 5. Calcul des paramètres spectraux (λ, β, γ, ε)
def compute_parameters(ssta, region_name):
    ssta_enso, ssta_pdo, ssta_sat, ssta_trend = decompose_ssta(ssta, region_name)
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

    n_t = calculate_n(ssta, region_name)
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

    return lambda_val, beta, gamma, epsilon, n_t

# 6. Analyser chaque région avec les nouvelles bandes de saturation
regions = {col: data[col].values for col in data.columns if col != 'Year' and col in regional_params}

# Créer un DataFrame pour stocker les résultats
results_df = pd.DataFrame(columns=['Region', 'n_2025', 'lambda', 'beta', 'gamma', 'epsilon'])

results = {}

for region_name, ssta in regions.items():
    lambda_val, beta, gamma, epsilon, n_t = compute_parameters(ssta, region_name)
    results[region_name] = {
        'n_2025': n_t[-1],
        'lambda': lambda_val,
        'beta': beta,
        'gamma': gamma,
        'epsilon': epsilon,
        'n_t': n_t
    }

    # Ajouter les résultats au DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Region': region_name,
        'n_2025': n_t[-1],
        'lambda': lambda_val,
        'beta': beta,
        'gamma': gamma,
        'epsilon': epsilon
    }])], ignore_index=True)

# 7. Sauvegarder les résultats dans un fichier Excel
import os

output_path = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\results_spectral_parameters.xlsx"

# 1. On s'assure que le DataFrame n'est pas vide
if not results_df.empty:
    try:
        # 2. Tentative de sauvegarde
        print(f"Tentative de sauvegarde vers : {output_path}")
        results_df.to_excel(output_path, index=False, engine='openpyxl')
        
        # 3. Vérification immédiate sur le disque
        if os.path.exists(output_path):
            print(f"✅ SUCCÈS : Le fichier a été créé ({os.path.getsize(output_path)} octets).")
        else:
            print("❌ ERREUR : La fonction n'a pas renvoyé d'erreur, mais le fichier est introuvable.")
            
    except PermissionError:
        print("❌ ERREUR : Le fichier est probablement ouvert dans Excel. Fermez-le et réessayez.")
    except Exception as e:
        print(f"❌ ERREUR inattendue lors de la sauvegarde : {e}")
else:
    print("⚠️ ATTENTION : Le DataFrame est vide, rien à sauvegarder.")

# 8. Afficher les résultats
print("\n=== Résultats par Région (Avec bandes de saturation spécifiques) ===")
for region, params in results.items():
    print(f"\n{region}:")
    print(f"  n(2025): {params['n_2025']}")
    print(f"  λ: {params['lambda']:.3f} yr⁻¹" if not np.isnan(params['lambda']) else "  λ: N/A")
    print(f"  β: {params['beta']:.3f}" if not np.isnan(params['beta']) else "  β: N/A")
    print(f"  γ: {params['gamma']:.3f} yr⁻¹" if not np.isnan(params['gamma']) else "  γ: N/A")
    print(f"  ε: {params['epsilon']:.3f}" if not np.isnan(params['epsilon']) else "  ε: N/A")

# 9. Visualisation améliorée pour toutes les régions
for region_name, ssta in regions.items():
    ssta_enso, ssta_pdo, ssta_sat, ssta_trend = decompose_ssta(ssta, region_name)
    n_t = results[region_name]['n_t']

    plt.figure(figsize=(12, 10))

    # SSTa brutes
    plt.subplot(3, 1, 1)
    plt.plot(data['Year'], ssta, label='SSTa Brutes', color='gray', alpha=0.5)
    plt.title(f'{region_name}: SSTa Brutes')

    # Composantes spectrales
    plt.subplot(3, 1, 2)
    plt.plot(data['Year'], ssta_enso, label='ENSO (2-7 ans)', color='blue', alpha=0.7)
    plt.plot(data['Year'], ssta_pdo, label='PDO/AMO (10-30 ans)', color='green', alpha=0.7)
    plt.plot(data['Year'], ssta_sat, label=f'Saturation ({1/regional_params[region_name]["saturation_band"][1]:.0f}-{1/regional_params[region_name]["saturation_band"][0]:.0f} ans)', color='red')
    plt.title(f'{region_name}: Composantes Spectrales')
    plt.legend()

    # n(t) calculé
    plt.subplot(3, 1, 3)
    plt.plot(data['Year'][1:], n_t[1:], label='n(t)', color='purple')
    plt.title(f'{region_name}: Équilibres Cumulés n(t)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{region_name}_analysis_final.png')
    plt.close()