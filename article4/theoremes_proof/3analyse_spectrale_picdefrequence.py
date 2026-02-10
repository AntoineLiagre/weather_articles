from scipy.signal import welch
import matplotlib.pyplot as plt

# Calculer le spectre de puissance de n(t)
fs = 1  # Fréquence d'échantillonnage (1 par an)
f, Pxx = welch(n_t.dropna(), fs=fs, nperseg=min(256, len(n_t.dropna())//2))

# Visualiser le spectre de puissance
plt.figure(figsize=(12, 6))
plt.semilogy(f, Pxx)
plt.title('Spectre de Puissance de n(t)')
plt.xlabel('Fréquence (1/an)')
plt.ylabel('Puissance')
plt.grid(True)
plt.show()
