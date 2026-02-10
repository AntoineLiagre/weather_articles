import matplotlib.pyplot as plt

# Visualiser TEPRAT
plt.figure(figsize=(12, 6))
TEPRAT.plot()
plt.title('TEPRAT (Variance de la Fonction de Courant)')
plt.xlabel('Temps')
plt.ylabel('Variance')
plt.grid(True)
plt.show()
