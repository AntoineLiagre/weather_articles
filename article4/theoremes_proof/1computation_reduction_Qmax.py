# Filtrer les données avant et après 1980
q_max_before_1980 = merged_data[merged_data['Year'] < 1980]['Q_max']
q_max_after_1980 = merged_data[merged_data['Year'] >= 1980]['Q_max']

# Calculer les moyennes
mean_before = q_max_before_1980.mean()
mean_after = q_max_after_1980.mean()

# Calculer la réduction en pourcentage
reduction = ((mean_before - mean_after) / mean_before) * 100
print(f"Réduction de Q_max depuis 1980: {reduction:.2f}%")
