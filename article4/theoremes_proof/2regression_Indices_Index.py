import statsmodels.api as sm

# Préparer les données pour la régression
X = data[['lambda', 'beta', 'gamma', 'kappa_B', 'D_max']]
X = sm.add_constant(X)  # Ajouter une constante pour l'ordonnée à l'origine
y = data['ENSO']

# Ajuster le modèle de régression
model = sm.OLS(y, X).fit()

# Afficher les résultats
print(model.summary())
