lambda_val = data['lambda']

plt.figure(figsize=(12, 6))
plt.plot(data['Year'], n_t, label='n(t)')
plt.plot(data['Year'], lambda_val, label='lambda')
plt.title('Évolution de n(t) et lambda')
plt.xlabel('Année')
plt.ylabel('Valeur')
plt.legend()
plt.grid(True)
plt.show()
