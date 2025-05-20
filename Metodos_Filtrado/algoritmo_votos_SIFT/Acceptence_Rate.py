import numpy as np

# Cargar archivos
P = np.loadtxt("votos_positivos.txt")  # Votos positivos
N = np.loadtxt("votos_negativos.txt")  # Votos negativos

# Evitar división por cero
total_votos = P + N
with np.errstate(divide='ignore', invalid='ignore'):
    AR = np.true_divide(P, total_votos)
    AR[~np.isfinite(AR)] = -1  # Para streamlines nunca evaluadas (0/0)

# Clasificación
clases = np.full_like(AR, fill_value="inconclusive", dtype=object)
clases[AR == 1.0] = "positive"
clases[AR == 0.0] = "negative"
clases[AR == -1] = "not_evaluated"  # Opcional: fibras que nunca se evaluaron

# Estadísticas
total = len(AR)
print("Estadísticas de clasificación:")
print(f"Positivas (AR=1): {(clases == 'positive').sum()} fibras ({(clases == 'positive').sum()/total:.2%})")
print(f"Negativas (AR=0): {(clases == 'negative').sum()} fibras ({(clases == 'negative').sum()/total:.2%})")
print(f"Indefinido (0 < AR < 1): {(clases == 'inconclusive').sum()} fibras ({(clases == 'inconclusive').sum()/total:.2%})")
print(f"No evaluadas (AR indefinido): {(clases == 'not_evaluated').sum()} fibras ({(clases == 'not_evaluated').sum()/total:.2%})")

# Guardar resultados
np.savetxt("aceptance_rates.txt", AR, fmt="%.4f")
np.savetxt("clases.txt", clases, fmt="%s")
