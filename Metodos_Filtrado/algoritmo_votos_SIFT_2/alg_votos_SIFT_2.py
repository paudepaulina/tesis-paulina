import os
import numpy as np
import random
import nibabel as nib
import subprocess

# Parámetros
tract_file = "CC.tck"
fod_file = "/home/paulinabr/Escritorio/Tesis/ISMRM_2023_b3000_ODF.nii"
SS = [500]
tau = .1
peso_umbral = 0.347 # umbral para considerar aceptado un tracto

# Cargar tractografía completa
tractogram = nib.streamlines.load(tract_file)
streamlines = tractogram.streamlines
M = len(streamlines)

# Inicializar contador de votos
P = np.zeros(M, dtype=int)
N = np.zeros(M, dtype=int)

# Crear carpeta
os.makedirs("subsets_sift2", exist_ok=True)

for n in SS:
    Pn = np.zeros(M,dtype=int)
    Nn = np.zeros(M,dtype=int)
    k = max(1, int(tau * M / n))


    for i in range(k):

    # Seleccionar aleatoriamente los tractos
        indices = sorted(random.sample(range(M), n))
        subset_name = f"subsets_sift2/subset_{n}_{i}.tck"
        pesos_name = f"subsets_sift2/pesos_{n}_{i}.txt"


    # Guardar subconjuntos
        subset_streamlines = [streamlines[idx] for idx in indices]
        subset_tractogram = nib.streamlines.Tractogram(subset_streamlines, affine_to_rasmm=np.eye(4))
        nib.streamlines.save(subset_tractogram,subset_name)

    # Ejecutar SIFT2

        subprocess.run(["tcksift2",subset_name, fod_file, pesos_name, "-force"])

        if not os.path.exists(pesos_name):
            print(f"[ERROR] No se generó {pesos_name}")
            continue

        pesos = np.loadtxt(pesos_name)
        aceptados = set(np.where(pesos > peso_umbral)[0])
    
        for j, idx in enumerate(indices):
            if j in aceptados:
                Pn[idx] += 1
            else:
                Nn[idx] += 1
    P   += Pn
    N   += Nn


# Guardar resultados
print(f"Total de streamlines: {M}, Subconjuntos: {k}")
np.savetxt("votos_positivos_sift2.txt", P, fmt="%d")
np.savetxt("votos_negativos_sift2.txt", N, fmt="%d")
print("Votaciones para SIFT2 finalizadas.")
