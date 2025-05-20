import os
import numpy as np
import random
import nibabel as nib

# Parámetros
tractografia = "CC.tck"
fod = "/home/paulinabr/Escritorio/Tesis/ISMRM_2023_b3000_ODF.nii"  
SS = [500]  # cantidad de streamlines por subconjunto
tau = 10  # factor para calcular k = tau * M / n

# Cargar tractografía
tractogram = nib.streamlines.load(tractografia)
streamlines = tractogram.streamlines
M = len(streamlines)

# Inicializar P y N con ceros
P = np.zeros(M, dtype=int)
N = np.zeros(M, dtype=int)

# Crear carpeta temporal
os.makedirs("subsets", exist_ok=True)

for n in SS:
    Pn = np.zeros(M, dtype=int)
    Nn = np.zeros(M, dtype=int)
    k = int(tau * M / n)  #numero de subconjuntos
    
    for i in range(k):
        indices = sorted(random.sample(range(M), n)) #seleccionar aleatoriamente subconjunto de n tractos
        # nombres archivos temporales
        # subconjunto de tractos:
        subset_name = f"subsets/subset_{n}_{i}.tck"
        # archivo despues de aplicar sift al subconjunto:
        filtered_name = f"subsets/subset_{n}_{i}_sift.tck"
        
        # Guardar subconjunto temporal
        subset_streamlines = [streamlines[idx] for idx in indices] #toma los tractos usando los indices
        subset_tractogram = nib.streamlines.Tractogram(subset_streamlines, affine_to_rasmm=np.eye(4))
        nib.streamlines.save(subset_tractogram, subset_name) #guardar en nuevo archivo tck

        # Ejecutar SIFT
        os.system(f"tcksift -force {subset_name} {fod} {filtered_name}")

        # Cargar tractografía filtrada
        filtered_tractogram = nib.streamlines.load(filtered_name) # tractos validos por sift
        filtered_streamlines = filtered_tractogram.streamlines #se extraen en forma de lista
        

        # Comparar streamlines aceptados vs rechazados
        accepted_set = set() #almacenar indices de tractos aceptados
        for fs in filtered_streamlines:
            # comparar con el subconjunto original
            for j, s in enumerate(subset_streamlines):
                if np.array_equal(fs, s):
                    accepted_set.add(j)
                    break
        # revisar si se encuentra en la lista de tractos aceptados        
        for j, idx in enumerate(indices):
            if j in accepted_set:
                Pn[idx] += 1
            else:
                Nn[idx] += 1
    # si fue aceptado, aumenta un voto positivo 
    P += Pn
    N += Nn

# Guardar votos finales
np.savetxt("votos_positivos.txt", P, fmt="%d")
np.savetxt("votos_negativos.txt", N, fmt="%d")

