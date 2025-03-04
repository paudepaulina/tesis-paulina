import os
import amico.util
import numpy as np
import commit
from commit import trk2dictionary
import amico
import nibabel as nib
from nibabel.streamlines import Tractogram, TckFile

# 1. Definir archivos de entrada
tractografia = 'CC.tck'
mask = 'ISMRM_2023_b3000_mask.nii'
bval = 'ISMRM_2023_b3000.bval'
bvec = 'ISMRM_2023_b3000.bvec'
img_nii = 'ISMRM_2023_b3000.nii'
parcellation = 'Parcels_free.nii'  

# 2. Definir archivos de salida
connectome_file = 'connectome.csv'
filtered_tract_commit1 = 'CC_filtrado_commit1.tck'
filtered_tract_commit2 = 'CC_filtrado_commit2.tck'

# 3. Convertir bvals y bvecs a esquema
amico.util.fsl2scheme(bval, bvec, 'DWI.scheme')

# 4. Importar la tractografia y determinar el ajuste de las fibras
trk2dictionary.run(
    filename_tractogram=tractografia,
    filename_mask=mask,
    fiber_shift=0.5
)

# 5. ***************** Configurar COMMIT1 **********************
mit = commit.Evaluation('.', '.')
mit.set_verbose(4) # Para mostrar todos los detalles del ajuste
mit.load_data(img_nii, 'DWI.scheme') # Cargar los datos

# 6. En el ejemplo se usa el modelo SZB (Panagiotaki etal., NeuroImage, 2012)
mit.set_model('StickZeppelinBall')
# Se usan los parámetros especificados en el tutorial
mit.model.set(1.7E-3, [], [1.7E-3, 3.0E-3])
mit.generate_kernels(regenerate=True)
mit.load_kernels()

# 7. Construir el modelo con los datos cargados en trk2dictionary
mit.load_dictionary('COMMIT')
# Construcción del operador lineal A, que genera el producto entre
#matriz y vector para dar solución al sistema lineal
mit.set_threads(4)
mit.build_operator()

# 8. Ejecutar COMMIT1: Ajustando el modelo a los datos
mit.fit(tol_fun=1e-3, max_iter=1000)
#Guardar resultados
mit.save_results(path_suffix="_COMMIT1")

# Verificar si se generó el archivo de pesos para evitar errores
pesos_path = "Results_COMMIT1/streamline_weights.txt"
os.makedirs("Results_COMMIT1", exist_ok=True)

if os.path.exists(pesos_path):
    print(f"Pesos generados correctamente en {pesos_path}")
else:
    print("No se generó streamline_weights.txt.")
    print("Extraer manualmente los pesos.")
    #Acceder a los coeficientes estimados
    x_ic, x_ec, x_iso = mit.get_coeffs()
    np.savetxt(pesos_path, x_ic)
    print(f"Pesos guardados manualmente en '{pesos_path}'.")

# 9. Filtrar tractografía con el umbral en pesos de COMMIT1
pesos = np.loadtxt(pesos_path)
umbral_peso = 0.01  # Necesario ajustar el umbral

# Eliminar las fibras con pesos menores al umbral 
indices_fibras_buenas = np.where(pesos > umbral_peso)[0]
print(f"Se eliminarán {len(pesos) - len(indices_fibras_buenas)} fibras con peso < {umbral_peso}")


tracto = nib.streamlines.load(tractografia)
streamlines = tracto.streamlines
streamlines_filtradas = [streamlines[i] for i in indices_fibras_buenas]

nuevo_header_commit1 = tracto.header.copy()
nuevo_header_commit1["count"] = len(streamlines_filtradas)  # Actualizar el número de fibras

tracto_filtrado_commit1 = Tractogram(streamlines_filtradas, affine_to_rasmm=np.eye(4))
tckfile_commit1 = TckFile(tracto_filtrado_commit1, nuevo_header_commit1)
tckfile_commit1.save(filtered_tract_commit1)

print(f"Se guardó la tractografía filtrada por COMMIT1 en '{filtered_tract_commit1}', con {len(streamlines_filtradas)} fibras.")


# **********Implementación de COMMIT2:*****************

# 10. Se genera el conectoma de la tractografia
os.system('dice_connectome_build streamline_weights.txt connectome.csv -t CC.tck -a Parcels_free.nii -f')
# Usando la información, extraemos solo los streamlines que conectan
os.system('dice_tractogram_sort CC.tck Parcels_free.nii fibras_conectan_CM2.tck -f')

# 11. Importar la tractografia de streamlines que conectan
trk2dictionary.run(
    filename_tractogram = 'fibras_conectan_CM2.tck',
    filename_mask= 'ISMRM_2023_b3000_mask.nii',
    fiber_shift = 0.5
)

# Convertir bvals y bvecs a esquema
amico.util.fsl2scheme('ISMRM_2023_b3000.bval', 'ISMRM_2023_b3000.bvec', 'DWI_CM2.scheme')
# Verificar que se generó
if os.path.exists("DWI_CM2.scheme"):
    print("El archivo DWI_CM2.scheme se generó correctamente.")
else:
    raise FileNotFoundError("ERROR: No se pudo generar DWI_CM2.scheme. Verifica bval y bvec.")

# Cargar los datos
mit = commit.Evaluation('-','-')
mit.set_verbose(4)
mit.load_data('ISMRM_2023_b3000.nii', 'DWI_CM2.scheme')


# Usar modelo SZB, siguiendo el tutorial
mit.set_model('StickZeppelinBall')
d_par       = 1.7E-3             # Parallel diffusivity [mm^2/s]
d_perps_zep = []                 # Perpendicular diffusivity(s) [mm^2/s]
d_isos      = [ 1.7E-3, 3.0E-3 ] # Isotropic diffusivity(s) [mm^2/s]
mit.model.set( d_par, d_perps_zep, d_isos )

#Generar kernels
mit.generate_kernels(regenerate=True)
mit.load_kernels()

mit.load_dictionary('COMMIT')
mit.set_threads()
mit.build_operator()

#Ajuste
mit.fit(tol_fun=1e-3,max_iter=1000)
mit.save_results(path_suffix="_COMMIT1")

#Numero de streamlines en cada bundle a partir del conectoma

C = np.loadtxt('connectome.csv', delimiter=',')
C = np.triu( C )
group_size = C[C>0].astype(np.int32)

#Con esta información se crea un arreglo de indices de streamlines que componen cada bundle

tmp = np.insert(np.cumsum(group_size), 0, 0 )
group_idx = np.fromiter( [np.arange(tmp[i],tmp[i+1]) for i in range(len(tmp)-1)], dtype=np.object_)


# Se inicializa un diccionario para los parametros adicionales de la regularizacion lasso
params_IC = {}
params_IC['group_idx'] = group_idx
params_IC['group_weights_cardinality'] = True
params_IC['group_weights_adaptive'] = True

# Termino de regularización 
perc_lambda = 0.00025 # ****Investigar si es necesario ajustar el valor del parámetro*****

mit.set_regularisation(
    regularisers = ('group_lasso', None, None),
    is_nonnegative = (True, True, True),
    lambdas = (perc_lambda, None, None),
    params = (params_IC, None, None)
)

mit.fit(tol_fun = 1e-3, max_iter=1000)
mit.save_results(path_suffix="_COMMIT2")

# 13. Filtrar tractografía con pesos de COMMIT2
pesos_commit2_path = "COMMIT/Results_StickZeppelinBall_COMMIT2/streamline_weights.txt"

if os.path.exists(pesos_commit2_path):
    pesos_commit2 = np.loadtxt(pesos_commit2_path)
    print(f"Pesos de COMMIT2 encontrados en {pesos_commit2_path}")
else:
    print(f"No se encontró el archivo de pesos en {pesos_commit2_path}.")
    print("Contenido de la carpeta de resultados de COMMIT2:")
    print(os.listdir("Results_StickZeppelinBall_COMMIT2"))
    raise FileNotFoundError(f"No se encontró streamline_weights.txt en {pesos_commit2_path}.")

indices_fibras_buenas_commit2 = np.where(pesos_commit2 > 0.01)[0]
# Verificar el tamaño de las listas antes de indexar
if len(indices_fibras_buenas_commit2) == 0:
    raise ValueError("No hay fibras seleccionadas después de COMMIT2")

if np.max(indices_fibras_buenas_commit2) >= len(streamlines_filtradas):
    print(f"índices fuera del rango. Ajustando valores")
    indices_fibras_buenas_commit2 = indices_fibras_buenas_commit2[indices_fibras_buenas_commit2 < len(streamlines_filtradas)]

# Filtrar las fibras después de COMMIT2
streamlines_filtradas_commit2 = [streamlines_filtradas[i] for i in indices_fibras_buenas_commit2]

nuevo_header_commit2 = tracto.header.copy()
nuevo_header_commit2["count"] = len(streamlines_filtradas_commit2)

tracto_filtrado_commit2 = Tractogram(streamlines_filtradas_commit2, affine_to_rasmm=np.eye(4))
tckfile_commit2 = TckFile(tracto_filtrado_commit2, nuevo_header_commit2)
tckfile_commit2.save(filtered_tract_commit2)

print(f"Se guardó la tractografía filtrada por COMMIT2 en '{filtered_tract_commit2}', con {len(streamlines_filtradas_commit2)} fibras.")
