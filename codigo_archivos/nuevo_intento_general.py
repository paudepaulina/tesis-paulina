import numpy as np
import nibabel as nib
from dipy.io.streamline import save_tck
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import StatefulTractogram, Space

# 1. Carga de datos
archivo_mascara = "/home/paulinabr/Escritorio/Tesis/Picos archivos/mascara_redimensionada.nii.gz"  # máscara binaria de semillas
archivo_picos  = "/home/paulinabr/Escritorio/Tesis/peaks_mrtrix.nii"                           # modificacion: Uso de archivo peaks mrtrix
archivo_dwi    = "ISMRM_2023_b3000.nii"                                                         # volumen DWI original

# Leer los picos en forma: 90×108×90×9
picos = nib.load(archivo_picos).get_fdata()

# Leer la máscara 
mascara = nib.load(archivo_mascara).get_fdata()

# Cargar el volumen DWI completo y extraer datos + matriz affine
dwi_img    = nib.load(archivo_dwi)
dwi_datos  = dwi_img.get_fdata()     # datos DWI (90×108×90×65)
dwi_affine = dwi_img.affine          # affine voxel a físico

# Mostrar dimensiones para confirmar que todo coincide
print(f"Dimensiones de picos: {picos.shape}")
print(f"Dimensiones de máscara: {mascara.shape}")
print(f"Dimensiones DWI: {dwi_datos.shape}")

# 2. Parámetros de propagación
tamaño_paso    = 0.5    # mm a avanzar en cada iteración
angulo_maximo  = 60     # máximo giro permitido entre pasos (grados)
longitud_minima = 10    # longitud mínima aceptable de una fibra (mm)
max_pasos      = 100    # límite de pasos para cada trayectoria


# 3. Funciones auxiliares 
def calcular_longitud(streamline, affine):
   #Calcula la longitud total de la trayectoria (streamline) en milímetros
   
    # Transformar cada punto de voxel a coordenadas físicas (mm)
    pts = [nib.affines.apply_affine(affine, p) for p in streamline]
    # Sumar las distancias euclídeas entre cada par de puntos consecutivos
    return sum(
        np.linalg.norm(pts[i] - pts[i-1])
        for i in range(1, len(pts))
    )

def extraer_3vectores(voxel):
    
    #Toma un array de 9 valores (un voxel de picos) y lo convierte en 3 vectores 3D.
    #Se asume que están ordenados como:
    #  [x1, y1, z1, x2, y2, z2, x3, y3, z3]
    
    return np.vstack((
        voxel[0:3],  # primer vector
        voxel[3:6],  # segundo vector
        voxel[6:9]   # tercer vector
    ))

def interpolar_direcciones(peaks, pos_vox):
    # Convertir coordenadas continuas a índices enteros de voxel
    x0, y0, z0 = map(int, pos_vox)

    # Si estamos fuera del volumen o muy cerca del borde no hay dirección válida
    if not (0 <= x0 < peaks.shape[0]-1 
            and 0 <= y0 < peaks.shape[1]-1 
            and 0 <= z0 < peaks.shape[2]-1):
        return None

    # Extraer el bloque 2×2×2 de voxeles vecinos 
    vecinos = peaks[x0:x0+2, y0:y0+2, z0:z0+2]

    # Preparar matriz para almacenar la dirección dominante en cada esquina
    vects = np.zeros((2,2,2,3))

    # Para cada esquina del cubo 2×2×2:
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Convertir los 9 valores del voxel a 3 vectores de 3 componentes
                vols = extraer_3vectores(vecinos[i,j,k])
                # Calcular la magnitud de cada vector
                norms = np.linalg.norm(vols, axis=1)
                # Si alguno es significativo, quedarnos con el de mayor norma
                if norms.max() > 1e-3:
                    vects[i,j,k] = vols[np.argmax(norms)]

    # Si todas las esquinas no tienen dirección válida, terminar
    if np.allclose(vects, 0):
        return None

    # Calcular desplazamientos fraccionales dentro del voxel base
    dx, dy, dz = pos_vox - np.array([x0, y0, z0])

    # Construir los 8 pesos de interpolación trilineal
    pesos = np.array([
        [(1-dx)*(1-dy)*(1-dz), (1-dx)*(1-dy)*dz],
        [(1-dx)*dy*(1-dz),     (1-dx)*dy*dz],
        [dx*(1-dy)*(1-dz),     dx*(1-dy)*dz],
        [dx*dy*(1-dz),         dx*dy*dz]
    ]).reshape(2,2,2,1)

    # Interpolar sumando vectores ponderados
    dir_interp = np.sum(vects * pesos, axis=(0,1,2))

    # Normalizar para obtener dirección unitaria
    norm = np.linalg.norm(dir_interp)
    if norm > 1e-9:
        return dir_interp / norm

    return None

def calcular_angulo(v1, v2):
    return np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1)))

# 4. Propagación en espacio físico

def propagar_trayectoria(peaks, semilla, tamaño_paso, angulo_max, max_pasos, dwi_affine, dwi_shape, invertir=False):

    # Inicializar la trayectoria con la posición de la semilla (en mm)
    pos_mm = nib.affines.apply_affine(dwi_affine, semilla)
    dir_prev = None                     # No hay dirección previa aún
    trayectoria = [pos_mm.tolist()]     # Lista de puntos físicos
    
    # Precomputar la matriz inversa para pasar de espacio físico a voxel
    inv_affine = np.linalg.inv(dwi_affine)
    
    for paso in range(max_pasos):
        # Convertir la posición física actual a coordenadas continuas de voxel
        pos_vox = nib.affines.apply_affine(inv_affine, pos_mm)

        # Obtener la dirección interpolada a partir de los picos en ese voxel
        direccion = interpolar_direcciones(peaks, pos_vox)
        if direccion is None:
            # No hay dirección válida = terminar trayectoria
            break

        # invertir sólo en primer punto
        if dir_prev is None:
            if invertir:
                direccion = -direccion
        else:
            #  usar producto punto para decidir inversión 
            if np.dot(direccion, dir_prev) < 0:
                direccion = -direccion

            #  si supera ángulo máximo, cortar
            if calcular_angulo(dir_prev, direccion) > angulo_max:
                break

        # Avanzar la posición en espacio físico; Ecuación de propagación
        pos_mm = pos_mm + tamaño_paso * direccion

        # Redondear a voxel y verificar que esté dentro del volumen
        vox_round = np.round(nib.affines.apply_affine(inv_affine, pos_mm)).astype(int)
        if not np.all((vox_round >= 0) & (vox_round < dwi_shape[:3])):
            # Punto fuera del volumen =terminar trayectoria
            break

        # Añadir nuevo punto a la trayectoria y actualizar dirección previa
        trayectoria.append(pos_mm.tolist())
        dir_prev = direccion

    return trayectoria



def realizar_trayectoria_bidireccional(peaks, semilla, tamaño_paso, angulo_max, max_pasos, dwi_shape):
    
    #Genera una trayectoria completa bidireccional desde una semilla:
    #  - Primero propaga hacia adelante (invertir=False)
    #  - Luego propaga hacia atrás (invertir=True)
    #  - Une ambas trayectorias sin duplicar la semilla
    
    # Propagación hacia adelante desde la semilla
    fwd = propagar_trayectoria(
        peaks, semilla, tamaño_paso, angulo_max, max_pasos, dwi_affine, dwi_shape, invertir=False
    )

    # Propagación hacia atrás, luego invertimos el orden 
    bwd = propagar_trayectoria(
        peaks, semilla, tamaño_paso, angulo_max, max_pasos, dwi_affine, dwi_shape, invertir=True
    )[::-1]

    # Si la última posición de la trayectoria backward coincide con la semilla, la quitamos
    if bwd and np.allclose(bwd[-1], nib.affines.apply_affine(dwi_affine, semilla)):
        bwd.pop()

    # Concatenamos backward + forward para obtener la trayectoria completa
    return bwd + fwd


# 5. Generar y filtrar trayectorias
semillas = np.argwhere(mascara)  # Todas las coordenadas donde máscara==1
trayectorias = []

# Contador para llevar el progreso
total_semillas = len(semillas)

for idx, s in enumerate(semillas):
    print(f"Generando trayectoria para semilla {idx+1}/{total_semillas}: {s}")
    traj = realizar_trayectoria_bidireccional(
        picos, s, tamaño_paso, angulo_maximo, max_pasos, dwi_datos.shape
    )
    # Solo conservar trayectorias cuya longitud supere el umbral mínimo
    if traj and calcular_longitud(traj, dwi_affine) >= longitud_minima:
        trayectorias.append(traj)

print(f"Trayectorias válidas: {len(trayectorias)}")

# 6. Guardar en TCK
if trayectorias:
    # Convertir cada trayectoria a array (ya están en espacio físico)
    streamlines = Streamlines([np.array(t) for t in trayectorias])

    # Crear tractograma en espacio RASMM (coordenadas físicas) y guardar
    sft = StatefulTractogram(streamlines, nib.Nifti1Image(dwi_datos, dwi_affine), Space.RASMM)
    save_tck(sft, "Trayectorias_finales.tck")
    print("Guardado exitoso.")
else:
    print("No hay trayectorias para guardar.")
