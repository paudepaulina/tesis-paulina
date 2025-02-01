import numpy as np
import nibabel as nib
from dipy.io.streamline import save_tck
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import StatefulTractogram, Space

# 1. Cargar datos 
archivo_mascara = "mascara_CP_intento_4.nii"
archivo_picos = "/home/paulinabr/Escritorio/Tesis/Picos archivos/peaks_correcto.nii.gz"
archivo_dwi = "ISMRM_2023_b3000.nii"

picos = nib.load(archivo_picos).get_fdata()
mascara = nib.load(archivo_mascara).get_fdata()
dwi_datos = nib.load(archivo_dwi).get_fdata()
dwi_affine = nib.load(archivo_dwi).affine

print("Se cargaron los datos")
print(f"Dimensiones de picos: {picos.shape}")
print(f"Dimensiones de máscara: {mascara.shape}")
print(f"Dimensiones de los datos dwi: {dwi_datos.shape}")

# 2. Parámetros de propagación

tamaño_paso   = 0.5    # Tamaño del paso 
angulo_maximo = 60     # Ángulo máximo 
longitud_minima = 10   # Longitud mínima en mm
longitud_maxima = 20   # Longitud máxima en mm
max_pasos = 500        # Número máximo de pasos por trayectoria

# 3. Funciones

# Calculando la longitud de la trayectoria

def calcular_longitud(streamline, dwi_affine):
    puntos_fisicos = [nib.affines.apply_affine(dwi_affine, np.array(p)) for p in streamline]
    longitud = 0
    for i in range(1, len(puntos_fisicos)):
        longitud += np.linalg.norm(np.array(puntos_fisicos[i]) - np.array(puntos_fisicos[i-1]))
    return longitud

# Funciones de interpolación y propagación bidireccional:

def interpolar_direcciones(peaks, posicion):
    x, y, z = posicion
    x0, y0, z0 = int(x), int(y), int(z)
    if not (0 <= x0 < peaks.shape[0] - 1 and 0 <= y0 < peaks.shape[1] - 1 and 0 <= z0 < peaks.shape[2] - 1):
        return None
    vecinos = peaks[x0:x0+2, y0:y0+2, z0:z0+2]
    norm = np.linalg.norm(vecinos, axis=-1)
    max_indices = np.argmax(norm, axis=-1)
    vectores_principales = np.zeros((2,2,2,3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                vectores_principales[i,j,k] = vecinos[i,j,k, max_indices[i,j,k]]
    if np.all(np.linalg.norm(vectores_principales, axis=-1) == 0):
        normas = np.linalg.norm(peaks[x0, y0, z0], axis=1)
        if np.max(normas) > 1e-3:
            mejor_direccion = peaks[x0, y0, z0, np.argmax(normas)]
            return mejor_direccion / np.linalg.norm(mejor_direccion)
        return None
    pesos = np.array([
        (1 - (x - x0)) * (1 - (y - y0)) * (1 - (z - z0)),
        (x - x0) * (1 - (y - y0)) * (1 - (z - z0)),
        (1 - (x - x0)) * (y - y0) * (1 - (z - z0)),
        (1 - (x - x0)) * (1 - (y - y0)) * (z - z0),
        (x - x0) * (y - y0) * (1 - (z - z0)),
        (1 - (x - x0)) * (y - y0) * (z - z0),
        (x - x0) * (1 - (y - y0)) * (z - z0),
        (x - x0) * (y - y0) * (z - z0)
    ]).reshape(2,2,2,1)
    direccion_interpolada = np.sum(vectores_principales * pesos, axis=(0,1,2))
    if np.linalg.norm(direccion_interpolada) > 0:
        return direccion_interpolada / np.linalg.norm(direccion_interpolada)
    return None

def calcular_angulo(v1, v2):
    cos_angulo = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angulo, -1, 1)))

def propagar_trayectoria(peaks, semilla, dwi_affine, tamaño_paso, angulo_maximo, max_pasos, mascara, invertir=False):
    
    trayectoria = [semilla]
    posicion_actual = np.array(semilla, dtype=float)
    direccion_actual = None

    for _ in range(max_pasos):
        # Usar la posición continua para la interpolación
        direccion = interpolar_direcciones(peaks, posicion_actual)
        if direccion is None:
            break
        if invertir:
            direccion = -direccion
        if direccion_actual is not None:
            angulo = calcular_angulo(direccion_actual, direccion)
            if angulo > angulo_maximo:
                break
        nuevo_punto = posicion_actual + tamaño_paso * direccion
        nuevo_voxel = np.round(nuevo_punto).astype(int)
        if not (0 <= nuevo_voxel[0] < mascara.shape[0] and 
                0 <= nuevo_voxel[1] < mascara.shape[1] and 
                0 <= nuevo_voxel[2] < mascara.shape[2]) or mascara[tuple(nuevo_voxel)] == 0:
            break
        trayectoria.append(nuevo_punto.tolist())
        posicion_actual = nuevo_punto
        direccion_actual = direccion
    return trayectoria

def realizar_trayectoria_bidireccional(peaks, semilla, dwi_affine, tamaño_paso, angulo_maximo, max_pasos, mascara):
    trayectoria_forward = propagar_trayectoria(peaks, semilla, dwi_affine, tamaño_paso, angulo_maximo, max_pasos, mascara, invertir=False)
    trayectoria_backward = propagar_trayectoria(peaks, semilla, dwi_affine, tamaño_paso, angulo_maximo, max_pasos, mascara, invertir=True)
    trayectoria_backward = trayectoria_backward[::-1]
    if trayectoria_backward and np.array_equal(np.array(trayectoria_backward[-1]), np.array(semilla)):
        trayectoria_backward = trayectoria_backward[:-1]
    trayectoria_completa = trayectoria_backward + trayectoria_forward
    return trayectoria_completa

# 4. Bucle para generar trayectoria total
semillas = np.argwhere(mascara == 1)
print(f"Total de semillas en la máscara: {len(semillas)}")

trayectorias = []
for i, semilla in enumerate(semillas):
    print(f"Generando trayectoria para semilla {i+1}/{len(semillas)}: {semilla}")
    trayectoria = realizar_trayectoria_bidireccional(
        peaks=picos,
        semilla=semilla,
        dwi_affine=dwi_affine,
        tamaño_paso=tamaño_paso,
        angulo_maximo=angulo_maximo,
        max_pasos=max_pasos,
        mascara=mascara
    )
    if trayectoria:
        longitud = calcular_longitud(trayectoria, dwi_affine)
        if longitud >= longitud_minima:
            trayectorias.append(trayectoria)
            print(f"Trayectoria válida generada (longitud: {longitud:.2f} mm).")
        else:
            print(f"Trayectoria demasiado corta (longitud: {longitud:.2f} mm).")
    else:
        print("No se pudo generar trayectoria para esta semilla.")

print(f"Se generaron {len(trayectorias)} trayectorias válidas.")

#  5. Guardar las trayectorias 
if trayectorias:
    streamlines = Streamlines([np.array([nib.affines.apply_affine(dwi_affine, p) for p in t]) for t in trayectorias])
    sft = StatefulTractogram(streamlines, nib.Nifti1Image(dwi_datos, dwi_affine), Space.RASMM)
    save_tck(sft, "Trayectorias_finales.tck")
    print("Trayectorias guardadas con éxito.")
else:
    print("No se generaron trayectorias válidas.")
