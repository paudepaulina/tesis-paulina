import numpy as np
import nibabel as nib
from dipy.io.streamline import save_tck
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import StatefulTractogram, Space

# 1. Cargar datos de la máscara y de direcciones
archivo_mascara = "ISMRM_2023_b3000_mask.nii"
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

# 2. Funciones para seleccionar y analizar semilla de inicio

coordenadas = np.argwhere(mascara == 1)
semillas_exploradas = set()

def seleccionar_semilla_valida(coordenadas, peaks, mascara, semillas_exploradas, max_intentos=100):
    for _ in range(max_intentos):
        semilla = coordenadas[np.random.randint(0, len(coordenadas))]
        if tuple(semilla) in semillas_exploradas:
            continue  # Evitar usar la misma semilla
        x, y, z = semilla
        if mascara[x, y, z] == 1 and np.linalg.norm(peaks[x, y, z]) > 0:
            return semilla
    return None  # Si no encuentra una válida

def analizar_semilla(semilla, picos, mascara):
    x, y, z = semilla
    print(f"Analizando semilla en posición: {semilla}")

    # Verificar que esté dentro de la máscara

    if mascara[x, y, z] == 0:
        print("La semilla está FUERA de la máscara")
        return False
    else:
        print("La semilla está dentro de la máscara")

    # Obtener vectores de dirección en la semilla
    direcciones = picos[x, y, z]
    normas = np.linalg.norm(direcciones, axis=1)
    print(f"Vectores en la semilla: \n{direcciones}")
    print(f"Normas de los vectores: {normas}")

    # Verificar si si hay un vector significativo

    if np.all(normas < 1e-3):
        print("Todos los vectores tienen normas muy bajas, no es una buena semilla")
        return False
   
    # Revisar si al menos un vecino tiene buenos vectores

    vecinos_validos = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0: 
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < picos.shape[0] and 0 <= ny < picos.shape[1] and 0 <= nz < picos.shape[2]:
                    norm_vecino = np.linalg.norm(picos[nx, ny, nz], axis=1)
                    if np.any(norm_vecino > 1e-3):
                        vecinos_validos += 1
    
    print(f"Vecinos con vectores fuertes: {vecinos_validos}/26")
    if vecinos_validos == 0:
        print("Ningún vecino tiene vector válido, la trayectoria no puede continuar")
        return False
    return True

# 3. Definir parámetros

# Iniciar desde la semilla seleccionada

semilla_actual = seleccionar_semilla_valida(coordenadas, picos, mascara, semillas_exploradas)
if semilla_actual is None or not analizar_semilla(semilla_actual, picos, mascara):
    print("No se encontró una semilla válida en los intentos iniciales.")
    exit()

# Parámetros para propagación
tamaño_paso = 0.5    # Tamaño de cada paso 
angulo_maximo = 60   # Ángulo máximo entre pasos (en grados)
longitud_minima = 10 # Longitud mínima (mm)
longitud_maxima = 20 # Longitud máxima (mm)
max_pasos = 500      # Número máximo de pasos

# 4. Calcular el ángulo entre vectores
def calcular_angulo(v1, v2):
    cos_angulo = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angulo, -1, 1)))

# 5. Interpolación de direcciones
def interpolar_direcciones(peaks, posicion):
    x, y, z = posicion
    x0, y0, z0 = int(x), int(y), int(z)

    # Asegurar que estemos dentro del volumen
    if not (0 <= x0 < peaks.shape[0] - 1 and 0 <= y0 < peaks.shape[1] - 1 and 0 <= z0 < peaks.shape[2] - 1):
        return None

    #Extraer vecinos
    vecinos = peaks[x0:x0+2, y0:y0+2, z0:z0+2]


    #Extraer eigenvector principal
    norm = np.linalg.norm(vecinos, axis=-1)
    max_indices = np.argmax(norm, axis=-1)
    vectores_principales = np.zeros((2,2,2,3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                vectores_principales[i,j,k] = vecinos[i,j,k, max_indices[i,j,k]]
    
    
    if np.all(np.linalg.norm(vectores_principales, axis=-1) == 0):
        print(f"Advertencia: Todos los eigenvectores en {posicion} son cero. Probando alternativa.")
        normas = np.linalg.norm(peaks[x0, y0, z0], axis=1)
        
        #Elegir el vector con la norma más alta de la semilla
        if np.max(normas) > 1e-3:
            mejor_direccion = peaks[x0, y0, z0, np.argmax(normas)]
            return mejor_direccion / np.linalg.norm(mejor_direccion)
        
        return None #Si tampoco hay vectores validos, no continuar


    # Interpolación ponderada
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

    direccion_interpolada = np.sum(vectores_principales * pesos, axis=(0, 1, 2))
    
    if np.linalg.norm(direccion_interpolada) > 0:
        return direccion_interpolada / np.linalg.norm(direccion_interpolada)
    return None

# 6. Funciones para propagación bidireccional

def propagar_trayectoria(peaks, semilla, dwi_affine, tamaño_paso, angulo_maximo, max_pasos, mascara, invertir=False):
    
    #Generar una trayectoria desde una semilla siguiendo las direcciones de peaks

    trayectoria = [semilla]  # La semilla se puede guardar como punto inicial (continuo)
    # Iniciar con la semilla convertida a float
    posicion_actual = np.array(semilla, dtype=float)
    direccion_actual = None

    for _ in range(max_pasos):
        # Usar la posición continua para la interpolación
        direccion = interpolar_direcciones(peaks, posicion_actual)
        if direccion is None:
            break

        # Verificar la bidireccionalidad
        if invertir:
            direccion = -direccion
        
        #Verificar ángulo con la dirección anterior
        if direccion_actual is not None:
            angulo = calcular_angulo(direccion_actual, direccion)
            if angulo > angulo_maximo:
                break

        # Actualizar la posición en coordenadas continuas
        nuevo_punto = posicion_actual + tamaño_paso * direccion

        # Validar si el nuevo voxel es valido dentro de la máscada
        nuevo_voxel = np.round(nuevo_punto).astype(int)
        if not (0 <= nuevo_voxel[0] < mascara.shape[0] and 
                0 <= nuevo_voxel[1] < mascara.shape[1] and 
                0 <= nuevo_voxel[2] < mascara.shape[2]) or mascara[tuple(nuevo_voxel)] == 0:
            break
        # Guardar el punto continuo para mantener la suavidad de la trayectoria
        trayectoria.append(nuevo_punto.tolist())
        posicion_actual = nuevo_punto
        direccion_actual = direccion
    return trayectoria

def realizar_trayectoria_bidireccional(peaks, semilla, dwi_affine, tamaño_paso, angulo_maximo, max_pasos, mascara):
   #propagar en ambas direcciones
    trayectoria_forward = propagar_trayectoria(peaks, semilla, dwi_affine, tamaño_paso, angulo_maximo, max_pasos, mascara, invertir=False)
    trayectoria_backward = propagar_trayectoria(peaks, semilla, dwi_affine, tamaño_paso, angulo_maximo, max_pasos, mascara, invertir=True)
    trayectoria_backward = trayectoria_backward[::-1]
    # Evitar duplicar la semilla
    if trayectoria_backward and np.array_equal(np.array(trayectoria_backward[-1]), np.array(semilla)):
        trayectoria_backward = trayectoria_backward[:-1]
    trayectoria_completa = trayectoria_backward + trayectoria_forward
    return trayectoria_completa

def calcular_longitud(streamline, dwi_affine):
    #Calcular la longitud fisica de los streamlines
    # Convertir cada punto a coordenadas físicas
    puntos_fisicos = [nib.affines.apply_affine(dwi_affine, np.array(p)) for p in streamline]
    longitud = 0
    # Sumar las distancias Euclidianas entre puntos consecutivos
    for i in range(1, len(puntos_fisicos)):
        longitud += np.linalg.norm(np.array(puntos_fisicos[i]) - np.array(puntos_fisicos[i-1]))
    return longitud


# 7. Bucle principal para generar la trayectoria total
trayectorias = []
max_semilla_intentos = 30
intentos_en_semilla = 0
longitud_acumulada = 0

while longitud_acumulada < longitud_minima:
    if intentos_en_semilla >= max_semilla_intentos:
        print("Número máximo de semillas intentadas alcanzado. Terminando.")
        break #detener el proceso si no hay semillas utiles

    #Elegir nueva semilla si es necesario

    if tuple(semilla_actual) in semillas_exploradas:
        print(f"Semilla ya explorada: {semilla_actual}. Buscando otra.")
        intentos_en_semilla += 1
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]
        continue

    if mascara[tuple(semilla_actual)] == 0:
        print("Semilla fuera de la máscara. Buscando otra.")
        intentos_en_semilla += 1
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]
        continue

    semillas_exploradas.add(tuple(semilla_actual)) # Guardar semilla para evitar repetirla
    
    print(f"Probando trayectoria bidireccional con la semilla {semilla_actual}")
    trayectoria = realizar_trayectoria_bidireccional(
        peaks=picos,
        semilla=semilla_actual,
        dwi_affine=dwi_affine,
        tamaño_paso=tamaño_paso,
        angulo_maximo=angulo_maximo,
        max_pasos=max_pasos,
        mascara=mascara
    )
    
    if trayectoria:
        longitud_acumulada = calcular_longitud(trayectoria, dwi_affine)
        if longitud_acumulada >= longitud_minima:
            print(f"Trayectoria válida generada con longitud {longitud_acumulada:.2f} mm.")
            trayectorias.append(trayectoria)
            break #Salimos del bucle porque ya se encontró trayectoria valida
        else:
            print(f"Trayectoria demasiado corta ({longitud_acumulada:.2f} mm). Probando otra semilla.")
    else:
        print("No se pudo generar una trayectoria desde esta semilla. Buscando otra.")

    intentos_en_semilla += 1
    semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]

print("Fin de la búsqueda de semillas.")


# 8. Guardar la trayectoria generada (o trayectorias, según se requiera)
if trayectorias:
    streamlines = Streamlines([np.array([nib.affines.apply_affine(dwi_affine, punto) for punto in t]) for t in trayectorias])
    sft = StatefulTractogram(streamlines, nib.Nifti1Image(dwi_datos, dwi_affine), Space.RASMM)
    save_tck(sft, "Trayectorias_finales.tck")
    print("Trayectorias guardadas con éxito.")
else:
    print("No se generaron trayectorias válidas.")
