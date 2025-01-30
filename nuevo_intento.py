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

# 2. Obtener la semilla de inicio

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


# 3. Analizar características de la semilla seleccionada

def analizar_semilla(semilla, picos, mascara):
    x,y,z = semilla
    print(f"Analizando semilla en posición: {semilla}")

    #Verificar que esté dentro de la máscara
    
    if mascara[x,y,z] == 0:
        print("La semilla está FUERA de la máscara")
        return False
    else:
        print("La semilla está dentro de la máscara")

    # Obtener vectores de dirección en la semilla

    direcciones = picos[x,y,z]
    normas = np.linalg.norm(direcciones, axis=1)

    print(f"Vectores en la semilla: \n{direcciones}")
    print(f"Normas de los vectores: {normas}")

    # Verificar si si hay un vector significativo

    if np.all(normas < 1e-3):
        print("Todos los vectores tienen normas muy bajas, no es una buena semilla")
        return False
    else:
        print("La semilla tiene vectores con normas adecuadas")

    # Revisar si al menos un vecino tiene buenos vectores
    
    vecinos_validos = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:  # Saltar la semilla misma
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < picos.shape[0] and 0 <= ny < picos.shape[1] and 0 <= nz < picos.shape[2]:
                    norm_vecino = np.linalg.norm(picos[nx, ny, nz], axis=1)
                    if np.any(norm_vecino > 1e-3):
                        vecinos_validos += 1

    print(f"Vecinos con vectores fuertes: {vecinos_validos}/26")
    if vecinos_validos == 0:
        print("Ningun vecino tiene vector valido, la trayectoria no puede continuar")

        return False

semilla = seleccionar_semilla_valida(coordenadas,picos,mascara,semillas_exploradas)

if not analizar_semilla(semilla, picos, mascara):
    print("Buscando una nueva semilla...")
    semilla = seleccionar_semilla_valida(np.argwhere(mascara == 1), picos, mascara,semillas_exploradas)
    if semilla is None:
        print("No se encontraron semillas válidas después de múltiples intentos.")
        exit()
    
# 3. Definir parámetros

semilla_actual = semilla #Iniciar desde la semilla seleccionada
trayectoria = []    #Guardar puntos recorridos
semillas_exploradas = set() #Evitar que se repitan las semillas exploradas
max_intentos = 10 # busqueda de voxel
intentos = 0
longitud_minima = 10  # Longitud mínima del strln en mm
longitud_acumulada = 0
longitud_maxima = 20
paso_maximo = 100 #máxima cantidad de pasos
tamaño_paso = 0.5  #Tamaño de cada paso
angulo_maximo = 60  # angulo máximo entre vectores

# 4. Calcular el angulo entre vectores

def calcular_angulo(v1, v2):
    cos_angulo = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angulo, -1, 1)))

# 5. Interpolación de los vectores

def interpolar_direcciones(peaks, posicion):
    x, y, z = posicion
    x0, y0, z0 = int(x), int(y), int(z)

    # Asegurar que estamos dentro del volumen
    if not (0 <= x0 < peaks.shape[0] - 1 and 0 <= y0 < peaks.shape[1] - 1 and 0 <= z0 < peaks.shape[2] - 1):
        return None
    
    # Extraer vecinos
    vecinos = peaks[x0:x0+2, y0:y0+2, z0:z0+2]

    # Extraer eigenvector principal
    norm = np.linalg.norm(vecinos, axis=-1)
    max_indices = np.argmax(norm, axis=-1)

    vectores_principales = np.zeros((2,2,2,3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                vectores_principales[i,j,k] = vecinos[i,j,k, max_indices[i,j,k]]

    if np.all(np.linalg.norm(vectores_principales, axis=-1) == 0):
        print(f"⚠️ Advertencia: Todos los eigenvectores en {posicion} son cero. Probando alternativa.")
        
        # Nueva estrategia: elegir el vector con la norma más alta de la semilla
        normas = np.linalg.norm(peaks[x0, y0, z0], axis=1)
        if np.max(normas) > 1e-3:
            mejor_direccion = peaks[x0, y0, z0, np.argmax(normas)]
            return mejor_direccion / np.linalg.norm(mejor_direccion)

        return None  # Si tampoco hay vectores válidos, no continuar

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

    

# 6. Funcion para la  trayectoria 
def realizar_trayectoria(peaks, semilla, dwi_affine, tamaño_paso, angulo_maximo, longitud_maxima, mascara):
    """
    Genera una trayectoria desde una semilla siguiendo las direcciones en peaks.
    """
    trayectoria = [semilla]
    posicion_actual = np.array(semilla, dtype=float)
    longitud_acumulada = 0
    direccion_actual = None
    max_pasos = 500

    for _ in range(max_pasos):
        # Obtener dirección interpolada en la posición actual
        direccion_interpolada = interpolar_direcciones(peaks, np.round(posicion_actual).astype(int))

        # Si no se encontró dirección válida, probar otro vector en la misma semilla
        if direccion_interpolada is None:
            print(f"⚠️ No se encontró dirección válida en {posicion_actual}. Probando otro vector.")
            normas = np.linalg.norm(peaks[tuple(np.round(posicion_actual).astype(int))], axis=1)
            if np.max(normas) > 1e-3:
                mejor_direccion = peaks[tuple(np.round(posicion_actual).astype(int))][np.argmax(normas)]
                direccion_interpolada = mejor_direccion / np.linalg.norm(mejor_direccion)
            else:
                print(f"No hay direcciones válidas en {posicion_actual}. Deteniendo propagación.")
                break

        # Verificar ángulo con la dirección anterior
        if direccion_actual is not None:
            angulo = calcular_angulo(direccion_actual, direccion_interpolada)
            if angulo > angulo_maximo:
                print(f"Ángulo {angulo:.2f}° excede el umbral en {posicion_actual}. Intentando otro paso.")
                continue  # Intentar otro paso sin detener la propagación

        # Calcular el siguiente punto en la trayectoria
        nuevo_punto = posicion_actual + tamaño_paso * direccion_interpolada
        nuevo_voxel = np.round(nuevo_punto).astype(int).flatten()

        # Validar si el nuevo voxel es válido dentro de la máscara
        if not (0 <= nuevo_voxel[0] < mascara.shape[0] and
                0 <= nuevo_voxel[1] < mascara.shape[1] and
                0 <= nuevo_voxel[2] < mascara.shape[2]) or mascara[tuple(nuevo_voxel)] == 0:
            print(f"El punto {nuevo_voxel} está fuera de la máscara o volumen. Ajustando paso.")
            tamaño_paso *= 0.9  # Reducir tamaño de paso y volver a intentar
            continue

        # Calcular distancia física entre puntos
        punto_fisico_actual = nib.affines.apply_affine(dwi_affine, posicion_actual)
        punto_fisico_nuevo = nib.affines.apply_affine(dwi_affine, nuevo_voxel)
        distancia = np.linalg.norm(punto_fisico_nuevo - punto_fisico_actual)

        # Si la distancia es menor a un umbral, ajustar el tamaño del paso
        if distancia < 0.1:
            tamaño_paso *= 1.1  # Aumentar tamaño de paso si es muy pequeño
            continue  # Reintentar con un paso más grande

        # Verificar si la trayectoria alcanza la longitud máxima
        if longitud_acumulada + distancia > longitud_maxima:
            print(f"Longitud máxima alcanzada: {longitud_acumulada+distancia:.2f} mm. Terminando streamline.")
            break

        # Actualizar trayectoria
        longitud_acumulada += distancia
        trayectoria.append(nuevo_voxel.tolist())
        posicion_actual = nuevo_voxel
        direccion_actual = direccion_interpolada  # Guardar la nueva dirección

    return trayectoria, longitud_acumulada



# 7. Bucle principal para generar la trayectoria total
trayectorias = []
max_semilla_intentos = 100
intentos_en_semilla = 0

max_semilla_intentos = 30  # Limitar intentos para evitar loops infinitos
intentos_en_semilla = 0
semillas_exploradas = set()

while longitud_acumulada < longitud_minima:
    if intentos_en_semilla >= max_semilla_intentos:
        print("Número máximo de semillas intentadas alcanzado. Terminando.")
        break  # Detiene el proceso si no hay semillas útiles después de muchos intentos.

    # Elegir una nueva semilla si es necesario
    if tuple(semilla_actual) in semillas_exploradas:
        print(f"Semilla ya explorada: {semilla_actual}. Buscando otra.")
        intentos_en_semilla += 1
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]
        continue

    # Verificar si la semilla está dentro de la máscara
    if mascara[tuple(semilla_actual)] == 0:
        print("Semilla fuera de la máscara. Buscando otra.")
        intentos_en_semilla += 1
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]
        continue

    semillas_exploradas.add(tuple(semilla_actual))  # Guardar semilla para evitar repetirla

    # Obtener la dirección inicial para la propagación
    direccion_principal = interpolar_direcciones(picos, np.array(semilla_actual, dtype=float))

    if direccion_principal is not None:
        print(f"Probando trayectoria con la semilla {semilla_actual}...")

        trayectoria, longitud_acumulada = realizar_trayectoria(
            peaks=picos,
            semilla=semilla_actual,
            dwi_affine=dwi_affine,
            tamaño_paso=tamaño_paso,
            angulo_maximo=angulo_maximo,
            longitud_maxima=longitud_maxima,
            mascara=mascara
        )

        if longitud_acumulada >= longitud_minima:
            print(f" Trayectoria válida generada con longitud {longitud_acumulada:.2f} mm.")
            trayectorias.append(trayectoria)
            break  #  Salimos del bucle porque ya encontramos una trayectoria válida

        else:
            print(f" Trayectoria demasiado corta ({longitud_acumulada:.2f} mm). Probando expansión...")

            #  Intentar otro vector en la misma semilla antes de rendirse
            normas = np.linalg.norm(picos[tuple(np.round(semilla_actual).astype(int))], axis=1)
            if np.max(normas) > 1e-3:
                mejor_direccion = picos[tuple(np.round(semilla_actual).astype(int))][np.argmax(normas)]
                direccion_principal = mejor_direccion / np.linalg.norm(mejor_direccion)
                print(" Probando otra propagación desde la misma semilla...")
                continue  # Volver a intentar con una nueva dirección en la misma semilla
            else:
                print(" No se encontró dirección válida en la semilla. Buscando nueva semilla.")

    # Si la propagación no funciona, cambiar de semilla
    intentos_en_semilla += 1
    semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]

print("Fin de la búsqueda de semillas.")

# Guardar trayectorias si se generaron
if trayectorias:
    streamlines = Streamlines([np.array([nib.affines.apply_affine(dwi_affine, punto) for punto in t]) for t in trayectorias])
    sft = StatefulTractogram(streamlines, nib.Nifti1Image(dwi_datos, dwi_affine), Space.RASMM)
    save_tck(sft, "Trayectorias_finales.tck")
    print("Trayectorias guardadas con éxito.")
else:
    print("No se generaron trayectorias válidas.")
