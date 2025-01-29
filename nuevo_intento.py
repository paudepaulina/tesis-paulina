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

def seleccionar_semilla_valida(coordenadas, peaks, mascara, max_intentos=100):
    for _ in range(max_intentos):
        semilla = coordenadas[np.random.randint(0, len(coordenadas))]
        x, y, z = semilla
        if mascara[x, y, z] == 1 and np.linalg.norm(peaks[x, y, z]) > 0:
            return semilla
    return None  # Si no encuentra una válida

semilla = seleccionar_semilla_valida(coordenadas,picos,mascara)

if semilla is None:
    print("No se encontraron semillas validas despues de multiples intentos")
    exit()

print(f"Semilla seleccionada: {semilla}")

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
    """
    Interpola direcciones de los picos en un vecindario del voxel dado.
    peaks: Array 5D de picos [x, y, z, número_de_picos, componentes].
    voxel: Coordenadas del voxel actual (x, y, z).
    neighborhood_size: Tamaño del vecindario cúbico para la interpolación.
    """
    x, y, z = posicion
    x0, y0, z0 = int(x), int(y),int(z) # Voxel base
    dx, dy, dz = x - x0, y - y0, z - z0 # mantener dentro del voxel

    # Asegurar que estamos en el volumen
    if not (0 <= x0 < peaks.shape[0] - 1 and 0 <= y0 < peaks.shape[1] - 1 and 0 <= z0 < peaks.shape[2] - 1):
        return None # Asegurar que estamos en el voxel
    
    # Extraer vecindario
    
    vecinos = peaks[x0:x0+2, y0:y0+2, z0:z0+2]

    # Extraer eigenvector principal de cada vecino

    norm = np.linalg.norm(vecinos, axis=-1)
    max_indices = np.argmax(norm, axis=-1)

    vectores_principales = np.zeros((2,2,2,3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                vectores_principales[i,j,k] = vecinos[i,j,k, max_indices[i,j,k]]
    
    if np.all(np.linalg.norm(vectores_principales, axis=-1) == 0):
        print(f"Todos los eigenvectores en {posicion} son cero")
        return None
    

    # Ponderación trilineal basada en distancias dentro del voxel

    pesos = np.array([
        (1 - dx) * (1 - dy) * (1 - dz),
        dx * (1 - dy) * (1 - dz),
        (1 - dx) * dy * (1 - dz),
        (1 - dx) * (1 - dy) * dz,
        dx * dy * (1 - dz),
        (1 - dx) * dy * dz,
        dx * (1 - dy) * dz,
        dx * dy * dz     
    ]).reshape(2,2,2,1)

    
    # Interpolación ponderada de la dirección
    direccion_interpolada = np.sum(vectores_principales* pesos, axis=(0,1,2))

    norm_interpolada = np.linalg.norm(direccion_interpolada)

    # Normalizar el vector resultante
    
    if norm_interpolada > 0:
        return direccion_interpolada / norm_interpolada
    
    print(f"Dirección interpolada en {posicion} es cero")

    return None # Si no se encuentra dirección válida
    

# 6. Funcion para la  trayectoria 
def realizar_trayectoria(peaks,semilla, dwi_affine, tamaño_paso, angulo_maximo, longitud_maxima, mascara):
    
    #Propagación con el método de Euler

    trayectoria = [semilla]
    direccion_actual = None
    posicion_actual = np.array(semilla, dtype=float)
    longitud_acumulada = 0
    max_pasos = 500
    while len(trayectoria) < max_pasos:
        
        #Obtener eigenvector principal en la posicion actual
        
        direccion_interpolada = interpolar_direcciones(picos, np.round(posicion_actual).astype(int)) #para obtener el entero
        
        #REVISAR BIEN AQUI, SI ES NECESARIO TERMINAR LA TRAYECTORIA O ES POSIBLE BUSCAR
        #OTRA OPCION MEJOR
        
        if direccion_interpolada is None:
            print("No se encontró dirección válida")
            break

        # Verificar angulo con la dirección anterior
        if direccion_actual is not None:
            angulo = calcular_angulo(direccion_actual, direccion_interpolada)
            if angulo > angulo_maximo:
                print("Angulo excede el máximo permitido")
                break
        
        #**Aplicación del Método de Euler**

        nuevo_punto = posicion_actual + tamaño_paso * direccion_interpolada
        nuevo_voxel = np.round(nuevo_punto).astype(int).flatten()

        # Validar si el nuevo voxel se mantiene en la máscara
        if not (0 <= nuevo_voxel[0] < mascara.shape[0] and
                0 <= nuevo_voxel[1] < mascara.shape[1] and
                0 <= nuevo_voxel[2] < mascara.shape[2]):
            print("El punto se salió del volumen. Reintentando con otra dirección.")
            continue

        # Calculo de la distancia en mm
        punto_fisico_actual = nib.affines.apply_affine(dwi_affine, posicion_actual)
        punto_fisico_nuevo = nib.affines.apply_affine(dwi_affine,nuevo_voxel)
        distancia = np.linalg.norm(punto_fisico_nuevo-punto_fisico_actual)

        #Verificar si se mantiene la longitud permitida
        if longitud_acumulada + distancia > longitud_maxima:
            print(f"Longitud máxima alcanzada: {longitud_acumulada+distancia:.2f} mm. Terminando streamline.")
            break

        # Actualizar trayectoria
        longitud_acumulada += distancia
        trayectoria.append(nuevo_voxel.tolist())
        posicion_actual = nuevo_voxel
        direccion_actual = direccion_interpolada

    return trayectoria, longitud_acumulada

# 7. Bucle principal para generar la trayectoria total
trayectorias = []
max_semilla_intentos = 100
intentos_en_semilla = 0

while longitud_acumulada < longitud_minima:

    if intentos_en_semilla >= max_semilla_intentos:
        print("Número máximo de semillas intentadas alcanzado")
        break

    if tuple(semilla_actual) in semillas_exploradas:
        print(f"Semilla ya explorada: {semilla_actual}. Buscando otra.")
        intentos += 1
        intentos_en_semilla += 1
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]
        continue

    if mascara[tuple(semilla_actual)] == 0:
        print("Semilla fuera de la máscara. Buscando otra")
        intentos_en_semilla += 1
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]
        continue

    direccion_principal = interpolar_direcciones(picos, np.array(semilla_actual, dtype=float))
    if direccion_principal is None:
        print("Semilla sin dirección válida. Buscando otra")
        intentos_en_semilla += 1
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]
        continue

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
        print(f"Trayectoria válida generada con longitud {longitud_acumulada:.2f} mm.")
        trayectorias.append(trayectoria)
    else:
        print("Buscando nueva semilla.")
        intentos_en_semilla += 1
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]

# 8. Guardar trayectorias
if trayectorias:
    streamlines = Streamlines([np.array([nib.affines.apply_affine(dwi_affine, punto) for punto in t]) for t in trayectorias])
    sft = StatefulTractogram(streamlines, nib.Nifti1Image(dwi_datos, dwi_affine), Space.RASMM)
    save_tck(sft, "Trayectorias_finales.tck")
    print("Trayectorias guardadas con éxito.")
else:
    print("No se generaron trayectorias válidas.")