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
semilla = coordenadas[0]
print(f"Semilla seleccionada: {semilla}")

# Revisar si es valida
if mascara[semilla[0], semilla[1], semilla[2]] == 0:
    print("La semilla se encuentra fuera de la máscara, elegir una semilla válida.")
else:
    print("La semilla es válida.")

# 3. Definir parámetros

semilla_actual = semilla #Iniciar desde la semilla seleccionada
trayectoria = []    #Guardar puntos recorridos
semillas_exploradas = set() #Evitar que se repitan las semillas exploradas
max_intentos = 10 # busqueda de voxel
intentos = 0
longitud_minima = 10  # Longitud mínima en mm
longitud_acumulada = 0
paso_maximo = 100 #máxima cantidad de pasos
tamaño_paso = 0.5  #Tamaño de cada paso
angulo_maximo = 90   # angulo máximo entre vectores

# 4. Calcular el angulo entre vectores

def calcular_angulo(v1, v2):
    cos_angulo = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angulo, -1, 1)))

# 5. Generar la trayectoria

while longitud_acumulada < longitud_minima:
    # Evitar semillas repetidas
    if tuple(semilla_actual) in semillas_exploradas:
        print(f"Semilla ya explorada: {semilla_actual}. Buscando otra.")
        intentos += 1
        if intentos >= max_intentos:
            print("Se alcanzó el máximo número de intentos. Finalizando.")
            break
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]
        continue

    # Inicializar la dirección a partir de los picos del voxel

    semillas_exploradas.add(tuple(semilla_actual))
    direccion_actual = picos[semilla_actual[0], semilla_actual[1], semilla_actual[2], :, :]
    normas = np.linalg.norm(direccion_actual, axis=1) #calcular la norma de los 3 vectores
    indice_max = np.argmax(normas) #elegir el de mayor longitud
    direccion_actual = direccion_actual[indice_max] / np.linalg.norm(direccion_actual[indice_max]) #normalizar el seleccionado

    # Comenzar a trazar el camino
    for _ in range(paso_maximo):
        #cambiar a siguiente voxel
        nuevo_punto = semilla_actual + tamaño_paso * direccion_actual
        nuevo_voxel = np.round(nuevo_punto).astype(int).flatten()

        print(f"Punto actual: {semilla_actual}")
        print(f"Punto nuevo: {nuevo_voxel}")

        # Validar si se mantiene dentro de la máscara
        if (nuevo_voxel < 0).any() or (nuevo_voxel >= mascara.shape).any():
            print("El punto sale del volumen.")
            break
        if mascara[nuevo_voxel[0], nuevo_voxel[1], nuevo_voxel[2]] == 0:
            print("El punto está fuera de la máscara. Buscando una valida")
            continue

        # Probar todas las direcciones en el voxel
        direcciones = picos[nuevo_voxel[0], nuevo_voxel[1], nuevo_voxel[2], :, :]
        direccion_valida = False
        #Verificar magnitud
        for direccion in direcciones:
            if np.linalg.norm(direccion) == 0:
                continue
            direccion = direccion / np.linalg.norm(direccion)
            # Verificar el angulo entre la direccón actual y la nueva
            angulo = calcular_angulo(direccion_actual, direccion)
            if angulo <= angulo_maximo:
                direccion_actual = direccion
                direccion_valida = True
                break

        if not direccion_valida:
            print("No se encontraron direcciones válidas en el voxel actual. Buscando nueva semilla.")
            break

        # Verificar medidas fisicas (mm)

        punto_fisico_actual = nib.affines.apply_affine(dwi_affine, semilla_actual)
        punto_fisico_nuevo = nib.affines.apply_affine(dwi_affine, nuevo_voxel)
        
        # Obtener la distancia fisica entre los dos puntos
        distancia = np.linalg.norm(punto_fisico_nuevo - punto_fisico_actual)
        
        #Calcular longitud acumulada de toda la trayectoria
        longitud_acumulada += distancia
        
        #Actualizar trayectoria
        trayectoria.append(nuevo_voxel.tolist())
        semilla_actual = nuevo_voxel

        if longitud_acumulada >= longitud_minima:
            print(f"Longitud acumulada válida: {longitud_acumulada:.2f} mm.")
            break

    # Buscar nueva semilla si no se generó una trayectoria válida
    if longitud_acumulada < longitud_minima:
        print("Buscando nueva semilla.")
        semilla_actual = coordenadas[np.random.randint(0, len(coordenadas))]


# 6. Guardar la trayectoria si es válida

if longitud_acumulada >= longitud_minima:
    print(f"Trayectoria generada con {len(trayectoria)} puntos.")
    trayectoria_coordenadas = [nib.affines.apply_affine(dwi_affine, punto) for punto in trayectoria]
    streamlines = Streamlines([np.array(trayectoria_coordenadas)])
    if len(streamlines) > 0:
        sft = StatefulTractogram(streamlines, nib.Nifti1Image(dwi_datos, dwi_affine), Space.RASMM)
        save_tck(sft, "Trayectoria_primer_intento.tck")
        print("Trayectoria guardada con éxito.")
    else:
        print("No se generaron streamlines válidos. No se guardará.")
else:
    print("No se alcanzó la longitud mínima requerida. No se guardará.")
