import subprocess

# Archivos de entrada y salida
tractografia = "CC.tck"
odf = "ISMRM_2023_b3000_ODF.nii"

# SIFT
out_tracks_sift = "salida_sift_1.tck"

# SIFT2
out_weights_sift2 = "salida_pesos_sift_2.txt"
filtered_sift2 = "filtrada_sift2.tck"

# Mapas de densidad
density_total = "densidad_total.nii"
density_sift = "densidad_sift.nii"
density_sift2 = "densidad_sift2.nii"

# Tractos eliminados
removed_sift = "tractos_eliminados_sift.tck"
removed_sift2 = "tractos_eliminados_sift2.tck"

# Paso 1: Aplicar SIFT:-fd_scale_gm: Densidad de fibras
subprocess.run(["tcksift", "-fd_scale_gm", tractografia, odf, out_tracks_sift, "-force"])

# Paso 2: Aplicar SIFT2: fd_scale_gm
subprocess.run(["tcksift2", "-fd_scale_gm", tractografia, odf, out_weights_sift2, "-force"])

# Filtrar tractos con pesos bajos usando SIFT2, a partir del archivo de pesos
subprocess.run([
    "tckedit", tractografia, "-tck_weights_in", out_weights_sift2,
    "-minweight", "0.5", filtered_sift2, "-force"
])

# Paso 3: Comparar tractos eliminados

# Tractos eliminados por SIFT
subprocess.run([
    "tckedit", tractografia, "-exclude", out_tracks_sift, removed_sift, "-force"
])

# Tractos eliminados por SIFT2
subprocess.run([
    "tckedit", tractografia, "-exclude", filtered_sift2, removed_sift2, "-force"
])

# Paso 4: Generar mapas de densidad
subprocess.run([
    "tckmap", tractografia, "-template", odf, density_total, "-force"
])

subprocess.run([
    "tckmap", out_tracks_sift, "-template", odf, density_sift, "-force"
])

subprocess.run([
    "tckmap", filtered_sift2, "-template", odf, density_sift2, "-force"
])

# Paso 5: Contar el número de tractos
def contar_tractos(archivo):
    result = subprocess.run(["tckinfo", archivo], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "count:" in line:
            return int(line.split(":")[1].strip())
    return None

count_total = contar_tractos(tractografia)
count_sift = contar_tractos(out_tracks_sift)
count_sift2 = contar_tractos(filtered_sift2)

print(f"Número de tractos originales: {count_total}")
print(f"Número de tractos tras SIFT: {count_sift}")
print(f"Número de tractos tras SIFT2 (filtro >= 0.5): {count_sift2}")

# Paso 6: Visualización y Análisis
print(f"Original: {tractografia}")
print(f"Filtrado SIFT: {out_tracks_sift}")
print(f"Filtrado SIFT2: {filtered_sift2}")
print(f"Mapas de densidad: {density_total}, {density_sift}, {density_sift2}")
