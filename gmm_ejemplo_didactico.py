"""
Ejemplo Didáctico: Gaussian Mixture Model (GMM)
===============================================
Este script muestra cómo funciona un GMM de forma visual y sencilla.

¿QUÉ ES UN GMM?
---------------
Un Gaussian Mixture Model (GMM) es un modelo probabilístico que asume que los datos
provienen de una mezcla de varias distribuciones gaussianas (normales). A diferencia
de K-means que hace asignaciones rígidas (un punto pertenece a UN solo cluster),
el GMM calcula PROBABILIDADES de pertenencia a cada cluster.

VENTAJAS DEL GMM:
- Maneja mejor la ambigüedad en regiones donde los clusters se solapan
- Proporciona información probabilística (no solo asignación)
- Puede modelar clusters con formas elípticas y orientadas
- Es más flexible que K-means para datos con superposición
"""

# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

# ============================================================================
# BLOQUE 1: GENERACIÓN DE DATOS SINTÉTICOS
# ============================================================================
# 
# ¿POR QUÉ DATOS SINTÉTICOS?
# ---------------------------
# Usamos datos sintéticos porque:
# 1. Conocemos la estructura real (podemos validar el modelo)
# 2. Podemos controlar el nivel de superposición entre clusters
# 3. Es más fácil de visualizar y explicar en 2D
#
# IMPORTANTE: Generamos clusters CON SUPERPOSICIÓN para demostrar la ventaja
# del GMM sobre K-means. Con clusters muy separados, ambos métodos funcionan
# similar, pero con superposición, GMM es superior porque maneja probabilidades.

print("Generando dataset sintético con 3 grupos SOBREPUESTOS...")
print("(Los clusters tienen superposición moderada para mostrar la ventaja del GMM sobre K-means)\n")

# Definimos los centros de los 3 clusters manualmente para el ejemplo
# Estos están lo suficientemente cerca para crear superposición
centros_cercanos = np.array([
    [0, 0],      # Centro del cluster 1
    [4, 2],      # Centro del cluster 2 (cerca del 1 para crear solapamiento)
    [2, 4]       # Centro del cluster 3 (cerca de ambos para crear solapamiento)
])

# make_blobs genera puntos agrupados alrededor de los centros especificados
# cluster_std=1.8 es ALTO, lo que significa que los puntos se dispersan mucho
# y crean zonas de superposición entre clusters
X, y_true = make_blobs(
    n_samples=300,           # Total de puntos a generar
    centers=centros_cercanos, # Dónde ubicar los centros de los clusters
    n_features=2,            # 2 dimensiones (X, Y) para visualización fácil
    random_state=42,         # Semilla para reproducibilidad (mismos resultados)
    cluster_std=1.8          # Desviación estándar ALTA = más dispersión = más solapamiento
)

print(f"Dataset generado: {X.shape[0]} puntos en {X.shape[1]} dimensiones")
print(f"Grupos reales: {len(np.unique(y_true))} clusters\n")

# ============================================================================
# BLOQUE 2: ENTRENAMIENTO DEL MODELO GMM
# ============================================================================
#
# ¿QUÉ HACE EL ENTRENAMIENTO?
# ---------------------------
# El algoritmo EM (Expectation-Maximization) aprende:
# 1. MEDIAS (μ): Los centros de cada distribución gaussiana
# 2. COVARIANZAS (Σ): La forma y orientación de cada elipse
# 3. PESOS (π): La probabilidad a priori de cada componente
#
# El modelo "aprende" estos parámetros iterativamente hasta converger.
# No necesita saber de antemano qué puntos pertenecen a qué cluster.

print("Entrenando modelo GMM con 3 componentes...")

# Creamos el modelo GMM especificando:
gmm = GaussianMixture(
    n_components=3,          # Sabemos que hay 3 clusters (en la práctica, esto se puede optimizar)
    random_state=42,         # Semilla para reproducibilidad
    covariance_type='full'   # 'full' permite elipses orientadas (más flexible que círculos)
)

# ENTRENAMIENTO: El modelo analiza TODOS los puntos y aprende:
# - Dónde están los centros de las 3 gaussianas
# - Qué forma tienen (elipses orientadas)
# - Qué peso tiene cada componente en la mezcla
gmm.fit(X)

print("Modelo entrenado exitosamente!")
print(f"Medias aprendidas:\n{gmm.means_}")
print(f"Pesos de cada componente: {gmm.weights_}\n")

# ============================================================================
# BLOQUE 3: PREDICCIÓN Y ASIGNACIÓN DE CLUSTERS
# ============================================================================
#
# ¿CÓMO FUNCIONA LA PREDICCIÓN?
# -----------------------------
# predict() asigna cada punto al cluster con MAYOR PROBABILIDAD.
# Pero internamente, el GMM calcula probabilidades para TODOS los clusters.
# Esto es diferente de K-means que solo hace asignación rígida (todo o nada).

# Asignación "hard": cada punto va al cluster más probable
y_pred = gmm.predict(X)
print(f"Puntos asignados a cada cluster: {np.bincount(y_pred)}\n")

# COMPARACIÓN CON K-MEANS (opcional, para demostrar la diferencia)
# -----------------------------------------------------------------
# K-means hace asignaciones rígidas: un punto pertenece a UN solo cluster.
# GMM calcula probabilidades: un punto puede tener 60% cluster 1, 30% cluster 2, 10% cluster 3.
# Con clusters sobrepuestos, esta diferencia es crucial.

from sklearn.cluster import KMeans
print("Comparación con K-means (para mostrar la diferencia):")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)
print(f"K-means asigna: {np.bincount(y_kmeans)}")
print(f"GMM asigna:     {np.bincount(y_pred)}")
print("(Nota: Con clusters sobrepuestos, GMM maneja mejor la ambigüedad)\n")

# ============================================================================
# BLOQUE 4: FUNCIÓN AUXILIAR PARA VISUALIZAR LAS GAUSSIANAS
# ============================================================================
#
# ¿POR QUÉ ELIPSES?
# -----------------
# Una distribución gaussiana 2D tiene forma de elipse (no círculo).
# La matriz de covarianza define:
# - El tamaño de la elipse (valores propios)
# - La orientación de la elipse (vectores propios)
#
# Esta función convierte la matriz de covarianza aprendida en una elipse visual.

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Dibuja una elipse que representa una distribución gaussiana 2D.
    
    La elipse muestra dónde está concentrada la probabilidad de cada cluster.
    Una elipse más grande = más dispersión de los datos.
    Una elipse orientada = correlación entre las dimensiones X e Y.
    
    Parámetros:
    - position: media (centro) de la gaussiana
    - covariance: matriz de covarianza (define forma y orientación)
    - ax: eje de matplotlib donde dibujar
    """
    if ax is None:
        ax = plt.gca()
    
    # Descomposición SVD: convierte la matriz de covarianza en:
    # - Valores propios (s): tamaño de la elipse
    # - Vectores propios (U): orientación de la elipse
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))  # Ángulo de rotación
        width, height = 2 * np.sqrt(s) * 2  # Ancho y alto (2 desviaciones estándar)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance) * 2
    
    # Dibujar la elipse para 1 y 2 desviaciones estándar
    # 1 desviación estándar ≈ 68% de los datos
    # 2 desviaciones estándar ≈ 95% de los datos
    for nsig in range(1, 3):
        ax.add_patch(Ellipse(
            position, 
            nsig * width, 
            nsig * height,
            angle=angle, 
            **kwargs
        ))

# ============================================================================
# BLOQUE 5: VISUALIZACIÓN COMPARATIVA (ANTES Y DESPUÉS)
# ============================================================================
#
# Esta visualización muestra el poder del GMM:
# - IZQUIERDA: Datos sin clasificar (solo puntos grises)
# - DERECHA: Datos clasificados (colores por cluster)
#
# Es útil para mostrar cómo el modelo "descubrió" la estructura oculta.

print("Generando visualización ANTES y DESPUÉS...")

# Definimos colores distintivos para cada cluster
colors = ['#2196F3', '#E91E63', '#FFC107']  # Azul, Rosado/Rojo, Amarillo

# Crear figura con dos subplots lado a lado para comparación
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- SUBPLOT 1: ANTES (datos originales sin clasificar) ---
ax1.scatter(X[:, 0], X[:, 1], c='gray', s=50, alpha=0.6, 
           edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Característica X', fontsize=12, fontweight='bold')
ax1.set_ylabel('Característica Y', fontsize=12, fontweight='bold')
ax1.set_title('ANTES: Datos Originales\n(Sin clasificar)', 
             fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal', adjustable='box')

# --- SUBPLOT 2: DESPUÉS (datos clasificados con GMM) ---
for i, color in enumerate(colors):
    mask = y_pred == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=color, s=50, 
               alpha=0.6, edgecolors='black', linewidth=0.5,
               label=f'Cluster {i+1} ({np.sum(mask)} puntos)')

ax2.set_xlabel('Característica X', fontsize=12, fontweight='bold')
ax2.set_ylabel('Característica Y', fontsize=12, fontweight='bold')
ax2.set_title('DESPUÉS: Datos Clasificados con GMM\n(3 clusters identificados)', 
             fontsize=13, fontweight='bold', pad=15)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('gmm_antes_despues.png', dpi=300, bbox_inches='tight')
print("Visualización 'antes y después' guardada como 'gmm_antes_despues.png'\n")
plt.close()

# ============================================================================
# BLOQUE 6: SECUENCIA DE IMÁGENES DEL PROCESO
# ============================================================================
#
# Generamos 3 imágenes que muestran el proceso paso a paso:
# 1. Datos originales (sin procesar)
# 2. Clasificación (asignación de clusters)
# 3. Visualización completa (con elipses gaussianas)
#
# Esto ayuda a entender el flujo completo del algoritmo.

print("Generando secuencia de imágenes del proceso...")

# --- IMAGEN 1: Datos originales ---
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
ax1.scatter(X[:, 0], X[:, 1], c='gray', s=50, alpha=0.6, 
           edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Característica X', fontsize=12, fontweight='bold')
ax1.set_ylabel('Característica Y', fontsize=12, fontweight='bold')
ax1.set_title('Paso 1: Datos Originales\n(300 puntos sin clasificar)', 
             fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('gmm_paso1_datos_originales.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Paso 1 guardado: 'gmm_paso1_datos_originales.png'")

# --- IMAGEN 2: Datos clasificados (sin elipses) ---
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
for i, color in enumerate(colors):
    mask = y_pred == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=color, s=50, 
               alpha=0.6, edgecolors='black', linewidth=0.5,
               label=f'Cluster {i+1} ({np.sum(mask)} puntos)')
ax2.set_xlabel('Característica X', fontsize=12, fontweight='bold')
ax2.set_ylabel('Característica Y', fontsize=12, fontweight='bold')
ax2.set_title('Paso 2: Clasificación con GMM\n(Asignación de clusters)', 
             fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('gmm_paso2_clasificacion.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Paso 2 guardado: 'gmm_paso2_clasificacion.png'")

# --- IMAGEN 3: Visualización final completa (con elipses) ---
print("  ✓ Paso 3: Generando visualización final completa...")

# ============================================================================
# BLOQUE 7: VISUALIZACIÓN FINAL COMPLETA
# ============================================================================
#
# Esta es la visualización más importante porque muestra:
# 1. Los puntos clasificados (colores)
# 2. Las elipses gaussianas aprendidas (distribuciones)
# 3. Los centros de cada cluster (marcados con X)
#
# Las elipses muestran la "zona de influencia" de cada gaussiana.
# Donde las elipses se solapan, hay ambigüedad (puntos con probabilidades balanceadas).

print("\nGenerando visualización final completa...")

# Crear figura con tamaño adecuado para presentación
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# PRIMERO: Dibujar las elipses de las distribuciones gaussianas aprendidas
# Estas elipses representan la "forma" de cada cluster aprendida por el modelo
for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm.covariances_, colors)):
    # La elipse muestra dónde está concentrada la probabilidad
    # 2 desviaciones estándar cubren aproximadamente el 95% de los datos
    draw_ellipse(
        mean,      # Centro de la elipse (media aprendida)
        covar,     # Forma y orientación (covarianza aprendida)
        ax=ax,
        alpha=0.3, # Transparencia para ver el solapamiento
        color=color,
        linewidth=2,
        linestyle='--'
    )
    # Marcar el centro (media) de cada gaussiana con una X grande
    ax.scatter(mean[0], mean[1], c=color, s=200, marker='x', 
               linewidths=3, label=f'Centro Cluster {i+1}', zorder=5)

# SEGUNDO: Dibujar los puntos de datos coloreados según su cluster asignado
# Cada punto se colorea según el cluster al que fue asignado (mayor probabilidad)
for i, color in enumerate(colors):
    mask = y_pred == i  # Máscara booleana: True para puntos del cluster i
    ax.scatter(X[mask, 0], X[mask, 1], c=color, s=50, 
              alpha=0.6, edgecolors='black', linewidth=0.5,
              label=f'Cluster {i+1} ({np.sum(mask)} puntos)')

# Configuración de la visualización
ax.set_xlabel('Característica X', fontsize=12, fontweight='bold')
ax.set_ylabel('Característica Y', fontsize=12, fontweight='bold')
ax.set_title('Gaussian Mixture Model (GMM)\n',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('gmm_paso3_visualizacion_final.png', dpi=300, bbox_inches='tight')
plt.savefig('gmm_visualizacion.png', dpi=300, bbox_inches='tight')  # Mantener nombre original
print("  ✓ Paso 3 guardado: 'gmm_paso3_visualizacion_final.png'")
print("Visualización final guardada como 'gmm_visualizacion.png'\n")
plt.show()

# ============================================================================
# BLOQUE 8: PROBABILIDADES DE PERTENENCIA (LA VENTAJA CLAVE DEL GMM)
# ============================================================================
#
# ESTE ES EL BLOQUE MÁS IMPORTANTE PARA ENTENDER LA DIFERENCIA CON K-MEANS
# -------------------------------------------------------------------------
# K-means solo dice: "Este punto pertenece al cluster 1" (asignación rígida)
# GMM dice: "Este punto tiene 60% probabilidad cluster 1, 30% cluster 2, 10% cluster 3"
#
# Esto es especialmente útil en regiones donde los clusters se solapan.
# Un punto en el borde entre dos clusters tendrá probabilidades balanceadas.

print("=" * 70)
print("PROBABILIDADES DE PERTENENCIA (predict_proba)")
print("=" * 70)
print("\nEl GMM no solo asigna un cluster, sino que calcula la probabilidad")
print("de que cada punto pertenezca a cada uno de los clusters.\n")

# Calculamos probabilidades para TODOS los puntos
probabilidades_todas = gmm.predict_proba(X)  # Matriz: [n_puntos x n_clusters]

# Buscamos puntos "ambiguos" (con probabilidades balanceadas)
# Estos son los más interesantes porque muestran la ventaja del GMM
# Un punto ambiguo tiene probabilidades similares para varios clusters
ambiguedad = np.max(probabilidades_todas, axis=1) - np.min(probabilidades_todas, axis=1)
# Menor diferencia = más ambiguo (ej: [0.4, 0.35, 0.25] es más ambiguo que [0.9, 0.05, 0.05])

indices_ambiguos = np.argsort(ambiguedad)[:6]  # Los 6 puntos más ambiguos
indices_ejemplo = list(indices_ambiguos) + [0, 50, 100]  # Agregar algunos puntos "normales"
indices_ejemplo = indices_ejemplo[:8]  # Limitar a 8 puntos para no saturar la salida
probabilidades = gmm.predict_proba(X[indices_ejemplo])

print("Punto | Coordenadas (X, Y) | Probabilidades [Cluster 1, Cluster 2, Cluster 3] | Cluster Asignado")
print("-" * 100)

for idx, (i, prob) in enumerate(zip(indices_ejemplo, probabilidades)):
    cluster_asignado = np.argmax(prob)
    print(f"  {idx+1:2d}  | ({X[i,0]:6.2f}, {X[i,1]:6.2f}) | "
          f"[{prob[0]:.3f}, {prob[1]:.3f}, {prob[2]:.3f}] | "
          f"Cluster {cluster_asignado + 1} ({prob[cluster_asignado]*100:.1f}%)")

print("\n" + "=" * 70)
print("INTERPRETACIÓN:")
print("=" * 70)
print("- Cada fila muestra un punto de datos y sus probabilidades de pertenencia.")
print("- Las probabilidades suman 1.0 (100%) para cada punto.")
print("- El cluster asignado es aquel con mayor probabilidad.")
print("- Puntos cerca de los bordes entre clusters tienen probabilidades más balanceadas.")
print("- Puntos cerca del centro de un cluster tienen probabilidad alta en ese cluster.")
print("\n💡 VENTAJA DEL GMM SOBRE K-MEANS:")
print("   Con clusters sobrepuestos, GMM proporciona probabilidades suaves,")
print("   mientras que K-means hace asignaciones rígidas (todo o nada).")
print("   Esto es especialmente útil en regiones ambiguas donde los clusters se solapan.\n")

# ============================================================================
# BLOQUE 9: INFORMACIÓN DEL MODELO ENTRENADO
# ============================================================================
#
# Este bloque muestra los parámetros que el modelo aprendió durante el entrenamiento.
# Es útil para entender qué "vio" el algoritmo en los datos.

print("=" * 70)
print("INFORMACIÓN DEL MODELO ENTRENADO")
print("=" * 70)

# Número de componentes (clusters) que especificamos
print(f"\nNúmero de componentes: {gmm.n_components}")

# PESOS (π): Probabilidad a priori de cada componente
# Indica qué proporción de los datos pertenece a cada cluster
# Si todos los pesos son similares (≈0.33), los clusters tienen tamaños similares
print(f"\nPesos de cada componente (probabilidad a priori):")
for i, peso in enumerate(gmm.weights_):
    print(f"  Componente {i+1}: {peso:.3f} ({peso*100:.1f}%)")

# MEDIAS (μ): Los centros aprendidos de cada distribución gaussiana
# Estos son los "centroides" que el modelo encontró
print(f"\nMedias (centros) de cada componente:")
for i, media in enumerate(gmm.means_):
    print(f"  Componente {i+1}: ({media[0]:.2f}, {media[1]:.2f})")

# SCORE: Mide qué tan bien el modelo explica los datos
# Es el logaritmo de la verosimilitud promedio (mayor es mejor)
# Un score más alto significa que el modelo "encaja" mejor con los datos
print(f"\nScore logarítmico promedio: {gmm.score(X):.2f}")
print("(Mayor es mejor - mide qué tan bien el modelo explica los datos)\n")

print("=" * 70)
print("¡Ejemplo completado exitosamente!")
print("=" * 70)

