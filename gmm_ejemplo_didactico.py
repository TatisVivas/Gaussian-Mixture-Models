"""
Ejemplo Didáctico: Gaussian Mixture Model (GMM)
===============================================
Este script muestra cómo funciona un GMM de forma visual y sencilla.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

# ============================================================================
# PASO 1: Generar datos sintéticos con clusters sobrepuestos
# ============================================================================
# Creamos un dataset con 3 grupos (clusters) en 2D con superposición moderada
# Esto es clave para diferenciar GMM de K-means: GMM maneja mejor la ambigüedad
print("Generando dataset sintético con 3 grupos SOBREPUESTOS...")
print("(Los clusters tienen superposición moderada para mostrar la ventaja del GMM sobre K-means)\n")

# Definimos centros manualmente con superposición moderada
centros_cercanos = np.array([
    [0, 0],      # Centro 1
    [4, 2],      # Centro 2 (con superposición moderada)
    [2, 4]       # Centro 3 (con superposición moderada)
])

X, y_true = make_blobs(
    n_samples=300,           # 300 puntos de datos
    centers=centros_cercanos, # Centros definidos manualmente (con superposición moderada)
    n_features=2,            # 2 dimensiones (X, Y)
    random_state=42,         # Semilla para reproducibilidad
    cluster_std=1.8          # Desviación estándar ALTA para más solapamiento
)

print(f"Dataset generado: {X.shape[0]} puntos en {X.shape[1]} dimensiones")
print(f"Grupos reales: {len(np.unique(y_true))} clusters\n")

# ============================================================================
# PASO 2: Entrenar el modelo Gaussian Mixture Model
# ============================================================================
print("Entrenando modelo GMM con 3 componentes...")
# Un GMM asume que los datos provienen de una mezcla de distribuciones gaussianas
# Cada componente es una distribución gaussiana con su propia media y covarianza
gmm = GaussianMixture(
    n_components=3,          # Número de distribuciones gaussianas (clusters)
    random_state=42,         # Semilla para reproducibilidad
    covariance_type='full'   # Tipo de covarianza: 'full' permite elipses orientadas
)

# Entrenamos el modelo: aprende las medias, covarianzas y pesos de cada gaussiana
gmm.fit(X)

print("Modelo entrenado exitosamente!")
print(f"Medias aprendidas:\n{gmm.means_}")
print(f"Pesos de cada componente: {gmm.weights_}\n")

# ============================================================================
# PASO 3: Predecir a qué cluster pertenece cada punto
# ============================================================================
# El GMM asigna cada punto al cluster con mayor probabilidad
y_pred = gmm.predict(X)
print(f"Puntos asignados a cada cluster: {np.bincount(y_pred)}\n")

# Comparación rápida con K-means para mostrar la diferencia
from sklearn.cluster import KMeans
print("Comparación con K-means (para mostrar la diferencia):")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)
print(f"K-means asigna: {np.bincount(y_kmeans)}")
print(f"GMM asigna:     {np.bincount(y_pred)}")
print("(Nota: Con clusters sobrepuestos, GMM maneja mejor la ambigüedad)\n")

# ============================================================================
# PASO 4: Función auxiliar para dibujar elipses de las gaussianas
# ============================================================================
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Dibuja una elipse que representa una distribución gaussiana 2D.
    
    Parámetros:
    - position: media (centro) de la gaussiana
    - covariance: matriz de covarianza (define forma y orientación)
    - ax: eje de matplotlib donde dibujar
    """
    if ax is None:
        ax = plt.gca()
    
    # Convertir covarianza a elipse
    # Los valores propios y vectores propios definen la orientación y tamaño
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s) * 2  # Factor 2 para 2 desviaciones estándar
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance) * 2
    
    # Dibujar la elipse
    for nsig in range(1, 3):  # Dibujamos 1 y 2 desviaciones estándar
        ax.add_patch(Ellipse(
            position, 
            nsig * width, 
            nsig * height,
            angle=angle, 
            **kwargs
        ))

# ============================================================================
# PASO 5: Visualización ANTES y DESPUÉS
# ============================================================================
print("Generando visualización ANTES y DESPUÉS...")

# Colores para cada cluster
colors = ['#2196F3', '#E91E63', '#FFC107']  # Azul, Rosado/Rojo, Amarillo

# Crear figura con dos subplots: antes y después
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
# PASO 6: Secuencia de imágenes del proceso
# ============================================================================
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
# PASO 7: Visualización completa final
# ============================================================================
print("\nGenerando visualización final completa...")

# Crear figura con tamaño adecuado para presentación
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Dibujar las elipses de las distribuciones gaussianas aprendidas
for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm.covariances_, colors)):
    # Elipse para 2 desviaciones estándar (cubre ~95% de los datos)
    draw_ellipse(
        mean, 
        covar, 
        ax=ax,
        alpha=0.3,           # Transparencia
        color=color,
        linewidth=2,
        linestyle='--'
    )
    # Marcar el centro (media) de cada gaussiana
    ax.scatter(mean[0], mean[1], c=color, s=200, marker='x', 
               linewidths=3, label=f'Centro Cluster {i+1}', zorder=5)

# Dibujar los puntos de datos coloreados según su cluster asignado
for i, color in enumerate(colors):
    mask = y_pred == i
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
# PASO 8: Mostrar probabilidades de pertenencia
# ============================================================================
print("=" * 70)
print("PROBABILIDADES DE PERTENENCIA (predict_proba)")
print("=" * 70)
print("\nEl GMM no solo asigna un cluster, sino que calcula la probabilidad")
print("de que cada punto pertenezca a cada uno de los clusters.\n")

# Seleccionar algunos puntos, especialmente en regiones de solapamiento
# Buscamos puntos que estén cerca de múltiples clusters
probabilidades_todas = gmm.predict_proba(X)
# Encontrar puntos con probabilidades balanceadas (más interesantes para mostrar)
ambiguedad = np.max(probabilidades_todas, axis=1) - np.min(probabilidades_todas, axis=1)
indices_ambiguos = np.argsort(ambiguedad)[:6]  # Los 6 puntos más ambiguos
indices_ejemplo = list(indices_ambiguos) + [0, 50, 100]  # Agregar algunos normales
indices_ejemplo = indices_ejemplo[:8]  # Limitar a 8 puntos
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
# PASO 9: Información adicional del modelo
# ============================================================================
print("=" * 70)
print("INFORMACIÓN DEL MODELO ENTRENADO")
print("=" * 70)
print(f"\nNúmero de componentes: {gmm.n_components}")
print(f"\nPesos de cada componente (probabilidad a priori):")
for i, peso in enumerate(gmm.weights_):
    print(f"  Componente {i+1}: {peso:.3f} ({peso*100:.1f}%)")

print(f"\nMedias (centros) de cada componente:")
for i, media in enumerate(gmm.means_):
    print(f"  Componente {i+1}: ({media[0]:.2f}, {media[1]:.2f})")

print(f"\nScore logarítmico promedio: {gmm.score(X):.2f}")
print("(Mayor es mejor - mide qué tan bien el modelo explica los datos)\n")

print("=" * 70)
print("¡Ejemplo completado exitosamente!")
print("=" * 70)

