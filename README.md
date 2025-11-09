# 📊 Ejemplo Didáctico: Gaussian Mixture Model (GMM)

Este proyecto contiene un ejemplo completo y visual de cómo funciona un **Gaussian Mixture Model (GMM)** en Python. Está diseñado para ser didáctico y fácil de entender, ideal para explicar en clase o en una exposición de 15 minutos.

## 🎯 ¿Qué es un Gaussian Mixture Model?

Un **GMM** es un modelo probabilístico que asume que los datos provienen de una mezcla de varias distribuciones gaussianas (normales). A diferencia de K-means, el GMM:

- Asigna probabilidades de pertenencia a cada cluster (no solo una asignación rígida)
- Puede modelar clusters con formas elípticas y orientadas
- Proporciona información sobre la incertidumbre de las asignaciones

## 📋 Requisitos

- Python 3.7 o superior
- Las siguientes librerías (ver `requirements.txt`):
  - `numpy` >= 1.21.0
  - `matplotlib` >= 3.4.0
  - `scikit-learn` >= 1.0.0

## 🚀 Instalación y Ejecución

### Paso 1: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 2: Ejecutar el script

```bash
python3 gmm_ejemplo_didactico.py
```

O si estás usando Jupyter Notebook o VSCode, simplemente ejecuta las celdas o el archivo completo.

## 📦 Contenido del Proyecto

- **`gmm_ejemplo_didactico.py`**: Script principal con el ejemplo completo
- **`requirements.txt`**: Dependencias del proyecto
- **`README.md`**: Este archivo

## 🎨 ¿Qué hace el script?

1. **Genera datos sintéticos**: Crea un dataset 2D con 3 grupos usando `make_blobs`
2. **Entrena el modelo GMM**: Aprende las medias, covarianzas y pesos de 3 distribuciones gaussianas
3. **Visualiza los resultados**:
   - Puntos de datos coloreados según su cluster asignado
   - Elipses que representan las distribuciones gaussianas aprendidas
   - Centros de cada cluster marcados con 'X'
4. **Muestra probabilidades**: Imprime las probabilidades de pertenencia de varios puntos ejemplo
5. **Información del modelo**: Muestra pesos, medias y score del modelo

## 📊 Salida del Script

El script genera:

- **Visualización gráfica**: Una ventana con el gráfico interactivo y guarda `gmm_visualizacion.png` (alta resolución)
- **Información en consola**:
  - Estadísticas del dataset generado
  - Parámetros aprendidos del modelo (medias, pesos)
  - Probabilidades de pertenencia de puntos ejemplo
  - Métricas del modelo

## 🔍 Ejemplo de Visualización

La visualización muestra:
- **Puntos coloreados**: Cada color representa un cluster diferente
- **Elipses punteadas**: Representan las distribuciones gaussianas (2 desviaciones estándar)
- **Centros marcados**: Las 'X' indican el centro (media) de cada distribución
- **Leyenda**: Información sobre cada elemento del gráfico

## 💡 Conceptos Clave Explicados

- **Componentes**: Número de distribuciones gaussianas en la mezcla
- **Media (μ)**: Centro de cada distribución gaussiana
- **Covarianza (Σ)**: Define la forma y orientación de cada elipse
- **Pesos (π)**: Probabilidad a priori de cada componente
- **Probabilidades de pertenencia**: Probabilidad de que un punto pertenezca a cada cluster

## 📚 Uso Educativo

Este ejemplo es ideal para:
- Explicar el concepto de GMM de forma visual
- Mostrar la diferencia entre asignación rígida (K-means) y probabilística (GMM)
- Entender cómo funcionan los modelos de mezcla
- Visualizar distribuciones gaussianas en 2D

## 🛠️ Personalización

Puedes modificar fácilmente:
- **Número de clusters**: Cambia `n_components` en el modelo
- **Número de puntos**: Modifica `n_samples` en `make_blobs`
- **Forma de los clusters**: Ajusta `cluster_std` para cambiar la dispersión
- **Colores**: Modifica el array `colors` para usar otros colores

## 📝 Notas

- El script usa `random_state=42` para garantizar resultados reproducibles
- La visualización se guarda automáticamente como PNG de alta resolución
- Todos los pasos están comentados para facilitar la comprensión

## 🤝 Contribuciones

Siéntete libre de mejorar este ejemplo o agregar más visualizaciones y explicaciones.

---

**¡Disfruta aprendiendo sobre Gaussian Mixture Models!** 🎓

