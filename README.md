# Detección y reconocimiento facial

## Versiones:

- Versión 1: Únicamente se realiza la detección de los rostros. (Haarcascade)
- Versión 2: Se realiza la detección e identificación de rostros sin preprocesar las imágenes. (Haarcascade + Multi-Layer Perceptron)
- Versión 3: Se realiza la detección e identificación de rostros obteniendo las distancias apropiadas del rostro. (DLIB + Logistic Regression)

## Modo de uso:

### Ver una demostración

1. Ejecutar siguiente línea para ver prueba con un modelo entrenado y un vídeo pregrabado (seleccionar versión):
   ```
   python3 main.py example v1
   ```
   ```
   python3 main.py example v2
   ```
   ```
   python3 main.py example v3
   ```
   Es posible hacer una prueba con un vídeo propio introduciendo el vídeo en 'data/test' con el nombre 'EXAMPLE.MOV'

### Entrenar modelo nuevo con nuevos rostros
1. Introducir en el directorio 'data/train/' los vídeos de las personas que se desean detectar. Es necesario que el nombre del video lleve el nombre de la persona. (Ejemplo: Pedro.MOV, Maria.mp4, etc.)
2. Ejecutar siguiente línea para recolectar los datos de los vídeos generando un nuevo dataset y entrenar el mejor modelo (seleccionar version):
    ```
    python3 main.py train v2
   ```
   ```
    python3 main.py train v3
    ```

### Probar modelo en tiempo real

1. Ejecutar siguiente línea para probar funcionamiento en tiempo real (seleccionar versión):
    ```
    python3 main.py test v1
   ```
   ```
    python3 main.py test v2
   ```
   ```
    python3 main.py test v3
    ```
   (para las versiones 2 y 3 es necesario haber preentrenado un modelo nuevo con el comando 'train')

## Estructura del código

- Directorio data/: Contiene los datos con los que entrenar o probar el modelo.
  - data/train/: Vídeos de las personas para el entrenamiento.
  - data/test/: Video sobre el que testear si no se desea hacerlo en tiempo real. (EXAMPLE.MOV)
- Directorio models/: Contiene los modelos entrenados. (detector de landmarks, v2, v3 y examples)
- Directorio scipts/: Contiene el código.
  - Data.py: Realiza funciones relacionadas con las imágenes (recolectar datos de entrenamiento con Haarcarsade o DLIB)
  - Recognition.py: Extrae información de las imágenes (detección de rostros, de landmarks, dibujado sobre la imagen, etc.).
  - Model.py: Realiza funciones relacionas con los clasificadores (entrenar, evaluar, cargar, guardar, etc.).
  - ML-dist_optimization.ipynb: Ejemplo del preprocesado realizado para la optimizción del dataset (Machine Learning).
