# Human Activity Tracker – Entrega 3

Proyecto de clasificación de actividades humanas (sit, stand, turn, walk) usando **MediaPipe Pose**, **RandomForest** y un demo de clasificación **en tiempo real** con cámara web.

---

## 1. Requerimientos

- Python 3.10 o superior
- Git (opcional, para clonar el repositorio)
- Cámara web (para el demo en tiempo real)
- Sistema probado principalmente en **Windows**

Todas las librerías necesarias están en `requirements.txt`.

---

## 2. Crear y activar el entorno virtual (.venv)

1. Abrir una terminal en la carpeta raíz del proyecto (donde está `requirements.txt`).

2. Crear el entorno virtual:

```bash
py -m venv .venv
````

3. Activar el entorno:

```bash
.\.venv\Scripts\activate
```

4. Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> En Mac/Linux:
>
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> ```

---

## 3. Cómo ejecutar la aplicación en tiempo real

1. Se debe tener el entorno `.venv` **activo**.

2. Verifica que exista el modelo entrenado en:

   ```text
   models/rf_pose_classifier.pkl
   ```

3. Ejecutar el demo en tiempo real:

```bash
py -m src.models.realtime_demo
```

Se abrirá una ventana de OpenCV con:

* el video de la cámara,
* el esqueleto de MediaPipe dibujado,
* y el texto con la actividad predicha (`sit`, `stand`, `turn`, `walk`), suavizada en el tiempo.

Para cerrar el demo, presiona la tecla **q** sobre la ventana de video.

---

## 4. Pipeline completo (opcional): de videos crudos a modelo entrenado

Si quieres repetir todo el proceso desde cero (por ejemplo, con tus propios videos), el flujo es:

### 4.1. Colocar los videos crudos

Poner los archivos `.mp4` en:

```text
data/raw/
```

Se puede seguir la convención usada en el proyecto:

* `sit_down_p01_t01.mp4`
* `stand_up_p01_t01.mp4`
* `turn_left_p01_t01.mp4`
* `turn_right_p01_t01.mp4`
* `walk_away_p01_t01.mp4`
* `walk_toward_p01_t01.mp4`
* etc.

No es obligatorio copiar exactamente estos nombres, pero sí es importante que los videos estén en `data/raw` y que el script pueda inferir la actividad a partir del nombre (nombres que empiezen con "walk", "sit", "stand", "turn").

### 4.2. Extraer landmarks con MediaPipe (CSV por video)

Desde la raíz del proyecto, con el entorno activado:

```bash
py -m src.data.extract_landmarks
```

Esto:

* recorre todos los `.mp4` en `data/raw/`,
* ejecuta MediaPipe Pose,
* y genera un `.csv` por video en `data/processed/`, con los *landmarks* por frame.

### 4.3. Construir el dataset unificado

```bash
py -m src.data.build_dataset
```

Esto:

* lee todos los CSV de `data/processed/`,
* los combina en un solo archivo:

  ```text
  data/processed/all_landmarks.csv
  ```
* añade información sobre la actividad y el video de origen (`source`).

### 4.4. Entrenar el modelo y guardar el `.pkl`

```bash
py -m src.models.train_baseline
```

Este script:

* carga `data/processed/all_landmarks.csv`,
* genera características (landmarks normalizados + velocidades),
* realiza la partición de entrenamiento/prueba por video (GroupShuffleSplit),
* entrena dos modelos:

  * SVM (RBF),
  * RandomForest (tuned),
* imprime métricas (accuracy, precision, recall, F1 macro),
* guarda matrices de confusión en `reports/figures/`,
* **serializa el RandomForest final** en:

  ```text
  models/rf_pose_classifier.pkl
  ```

Este archivo `.pkl` es el que luego usa `realtime_demo` para clasificar en tiempo real.

---

## 5. Estructura de carpetas (resumen)

```text
.
├── data
│   ├── raw/              # Videos .mp4 de entrada
│   └── processed/        # CSV por video + all_landmarks.csv
├── models/
│   └── rf_pose_classifier.pkl   # Modelo entrenado (RandomForest)
├── reports/
│   └── figures/          # Matrices de confusión y figuras
├── src/
│   ├── data/
│   │   ├── extract_landmarks.py
│   │   └── build_dataset.py
│   └── models/
│       ├── train_baseline.py
│       └── realtime_demo.py
├── requirements.txt
└── README.md
```