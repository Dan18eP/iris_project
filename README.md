# iris_project

Estructura principal del proyecto Iris (creada automáticamente).

Estructura:

```
iris_project/
├─ data/
│  └─ iris.csv
├─ src/
│  ├─ data_processing.py
│  ├─ eda.py
│  ├─ model_train.py
│  ├─ utils.py
│  └─ app.py
├─ models/
│  └─ iris_rf.joblib
├─ requirements.txt
└─ README.md
```

Cómo usar:

- Instalar dependencias (recomiendo usar un virtualenv):

```pwsh
python -m pip install -r requirements.txt
```

- Entrenar y guardar el modelo:

```pwsh
python -m src.model_train
```

- Ejecutar la API Flask (usa el modelo guardado en `models/iris_rf.joblib`):

```pwsh
streamlit run src/app.py
```

Notas:

- `models/iris_rf.joblib` es un placeholder. Después de entrenar con `model_train` se sobrescribirá con el modelo real.
- Los archivos en `src/` contienen funciones base para cargar datos, EDA, entrenamiento y una API mínima.
