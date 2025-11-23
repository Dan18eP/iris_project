# iris_project

An end-to-end Iris species classification project featuring data preparation, model training, automated testing, and a polished Streamlit dashboard.

## Quick start

1. *(Optional)* Create and activate a virtual environment.
2. Install dependencies:

	```pwsh
	python -m pip install -r requirements.txt
	```

3. Train the models and persist the best one:

	```pwsh
	python -m src.model_train
	```

4. Launch the Streamlit dashboard:

	```pwsh
	streamlit run src/app.py
	```

## Project structure

```
iris_project/
├─ data/
│  └─ Iris.csv
├─ docs/
│  ├─ iris_eda_documentation.md
│  ├─ model_training_docs.md
│  └─ prediction_docs.md
├─ models/
│  ├─ best_model.joblib
│  ├─ label_encoder.joblib
│  └─ scaler.joblib
├─ src/
│  ├─ app.py
│  ├─ data_processing.py
│  ├─ eda.py
│  ├─ model_predict.py
│  ├─ model_train.py
│  ├─ preprocessing.py
│  └─ __pycache__/
├─ tests/
│  └─ prediction_test.py
├─ requirements.txt
└─ README.md
```

The core workflow is split across `src/`, persisted artifacts live in `models/`, and complementary documentation can be found in `docs/`.

## Project overview

This repository tackles the classic Iris dataset (150 samples, three species, four numeric measurements per flower) using a modern ML engineering stack:

- Data wrangling, validation, and feature engineering
- Model benchmarking across multiple algorithms
- Artifact persistence for inference
- Interactive visual analytics powered by Streamlit and Plotly

The currently deployed classifier is a linear Support Vector Machine that achieves 97.37% accuracy on the held-out test split.

## Data pipeline

| Stage | Location | Description |
| --- | --- | --- |
| **Ingestion** | `src/data_processing.py` | Loads `data/Iris.csv`, verifies required columns, and exposes a `load_iris` helper reused throughout the project. |
| **Preprocessing** | `src/preprocessing.py` | Encodes the `Species` column via `LabelEncoder`, splits the dataset with a stratified 75/25 train-test split, and standardizes numeric features using `StandardScaler`. The function returns train/test arrays plus the fitted scaler and encoder for downstream use. |
| **Model training** | `src/model_train.py` | Benchmarks Logistic Regression, linear SVM, KNN, and Random Forest. A stratified 5-Fold CV loop captures mean accuracy, then each model is fitted on the training data, evaluated on the test split, and summarized. The best-performing model (SVM) along with the scaler and label encoder are saved in `models/`. |
| **Inference utilities** | `src/model_predict.py` | Provides helpers for loading the persisted artifacts and running predictions programmatically (used by tests or other services). |

### Data preprocessing deep dive

1. **Integrity checks** – Ensures the canonical features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species`) exist.
2. **Label encoding** – Transforms species names into numeric labels so algorithms can interpret the categorical target.
3. **Stratified sampling** – Preserves the 50/50/50 class distribution when splitting into training and testing sets.
4. **Feature scaling** – Standardizes all numeric features to zero mean and unit variance, which benefits distance-based algorithms (SVM, KNN) and keeps feature ranges comparable.
5. **Artifact return** – The preprocessing function returns both datasets and fitted transformers, enabling consistent scaling during inference.

## Exploratory data analysis

- `src/eda.py` and `docs/iris_eda_documentation.md` walk through descriptive statistics, pairwise plots, and correlation studies.
- The key takeaway is that petal measurements possess the strongest discriminative power, whereas sepal width provides limited separation—insights that informed the model selection.

## Model training and evaluation

Running `python -m src.model_train` will:

1. Train the four candidate models listed above.
2. Report cross-validation metrics, test-set precision/recall/F1, and a formatted classification report.
3. Persist `models/best_model.joblib`, `models/scaler.joblib`, and `models/label_encoder.joblib` for reuse.

All metrics discussed in the dashboard (overall accuracy 97.37%, weighted F1 97.36%, per-class breakdowns) come from this script.

## Streamlit dashboard

`src/app.py` provides a multi-page UI with four sections:

1. **Home** – Project overview, dataset stats, and a curated species gallery.
2. **Prediction** – Adjustable sliders (0.2 cm increments) feed the trained SVM to generate real-time predictions, accompanied by species-specific imagery and measurement summaries.
3. **Data Analysis** – Interactive histograms, correlation heatmaps, and scatter-matrix visualizations to explore the dataset.
4. **Model Performance** – Key metrics, per-class comparisons, confusion matrix, and model benchmarking plots.

The app automatically adapts to the user’s light/dark OS preference and uses the artifacts saved in `models/`.

## Testing

Basic regression tests live in `tests/prediction_test.py`. Run them with:

```pwsh
python -m pytest tests/prediction_test.py
```

## Notes

- Re-run `python -m src.model_train` whenever you change preprocessing logic or want to refresh the trained artifacts.
- The Streamlit app assumes the trained model, scaler, and label encoder exist under `models/`; delete them if you need to regenerate from scratch.
- Additional documentation about EDA, training, and prediction flows is available in the `docs/` folder for deeper dives.

## Documentation
Comprehensive documentation is available in the docs/ directory:

- EDA Findings: Exploratory data analysis results
- Model Training: Training methodology and results
- Prediction Module: Documentation for prediction module
- Dashboard Guide: Dashboard features and deployment

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Fork the repository
1. Create your feature branch (git checkout -b feature/AmazingFeature)
2. Commit your changes (git commit -m 'Add some AmazingFeature')
3. Push to the branch (git push origin feature/AmazingFeature)
4. Open a Pull Request


# License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Daniel Echeverría
- Andres Negrete

(Systems engineering students, Universidad de la Costa, Barranquilla, Colombia)
