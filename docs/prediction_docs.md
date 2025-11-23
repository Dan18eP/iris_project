# Prediction Module - Iris Classification

## Overview

The prediction module (`src/predict.py`) provides a production-ready interface for classifying Iris flowers using the trained SVM model. It handles model loading, input validation, feature scaling, and prediction generation.

---

## Core Functions

### 1. `load_trained_artifacts()`

Loads the saved model and scaler from disk.

```python
model, scaler = load_trained_artifacts(
    model_path="models/best_model.joblib",
    scaler_path="models/scaler.joblib"
)
```

**Returns:**
- `model`: Trained SVM classifier
- `scaler`: StandardScaler fitted on training data

**Error Handling:** Raises `FileNotFoundError` if model or scaler files are missing.

---

### 2. `prepare_input_data(input_data)`

Converts various input formats to a standardized DataFrame.

**Accepts:**
- Python list: `[5.1, 3.5, 1.4, 0.2]`
- NumPy array: `np.array([5.1, 3.5, 1.4, 0.2])`
- Pandas DataFrame with feature columns

**Returns:** DataFrame with columns in correct order:
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm

---

### 3. `predict_species(input_data)`

**Main prediction function.** End-to-end pipeline from raw input to species prediction.

```python
# Example usage
sample = [5.1, 3.5, 1.4, 0.2]
species = predict_species(sample)
print(species)  # Output: "Iris-setosa"
```

**Pipeline:**
1. Load model and scaler
2. Format input data
3. Apply standardization (same scaling as training)
4. Generate prediction

**Returns:** String with species name (e.g., "Iris-setosa")

---

### 4. `validate_measurement()`

Validates individual feature inputs with range checks.

```python
sepal_length = validate_measurement(
    label="Sepal Length (cm)",
    value=user_input,
    min_val=0,
    max_val=8.0
)
```

**Validation Rules:**
- Must be numeric (float conversion)
- Must be > 0
- Must be within biological ranges (optional)

**Raises:** `ValueError` with descriptive message if validation fails.

---

## Valid Input Ranges

Based on Iris dataset statistics:

| Feature | Min | Max | Typical Range |
|---------|-----|-----|---------------|
| Sepal Length | 0 | 8.0 cm | 4.3-7.9 cm |
| Sepal Width | 0 | 4.5 cm | 2.0-4.4 cm |
| Petal Length | 0 | 7.5 cm | 1.0-6.9 cm |
| Petal Width | 0 | 2.6 cm | 0.1-2.5 cm |

**Note:** Values outside typical ranges will still produce predictions but may be less reliable.

---

## Usage Examples

### Interactive Mode (Command Line)

```bash
python src/predict.py
```

```
=== Iris Flower Prediction ===

Enter measurements:
Sepal Length (cm): 5.1
Sepal Width (cm): 3.5
Petal Length (cm): 1.4
Petal Width (cm): 0.2

Predicted Species: Iris-setosa
```

### Programmatic Usage

```python
from predict import predict_species

# Single prediction
sample = [6.5, 3.0, 5.5, 1.8]
result = predict_species(sample)
print(result)  # "Iris-virginica"

# Multiple predictions
samples = [
    [5.1, 3.5, 1.4, 0.2],  # setosa
    [6.0, 2.7, 5.1, 1.6],  # versicolor
    [6.5, 3.0, 5.5, 1.8]   # virginica
]

for sample in samples:
    print(predict_species(sample))
```

### DataFrame Input

```python
import pandas as pd

df = pd.DataFrame({
    'SepalLengthCm': [5.1, 6.0, 6.5],
    'SepalWidthCm': [3.5, 2.7, 3.0],
    'PetalLengthCm': [1.4, 5.1, 5.5],
    'PetalWidthCm': [0.2, 1.6, 1.8]
})

predictions = [predict_species(row) for _, row in df.iterrows()]
```

---

## Error Handling

### Missing Model Files

```python
FileNotFoundError: Model file not found: models/best_model.joblib
```

**Solution:** Run `python src/train_models.py` to train and save the model.

### Invalid Input Format

```python
ValueError: Input must be a list, NumPy array, or Pandas DataFrame.
```

**Solution:** Ensure input is in one of the supported formats.

### Invalid Measurements

```python
ValueError: Sepal Length (cm) must be a valid number.
ValueError: Petal Width (cm) must be greater than 0.
ValueError: Sepal Length (cm) cannot exceed 8.0.
```

**Solution:** Provide numeric values within valid ranges.

---

## Key Design Decisions

### 1. Feature Scaling is Mandatory
The SVM model was trained on standardized features. All predictions must use the **same scaler** to maintain consistent feature distributions.

**Wrong:**
```python
# This will produce incorrect predictions
model.predict([[5.1, 3.5, 1.4, 0.2]])
```

**Correct:**
```python
# Scale first, then predict
X_scaled = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
model.predict(X_scaled)
```

### 2. Flexible Input Formats
Supports multiple input types to accommodate different use cases:
- **Lists:** Simple API calls
- **NumPy arrays:** Scientific computing workflows
- **DataFrames:** Batch predictions, CSV imports

### 3. Input Validation
Range checks prevent obviously invalid inputs while allowing slight extrapolation beyond training data ranges.

---

## Integration Guidelines

### REST API Integration (Optional)

```python
from flask import Flask, request, jsonify
from predict import predict_species

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.json
    sample = [
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]
    
    try:
        species = predict_species(sample)
        return jsonify({'species': species})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
```

### Batch Processing

```python
import pandas as pd

# Load new data
new_data = pd.read_csv('new_measurements.csv')

# Generate predictions
new_data['Predicted_Species'] = new_data.apply(
    lambda row: predict_species(row), axis=1
)

# Save results
new_data.to_csv('predictions.csv', index=False)
```

---

## Testing Recommendations

### Unit Tests

```python
def test_predict_setosa():
    # Known setosa sample
    sample = [5.1, 3.5, 1.4, 0.2]
    assert predict_species(sample) == "Iris-setosa"

def test_predict_virginica():
    # Known virginica sample
    sample = [6.5, 3.0, 5.5, 1.8]
    assert predict_species(sample) == "Iris-virginica"

def test_invalid_input():
    with pytest.raises(ValueError):
        validate_measurement("Test", "not_a_number", 0, 10)
```

### Edge Cases to Test

1. **Boundary values:** Features at min/max ranges
2. **Outliers:** Values beyond training data distribution
3. **Missing features:** Incomplete input arrays
4. **Wrong order:** Features in incorrect sequence

---

## Performance Considerations

- **Prediction time:** < 1ms per sample (SVM linear kernel)
- **Model loading:** ~50ms (one-time cost, cache in production)
- **Memory footprint:** ~1MB (model + scaler)
- **Throughput:** 1000+ predictions/second (CPU)

**Optimization tip:** Load model once at startup, reuse for all predictions.

```python
# Good: Load once
model, scaler = load_trained_artifacts()
for sample in batch:
    X_scaled = scaler.transform([sample])
    prediction = model.predict(X_scaled)[0]

# Bad: Reload every time
for sample in batch:
    prediction = predict_species(sample)  # Reloads model each time
```

---

## Conclusion

The prediction module provides a robust, production-ready interface for Iris classification with:
-  Automatic feature scaling
-  Flexible input formats
-  Input validation and error handling
-  Simple API for integration (optional)

Expected accuracy: **97.37%** based on test set evaluation.