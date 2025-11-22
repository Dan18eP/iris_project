import joblib
import numpy as np
import pandas as pd
import os


def load_trained_artifacts(
    model_path="models/best_model.joblib",
    scaler_path="models/scaler.joblib"
):
    """
    Loads the trained classification model and the scaler used during preprocessing.

    Returns:
        model: Trained ML model
        scaler: StandardScaler used in training
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def prepare_input_data(input_data):
    """
    Ensures new input data is in the correct format.

    Expected features:
        - SepalLengthCm
        - SepalWidthCm
        - PetalLengthCm
        - PetalWidthCm

    Input can be:
        - Python list
        - NumPy array
        - Pandas DataFrame
    """

    required_cols = [
        'SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm'
    ]

    #If input is already a DataFrame, ensure correct order
    if isinstance(input_data, pd.DataFrame):
        return input_data[required_cols]

    #If input is a list or array
    if isinstance(input_data, (list, np.ndarray)):
        return pd.DataFrame([input_data], columns=required_cols)

    raise ValueError("Input must be a list, NumPy array, or Pandas DataFrame.")


def predict_species(input_data):
    """
    Generates a species prediction for new flower measurements.

    Steps:
        1. Load trained model and scaler
        2. Format input data
        3. Scale features
        4. Predict species label

    Returns:
        str - predicted species name
    """

    #Load artifacts
    model, scaler = load_trained_artifacts()

    #Standardize input format
    X = prepare_input_data(input_data)

    #Scale using the same scaler from training
    X_scaled = scaler.transform(X)

    #Predict species
    prediction = model.predict(X_scaled)[0]

    return prediction



# Validation Helpers


def validate_measurement(label, value, min_val=0, max_val=None):
    """
    Validates that a measurement:
        - Is a number
        - Is greater than 0
        - (Optional) Is within known Iris ranges

    Args:
        label (str): Name of the measurement
        value (str): Raw input value
        min_val (float): Minimum allowed value
        max_val (float or None): Maximum allowed value

    Returns:
        float: Validated numeric value
    """
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"{label} must be a valid number.")

    if value <= min_val:
        raise ValueError(f"{label} must be greater than {min_val}.")

    if max_val is not None and value > max_val:
        raise ValueError(f"{label} cannot exceed {max_val}.")

    return value


if __name__ == "__main__":

    print("\n=== Iris Flower Prediction ===\n")
    print("Enter measurements:")

    try:
        # Typical real value ranges of Iris dataset
        sl = validate_measurement(
            "Sepal Length (cm)",
            input("Sepal Length (cm): "),
            min_val=0,
            max_val=8.0
        )
        sw = validate_measurement(
            "Sepal Width (cm)",
            input("Sepal Width (cm): "),
            min_val=0,
            max_val=4.5
        )
        pl = validate_measurement(
            "Petal Length (cm)",
            input("Petal Length (cm): "),
            min_val=0,
            max_val=7.5
        )
        pw = validate_measurement(
            "Petal Width (cm)",
            input("Petal Width (cm): "),
            min_val=0,
            max_val=2.6
        )

    except ValueError as err:
        print(f"\nInput Error: {err}\n")
        exit(1)

    # Build sample
    sample = [sl, sw, pl, pw]

    result = predict_species(sample)

    print(f"\nPredicted Species: {result}\n")
