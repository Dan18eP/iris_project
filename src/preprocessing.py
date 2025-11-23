import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.data_processing import load_iris


def load_data():
    """
    Loads the Iris dataset from a CSV file and checks basic integrity.
    """
    df = load_iris()

    #Ensure the required columns exist in the dataset
    required_cols = [
        'SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm', 'Species'
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' is missing from dataset."
            )

    return df


def preprocess_iris(df):
    """
    Performs data preprocessing for the Iris dataset.
    Steps:
        1. Remove duplicates (not necessary)
        2. Handle missing values (not necessary)
        3. Encode categorical labels
        3. Separate features and labels
        4. Train/Test split
        5. Standardize numerical features
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """

    #3. Encode Species column into numeric labels
    label_encoder = LabelEncoder()
    df['Species'] = label_encoder.fit_transform(df['Species'])
   
    #4. Separate features (X) and target variable (y)

    X = df[
        [
            'SepalLengthCm', 'SepalWidthCm',
            'PetalLengthCm', 'PetalWidthCm'
        ]
    ]
    y = df['Species']

 
    #5. Split into training and testing sets
    #Stratified split preserves class proportions

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,       # 25% of data for testing
        random_state=42,      # ensures reproducibility
        stratify=y            # keeps class distribution balanced
    )


    #6. Standardize numerical features
    #This ensures all variables share the same scale

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit scaler only on training data
    X_test = scaler.transform(X_test)        # Transform test data using same scaler

    return X_train, X_test, y_train, y_test, scaler, label_encoder


if __name__ == "__main__":
    #Load dataset
    df = load_data()

    #Run preprocessing pipeline
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_iris(df)

    print("Preprocessing completed successfully")
    print("Training set size:", X_train.shape)
    print("Test set size:", X_test.shape)