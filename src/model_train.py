import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from data_processing import load_iris
from preprocessing import preprocess_iris
import joblib


def train_models():
    """
    Train multiple classification models and compare performance.
    Models trained:
        - Logistic Regression
        - SVM
        - KNN
        - Random Forest

    Most discriminative features are Petal Length and Width,
    but all 4 numeric features are used for maximum comparability.
    """


    #1. Load preprocessed data
    
    df = load_iris()
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_iris(df)


    #2. Define the classification models

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "SVM": SVC(kernel='linear', probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }


    #3. Evaluation framework

    #Stratified K-Fold ensures equal class balance in small datasets.

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

  
    #4. Train models and evaluate using cross-validation + test performance

    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")

        #Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        #Fit the model
        model.fit(X_train, y_train)

        #Predict on test set
        y_pred = model.predict(X_test)

        #Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Test Accuracy:  {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1 Score:  {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        results.append({
            "Model": model_name,
            "CV Accuracy Mean": cv_scores.mean(),
            "Test Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Trained Model": model
        })


    #5. Select best model based on Test Accuracy
    
    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
    best_model_row = results_df.iloc[0]
    best_model = best_model_row["Trained Model"]

    print("\n==============================")
    print(" FINAL MODEL COMPARISON TABLE")
    print("==============================")
    print(results_df[["Model", "CV Accuracy Mean", "Test Accuracy", "F1 Score"]])

    print("\nBest model:", best_model_row["Model"])

  
    #6. Save best model and scaler to disk
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    print("\nBest model saved to: models/best_model.joblib")
    print("Scaler saved to: models/scaler.joblib")

    return results_df


if __name__ == "__main__":
    train_models()
    print("\nModel training and evaluation completed.")
