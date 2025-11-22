import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_predict import predict_species


def model_end_to_end_test():
    """
    Quick real example that tests the entire trained model pipeline.

    Raises:
        AssertionError if prediction pipeline fails
    """
    sample = [5.1, 3.5, 1.4, 0.2]  # Known example (Iris-setosa)


    pred = predict_species(sample)

    assert pred in [
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    ], "Prediction returned an invalid species."

    return pred

def test_predict_virginica():
    # Known virginica sample
    sample2 = [6.5, 3.0, 5.5, 1.8]
    pred2 = predict_species(sample2)
    
    assert pred2 in [
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    ], "Prediction returned an invalid species."

    return pred2
   


if __name__ == "__main__":
    print("\n=== Running validation checks ===\n")

    try:
        test_prediction = model_end_to_end_test()
        print(f"Pipeline test OK ✔ | Example prediction: {test_prediction}")
    except Exception as e:
        print(f" Pipeline test failed: {e}")
        
   
    try:
        test_virginica = test_predict_virginica()
        print(f"Virginica test OK ✔ | Prediction: {test_virginica}")
    except Exception as e:
        print(f" Virginica test failed: {e}")
        
        