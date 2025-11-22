import pandas as pd
import os

def load_iris(csv_path=None):
    # Si no se pasa una ruta, usar la del proyecto
    if csv_path is None:
        csv_path = os.path.join("data", "Iris.csv")

    # Leer el CSV tal cual
    df = pd.read_csv(csv_path)
    return df
