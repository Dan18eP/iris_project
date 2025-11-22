# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_iris

def run_eda():
    df = load_iris()

    print("\n** HEAD **")
    print(df.head())

    print("\n** INFO **")
    print(df.info())

    print("\n** SUMMARY **")
    print(df.describe())

    print("\n** CLASS BALANCE **")
    print(df['Species'].value_counts())
    
    print("\n** MISSING VALUES **")
    print(df.isnull().sum())
    
    print("\n** DUPLICATE ROWS **")
    print(df.duplicated().sum())
    
    
    #HISTOGRAMS
 
    features = ['SepalLengthCm', 'SepalWidthCm', 
                'PetalLengthCm', 'PetalWidthCm']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.histplot(data=df, x=feat, hue='Species', kde=True, ax=axes[i])
        axes[i].set_title(f"Histograma de {feat}")
    plt.tight_layout()
    plt.show()

    #SCATTER MATRIX
  
    sns.pairplot(df, hue='Species', diag_kind='hist')
    plt.suptitle("Pairplot Iris")
    plt.show()

  
    #CORRELATION HEATMAP 

    plt.figure(figsize=(7,5))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='Blues')
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()


    #BOXPLOTS
   
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.boxplot(data=df, x='Species', y=feat, ax=axes[i])
        axes[i].set_title(f"Boxplot de {feat} por especie")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_eda()
