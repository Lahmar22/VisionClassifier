import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def explorationPreparation():
    datasetPath = "data/Intel_Image_Classification/data_classifier"
    totalImage = 0
    data = []
    for classe in os.listdir(datasetPath):
            chemin = os.path.join(datasetPath, classe)
            if os.path.isdir(chemin):
                images = os.listdir(chemin)
                totalImage += len(images)
    
    for classe in os.listdir(datasetPath):
        chemin = os.path.join(datasetPath, classe)
        if os.path.isdir(chemin):
            images = os.listdir(chemin)
            prc = (len(images) / totalImage) * 100
            # print(f"{classe} {len(images)} images {prc:.2f}% du dataset")
            data.append({"classe": classe,
                        "porcentage": prc})
    return data
     

def visualisation():
    data = explorationPreparation()
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="classe",
        y="porcentage"
    )

    plt.title("Pourcentage par classe")
    plt.xlabel("Classe")
    plt.ylabel("Pourcentage (%)")

    plt.show()

visualisation()