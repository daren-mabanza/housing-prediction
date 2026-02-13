from merge.dataset import fusion_donnees_externes
from econometrics.cleaning_ante_econometrics import nettoyage_post_eda
from econometrics.cleaning_post_econometrics import data_post_econometrie
from model.logistic_regression import pipeline_model

def full_data_housing_project():
    """
    Exécute la pipeline complète :
    Merge Open Data -> Cleaning post-EDA -> Feature Eng post-économétrie -> Modèle ML
    """

    # Fusion avec les données de l'open data
    step_1 = fusion_donnees_externes()

    print("Fusion avec les données de l'open data : OK")
    print("="*50)

    # Nettoyage post EDA
    step_2 = nettoyage_post_eda()

    print("Nettoyage post EDA : OK")
    print("="*50)

    # Nettoyage post ECONOMETRIE
    step_3 = data_post_econometrie()

    print("Nettoyage post ECONOMETRIE : OK")
    print("="*50)

    # Pipeline du modèle
    pipeline_model()

    print("Pipeline du modèle : OK")
    print("="*50)



