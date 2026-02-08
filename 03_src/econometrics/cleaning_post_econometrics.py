from pathlib import Path
import great_expectations as gx
from cheat_tools.data_quality import initialisation_gx, afficher_resultats_validation
from cheat_tools.econometrics_tools import add_centered_vars, add_centered_quadratic
import pandas as pd
import numpy as np

chemin_repo = Path(r"C:\Users\user\Desktop\data_science_documents\data_science_projets_perso\projet_housing_prediction\housing-prediction")
# RAJOUTER DES LOGS

def data_post_econometrie():

    # Import des données 

    data_housing = pd.read_csv(f"{chemin_repo}/01_data/02_processed/02_eda/housing_data_eda_processed.csv",
                           dtype={'id_annonce':"object"})
    
    df = data_housing.copy()


    # Feature Engineering

        # Taux de criminalté pour 1000 habitants
    df = df.eval('taux_criminalite_1000 = (nb_actes_delinquants/population)*1000')
    df["taux_criminalite_1000"] = df["taux_criminalite_1000"].clip(0,300)


        # Taille moyenne d'un ménage fiscal (cette variable ne sera pas conservé car peu de pouvoir discrimant)
    df = df.eval('taille_moyenne_menage = population / nb_menages_fiscaux')

        # Taux de pression touristique pour 1000 habitants
    df = df.eval('taux_pression_touristique_1000 = (nb_lits_tourisme / population)*1000')
    df["est_attrait_touristique"] = np.where(df["taux_pression_touristique_1000"]>10,1,0)

    
        # Centrage des variables ("size","taux_criminalite_1000","population","nb_photos")
    add_centered_vars(
        df,
        ["size","taux_criminalite_1000","population","nb_photos"]
    )

        # Transformations quadratiques ("size","quadrattaux_criminalite_1000ique","population","nb_photos")
    add_centered_quadratic(
        df,
        ["size","taux_criminalite_1000","population","nb_photos"]
    )

        # Binarisation (nb_rooms)
    conditions = [
        df["nb_rooms"].between(0,2),
        df["nb_rooms"]==3,
        df["nb_rooms"]==4,
        df["nb_rooms"]>4
        ]                                                                       

    resultats = ["0-2","3","4","5+"]

    df["nb_rooms_bins"] = np.select(conditions,resultats,default="NA")

        # Transformation logarithmique (revenu_median)
    df["revenu_median_log"] = np.log(df["revenu_median"])

    
    # Selection finale des variables 

    selection_features = ['id_annonce','code_insee',
                          'property_type','taille_agglomeration','est_attrait_touristique',
                          'nb_rooms_bins','approximate_latitude','approximate_longitude',
                          'densite_services_rayon','size_c','taux_criminalite_1000_c',
                          'population_c','nb_photos_c','size_2_c','taux_criminalite_1000_2_c',
                          'population_2_c','nb_photos_2_c','revenu_median_log','target']


    df = df[selection_features].copy()


    # Controle de la validité des données (Great Expectations)

        # Paramétrage du batch
    context = gx.get_context()

    batch_request = initialisation_gx(
        context=context,
        dataframe=df,
        ds_name="housing_project",
        asset_name="data_post_econometrics"
        )

    validator = context.get_validator(batch_request=batch_request)


        # Paramétrage des conditions
    validator.expect_table_columns_to_match_set(selection_features)
    validator.expect_column_values_to_not_be_null('target')
    validator.expect_compound_columns_to_be_unique(['approximate_latitude', 'approximate_longitude'])
    validator.expect_column_values_to_be_unique('id_annonce')
    

        # Validation des résultats et export de la table sous condition de validité des données
    results = validator.validate()

    afficher_resultats_validation(results)


    if results.success == True:
        df.to_csv(f"{chemin_repo}/01_data/02_processed/03_econometrics/housing_data_econometrics_processed.csv", index = False)
        print("Check de la validité des données : OK")
        print("Export de la table : OK")

    else:
        print("Données non valides !")


    return df
    