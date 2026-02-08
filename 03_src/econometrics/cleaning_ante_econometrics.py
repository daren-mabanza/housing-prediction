from pathlib import Path
import great_expectations as gx
from cheat_tools.data_quality import initialisation_gx, afficher_resultats_validation
from cheat_tools.query_cleaning_post_eda import requete_nettoyage_post_eda
import pandas as pd
import numpy as np
import duckdb

chemin_repo = Path(r"C:\Users\user\Desktop\data_science_documents\data_science_projets_perso\projet_housing_prediction\housing-prediction")


def nettoyage_post_eda():

    # Import | copie de la base de données | selection des features d'interet
    data_housing = pd.read_csv(f"{chemin_repo}/01_data/02_processed/01_merge/housing_open_data_processed.csv",
                           dtype={'id_annonce':"object"})
    
    df = data_housing.copy()

    print("Import de la base de données : OK")
    print("="*50)

    colonnes_retenus = ["id_annonce","code_insee",
                        "approximate_latitude","approximate_longitude",
                        "size","nb_rooms","nb_photos","population",
                        "nb_menages_fiscaux","revenu_median","nb_actes_delinquants",
                        "nb_lits_tourisme","densite_services_rayon",
                        "type_rural_urbain","position_commune_unite_urbaine",
                        "property_type","type_degre_densite","type_unite_urbaine",
                        "taille_pole_et_couronne","taille_agglomeration",
                        "target"]

    df = df[colonnes_retenus]

    print("Selection des colonnes d'interet : OK")
    print("="*50)

    stats_nbrooms = (data_housing
    .groupby(["property_type"])["nb_rooms"]
    .agg(
     min_nbrooms=lambda x: x.min(),
     q1_nbrooms=lambda x: x.quantile(0.25),
     mediane_nbrooms=lambda x: x.median(),
     moyenne_nbrooms=lambda x: x.mean(),
     q3_nbrooms=lambda x: x.quantile(0.75),
     max_nbrooms=lambda x: x.max()
    )
    ).reset_index()

    stats_size = (data_housing
    .groupby(["property_type"])["size"]
    .agg(
     min_size=lambda x: x.min(),
     q1_size=lambda x: x.quantile(0.25),
     mediane_size=lambda x: x.median(),
     moyenne_size=lambda x: x.mean(),
     q3_size=lambda x: x.quantile(0.75),
     max_size=lambda x: x.max()
    )
    ).reset_index()

    print("Statistiques descriptives pour 'size' et 'nb_rooms' par modalité de 'property_type' : OK")
    print("="*50)

    
    # Nettoyage des valeurs incohérentes pour "size" et "property_type" | selection des colonnes d'interet et renommage des variables (size_corrige --> size | nb_rooms_corrige --> nb_rooms)
    df["property_type"] = np.where(df["property_type"]=="viager","appartement",df["property_type"])

    df = df[~df["property_type"].isin(["divers", "hôtel", "parking", "terrain", "terrain à bâtir"])]

    print("Passage de viager en appartement + suppression de la modalité divers : OK")
    print("="*50)

    con = duckdb.connect()
    con.register("df", df)
    con.register("stats_nbrooms",stats_nbrooms)
    con.register("stats_size",stats_size)

    df = con.sql(requete_nettoyage_post_eda()).df()

    con.close()

    colonnes_retenus = ["id_annonce","code_insee",
                    "approximate_latitude","approximate_longitude",
                    "size_corrige","nb_rooms_corrige","nb_photos","population",
                    "nb_menages_fiscaux","revenu_median","nb_actes_delinquants",
                    "nb_lits_tourisme","densite_services_rayon",
                    "type_rural_urbain","position_commune_unite_urbaine",
                    "property_type","type_degre_densite","type_unite_urbaine",
                    "taille_pole_et_couronne","taille_agglomeration",
                    "target"]
    
    print("Requete SQL : OK")
    print("="*50)

    df = df[colonnes_retenus].rename(columns={"size_corrige":"size","nb_rooms_corrige":"nb_rooms"})

    print("Renommage des colonnes 'size' et 'nb_rooms' : OK")
    print("="*50)

    df["property_type"] = np.where(
        df["property_type"].isin(["appartement","maison"]),
        df["property_type"],
        "autre"
    )

    print("Passage à 3 modalités sur 'property_type' (appartement, maison, autre)' : OK")
    print("="*50)

    # Controle de la qualité des données 
    context = gx.get_context()

        # Paramétrage du batch
    batch_request = initialisation_gx(
    context=context,
    dataframe=df,
    ds_name="housing_project",
    asset_name="housing_post_merge"
    )

    validator = context.get_validator(batch_request=batch_request)
    
    print("(Great Exepectations) Paramétrage du batch : OK")
    print("="*50)

        # Paramétrage des conditions
    validator.expect_column_stdev_to_be_between('size', 15, 150)
    validator.expect_column_stdev_to_be_between('nb_rooms', 1.75, 4)
    validator.expect_column_values_to_be_in_set('property_type', ['appartement', 'maison','autre'])
    validator.expect_compound_columns_to_be_unique(['approximate_latitude', 'approximate_longitude'])
    validator.expect_column_to_exist('target')
    validator.expect_column_values_to_not_be_null('target')
    validator.expect_table_column_count_to_equal(21)
    validator.expect_column_value_lengths_to_be_between('code_insee', 5, 5)
    validator.expect_column_value_lengths_to_be_between('id_annonce', 5, 5)
    validator.expect_column_values_to_be_unique('id_annonce')

    print("(Great Exepectations) Paramétrage des conditions : OK")
    print("="*50)

        # Check de la validité des conditions
    results = validator.validate()

    afficher_resultats_validation(results)

    if results.success == True:
        df.to_csv(f"{chemin_repo}/01_data/02_processed/02_eda/housing_data_eda_processed.csv", index = False)
        print("Check de la validité des données : OK")
        print("Export de la table : OK")
        print("Fin de la pipeline !")

    else:
        print("Données non valides !")


    return df 


